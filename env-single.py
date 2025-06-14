import math

import libsumo as traci
import sumolib
from sumolib import checkBinary
import numpy as np

# Cell 2: TrafficEnv Class
class TrafficEnv:
    def __init__(self, config_file, mode='binary', tripinfo_file="results/tripinfo.xml", seed="156"):
        self.config_file = config_file
        self.tripinfo_file = tripinfo_file
        self.sumoBinary = checkBinary('sumo-gui') if mode == 'gui' else checkBinary('sumo')
        self.sumoCmd = [
            self.sumoBinary, "-c", self.config_file, '--no-step-log', '-W',
            '--scale', '1', '--seed', seed, '--tripinfo-output', self.tripinfo_file
        ]
        self.time = 0
        self.decision_time = 10
        self.n_intersections = None
        self.phase_counts = {}
        self.prev_waiting_times = {}
        self.prev_queue_lengths = {}
        self.emergency_detected = False
        self.total_emergency_speed = 0.0
        self.emergency_speed_count = 0
        self.total_non_emergency_speed = 0.0
        self.non_emergency_speed_count = 0
        self.coordinates = {}
        self.nearest_neighbors = {}

    def reset(self):
        traci.start(self.sumoCmd)
        traci.simulationStep()
        self.time = 0
        self.n_intersections = len(traci.trafficlight.getIDList())

        # Load the network
        net = sumolib.net.readNet(self.config_file.replace('.sumocfg', '.net.xml'))

        # Store coordinates for each traffic light
        self.coordinates = {}
        tl_ids = traci.trafficlight.getIDList()
        for intersection_ID in tl_ids:
            # Get the junctions controlled by this traffic light
            controlled_junctions = traci.trafficlight.getControlledJunctions(intersection_ID)
            if controlled_junctions:
                # Use the first controlled junction as the representative
                junction_id = controlled_junctions[0]
                try:
                    junction = net.getNode(junction_id)
                    x, y = junction.getCoord()
                    self.coordinates[intersection_ID] = (x, y)
                except KeyError:
                    print(f"Warning: Junction {junction_id} not found for traffic light {intersection_ID}")
                    # Fallback to average coordinates of controlled junctions or (0, 0)
                    self.coordinates[intersection_ID] = (0, 0)
            else:
                print(f"Warning: No controlled junctions for traffic light {intersection_ID}")
                self.coordinates[intersection_ID] = (0, 0)

            # Get phase counts
            logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(intersection_ID)[0]
            self.phase_counts[intersection_ID] = len(logic.phases)

        # Calculate two nearest neighbors for each intersection
        self.nearest_neighbors = {}
        for intersection_ID in tl_ids:
            distances = []
            x1, y1 = self.coordinates[intersection_ID]
            for other_ID in tl_ids:
                if other_ID != intersection_ID:
                    x2, y2 = self.coordinates[other_ID]
                    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                    distances.append((other_ID, distance))
            # Sort by distance and take top 2
            distances.sort(key=lambda x: x[1])
            self.nearest_neighbors[intersection_ID] = [dist[0] for dist in distances[:2]]

        self.prev_waiting_times = {tl_id: self._get_total_waiting_time(tl_id) for tl_id in tl_ids}
        self.prev_queue_lengths = {tl_id: self._get_queue_length(tl_id) for tl_id in tl_ids}
        return self.get_state()

    def _get_queue_length(self, intersection_ID):
        total_queue = 0
        for lane_id in traci.trafficlight.getControlledLanes(intersection_ID):
            total_queue += traci.lane.getLastStepHaltingNumber(lane_id)
        return total_queue

    def _get_total_waiting_time(self, intersection_ID):
        total_waiting_time = 0
        for lane_id in traci.trafficlight.getControlledLanes(intersection_ID):
            vehicle_list = traci.lane.getLastStepVehicleIDs(lane_id)
            for vehicle_id in vehicle_list:
                total_waiting_time += traci.vehicle.getWaitingTime(vehicle_id)
        return total_waiting_time

    def _get_average_speed(self, intersection_ID):
        total_speed = 0
        count = 0
        for lane_id in traci.trafficlight.getControlledLanes(intersection_ID):
            vehicle_list = traci.lane.getLastStepVehicleIDs(lane_id)
            for vehicle_id in vehicle_list:
                total_speed += traci.vehicle.getSpeed(vehicle_id)
                count += 1
        return total_speed / max(1, count)


    def _get_emergency_metrics(self, intersection_ID):
            emergency_waiting = 0
            emergency_count = 0
            emergency_speed = 0
            for lane_id in traci.trafficlight.getControlledLanes(intersection_ID):
                vehicle_list = traci.lane.getLastStepVehicleIDs(lane_id)
                for vehicle_id in vehicle_list:
                    vehicle_speed = traci.vehicle.getSpeed(vehicle_id)
                    if traci.vehicle.getTypeID(vehicle_id) == "emergency":
                        emergency_count += 1
                        emergency_waiting += traci.vehicle.getWaitingTime(vehicle_id)
                        emergency_speed += vehicle_speed
                        # Cập nhật tổng tốc độ và số lượng xe emergency có tốc độ > 0
                        if vehicle_speed > 0:
                            self.total_emergency_speed += vehicle_speed
                            self.emergency_speed_count += 1
                    else:
                        # Cập nhật tổng tốc độ và số lượng xe không phải emergency có tốc độ > 0
                        if vehicle_speed > 0:
                            self.total_non_emergency_speed += vehicle_speed
                            self.non_emergency_speed_count += 1
            emergency_speed = emergency_speed / max(1, emergency_count) if emergency_count > 0 else 0
            return emergency_count, emergency_waiting, emergency_speed

        # Hàm lấy tốc độ trung bình của xe emergency
    def get_avg_emergency_speed(self):
            return self.total_emergency_speed / max(1,
                                                    self.emergency_speed_count) if self.emergency_speed_count > 0 else 0.0

        # Hàm lấy tốc độ trung bình của xe không phải emergency
    def get_avg_non_emergency_speed(self):
            return self.total_non_emergency_speed / max(1,
                                                        self.non_emergency_speed_count) if self.non_emergency_speed_count > 0 else 0.0

    def get_state(self):
        self.emergency_detected = False
        state = []
        tl_ids = traci.trafficlight.getIDList()
        for i, intersection_ID in enumerate(tl_ids):
            observation = []
            for lane in traci.trafficlight.getControlledLanes(intersection_ID):
                observation.extend([
                    traci.lane.getLastStepVehicleNumber(lane),
                    traci.lane.getLastStepHaltingNumber(lane)
                ])
            emergency_presence = []
            emergency_waiting = 0
            for lane in traci.trafficlight.getControlledLanes(intersection_ID):
                vehicle_list = traci.lane.getLastStepVehicleIDs(lane)
                has_emergency = 0
                for vehicle_id in vehicle_list:
                    if traci.vehicle.getTypeID(vehicle_id) == "emergency":
                        has_emergency = 1
                        self.emergency_detected = True
                        emergency_waiting += traci.vehicle.getWaitingTime(vehicle_id)
                emergency_presence.append(has_emergency)
            observation.extend(emergency_presence)
            observation.append(emergency_waiting / max(1, sum(emergency_presence)))
            n_phases = self.phase_counts[intersection_ID]
            phase = [0] * n_phases
            phase[traci.trafficlight.getPhase(intersection_ID)] = 1
            observation.extend(phase)

            for neighbor_id in tl_ids:
                if neighbor_id != intersection_ID:
                    neighbor_observation = []
                    for lane in traci.trafficlight.getControlledLanes(neighbor_id):
                        neighbor_observation.extend([
                            traci.lane.getLastStepVehicleNumber(lane),
                            traci.lane.getLastStepHaltingNumber(lane)
                        ])
                    observation.extend(neighbor_observation)
            state.append(np.array(observation, dtype=np.float32))
        return state

    def apply_action(self, actions):
        for i, intersection_ID in enumerate(traci.trafficlight.getIDList()):
            current_action = traci.trafficlight.getPhase(intersection_ID)
            if actions[i] == current_action:
                continue
            else:
                n_phases = self.phase_counts[intersection_ID]
                action = int(actions[i]) % n_phases
                traci.trafficlight.setPhase(intersection_ID, action)

    def step(self, actions, timeout):
        if self.emergency_detected == True:
            self.decision_time = 1
        else:
            self.decision_time = 10
        self.apply_action(actions)
        for _ in range(self.decision_time):
            traci.simulationStep()
            self.time += 1
        state = self.get_state()  # Returns list of arrays
        reward_normal, reward_emergency = self.get_reward()
        done = self.get_done()
        if self.time / 10 >= timeout:
            done = True
        return state, reward_normal, reward_emergency, done

    def normal_step(self):
        if self.emergency_detected == True:
            self.decision_time = 1
        else:
            self.decision_time = 10
        for _ in range(self.decision_time):
            traci.simulationStep()
            self.time += 1
        reward_normal, reward_emergency = self.get_reward()
        done = self.get_done()
        state = self.get_state()
        return state, reward_normal, reward_emergency, done

    def get_reward(self):
        reward_normal = []
        for i, intersection_ID in enumerate(traci.trafficlight.getIDList()):
            total_reward = 0.0
            emergency_count, emergency_waiting, emergency_speed = self._get_emergency_metrics(intersection_ID)
            current_queue = self._get_queue_length(intersection_ID)
            current_waiting_time = self._get_total_waiting_time(intersection_ID)
            current_speed = self._get_average_speed(intersection_ID)
            queue_change = self.prev_queue_lengths[intersection_ID] - current_queue
            waiting_time_change = self.prev_waiting_times[intersection_ID] - current_waiting_time
            self.prev_queue_lengths[intersection_ID] = current_queue
            self.prev_waiting_times[intersection_ID] = current_waiting_time

            if emergency_count > 0:
                emergency_waiting_penalty = -5.0 * emergency_waiting
                max_speed = 60.0
                emergency_speed_penalty = -2.0 * (max_speed - emergency_speed) if emergency_speed < max_speed else 0
                queue_penalty = -0.05 * current_queue  # Small penalty to avoid extreme congestion
                total_reward += emergency_waiting_penalty + emergency_speed_penalty + queue_penalty
            else:
                queue_reward = 0.5 * queue_change
                waiting_time_reward = 0.01 * waiting_time_change
                queue_penalty = -0.2 * current_queue
                total_reward +=  queue_reward + waiting_time_reward + queue_penalty

            reward_normal.append(total_reward)

        return np.array(reward_normal), np.array([0.0] * len(reward_normal))

    def get_done(self):
        return traci.simulation.getMinExpectedNumber() == 0

    def close(self):
        traci.close()

    def get_state_dim(self):
        sample_state = self.get_state()
        return [s.shape[0] for s in sample_state]

    def get_action_dim(self):
        return max(self.phase_counts.values()) if self.phase_counts else 2

    def get_episode_duration(self):
        return self.time