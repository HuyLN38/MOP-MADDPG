import os
import sys
import argparse
import libsumo as traci
from env import TrafficEnv

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", type=str, required=True, help="Path to the SUMO config file (*.sumocfg)")
parser.add_argument("-R", "--render", action="store_true", help="Render the simulation with GUI")
parser.add_argument("-o", "--output-dir", type=str, default="results", help="Directory for output files")
parser.add_argument("-s", "--scale", type=str, default="1", help="Tripinfo file name")
args = parser.parse_args()

if __name__ == "__main__":
    if 'SUMO_HOME' not in os.environ:
        sys.exit("Please declare environment variable 'SUMO_HOME'")
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)

    os.makedirs(args.output_dir, exist_ok=True)
    tripinfo_file = os.path.join(args.output_dir, "tripinfo.xml")

    # Initialize environment
    print("=== Initializing Traffic Environment ===")
    env = TrafficEnv(args.config, mode="gui" if args.render else "binary", tripinfo_file=tripinfo_file)
    initial_state = env.reset()

    # Get dimensions and initial details
    state_dim = env.get_state_dim()
    action_dim = env.get_action_dim()
    n_agents = len(traci.trafficlight.getIDList())

    print("=========================\n")
    print("=========================\n")
    print("=========================\n")
    print("=========================\n")

    print("Nearest Traffic of each Light IDs:\n")
    # print(env.nearest_neighbors)

    print("=========================\n")
    print("=========================\n")
    print("=========================\n")
    print("=========================\n")

    # Print initialization details
    print(f"Number of Agents (Traffic Lights): {n_agents}")
    print(f"State Space Dimensions (per agent): {state_dim}")
    print(f"Action Space Dimension: {action_dim}")
    print("Initial State Set:")
    for i, state in enumerate(initial_state):
        print(f"  Agent {i + 1}: {state}\n")
    print(f"Phase Counts per Intersection: {env.phase_counts}")
    print(f"Initial Waiting Times: {env.prev_waiting_times}")
    print(f"Initial Queue Lengths: {env.prev_queue_lengths}")
    print(f"Emergency Detected: {env.emergency_detected}")

    # Run the simulation until 3382 seconds
    print("\n=== Running Simulation to 3382 Seconds ===")
    target_time = 3382  # Target simulation time in seconds
    state = initial_state
    done = False
    while env.time < target_time and not done:
        # Use the current phase as the action to maintain default traffic light behavior
        actions = [traci.trafficlight.getPhase(tl_id) for tl_id in traci.trafficlight.getIDList()]

        # Step the environment
        state, reward_normal, reward_emergency, done = env.step(actions, timeout=float(
            'inf'))  # Set timeout to infinity to avoid early termination

    # Collect and print information at 3382 seconds
    print(f"\n=== Simulation State at {env.time} Seconds ===")
    print(f"Current State Set:")
    for i, s in enumerate(state):
        print(f"  Agent {i + 1}: {s}")
    print(f"Current Waiting Times: {env.prev_waiting_times}")
    print(f"Current Queue Lengths: {env.prev_queue_lengths}")
    print(f"Emergency Detected: {env.emergency_detected}")
    print(f"Episode Duration: {env.get_episode_duration()} seconds")
    print(f"Normal Rewards: {reward_normal}")
    print(f"Emergency Rewards: {reward_emergency}")

    # Close the environment
    env.close()
    print("\n=== Simulation Complete ===")