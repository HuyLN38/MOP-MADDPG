import os
import sys
import argparse
import matplotlib.pyplot as plt
import libsumo as traci
from env import TrafficEnv
from maddpg import MADDPG
from utils import get_traffic_metrics

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", type=str, required=True, help="Path to the SUMO config file (*.sumocfg)")
parser.add_argument("-R", "--render", action="store_true", help="Render the simulation with GUI")
parser.add_argument("-o", "--output-dir", type=str, default="results", help="Directory for output files")
args = parser.parse_args()


def plot_performance(metrics_dict, output_dir, title_prefix):
    plt.style.use('ggplot')

    # Individual plots for selected metrics
    for metric_name, values in metrics_dict.items():
        if metric_name not in ["emergency_travel_time", "normal_travel_time"]:  # Skip these for individual plots
            plt.figure(figsize=(10.8, 7.2), dpi=120)
            plt.plot(values)
            plt.xlabel('# of Episodes')
            plt.ylabel(metric_name.replace('_', ' ').title())
            plt.title(f"{title_prefix} - {metric_name.replace('_', ' ').title()}")
            plt.savefig(os.path.join(output_dir, f"{title_prefix.lower().replace(' ', '_')}_{metric_name}.png"))
            plt.close()

    # Comparison plot: Emergency Travel Time vs Normal Travel Time
    plt.figure(figsize=(10.8, 7.2), dpi=120)
    plt.plot(metrics_dict["emergency_travel_time"], label="Emergency Travel Time", color='red')
    plt.plot(metrics_dict["normal_travel_time"], label="Normal Travel Time", color='blue')
    plt.xlabel('# of Episodes')
    plt.ylabel('Travel Time (s)')
    plt.title(f"{title_prefix} - Emergency vs Normal Travel Time")
    plt.legend()
    plt.savefig(
        os.path.join(output_dir, f"{title_prefix.lower().replace(' ', '_')}_emergency_vs_normal_travel_time.png"))
    plt.close()


if __name__ == "__main__":
    if 'SUMO_HOME' not in os.environ:
        sys.exit("Please declare environment variable 'SUMO_HOME'")
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)

    os.makedirs(args.output_dir, exist_ok=True)
    tripinfo_file = os.path.join(args.output_dir, "tripinfo.xml")

    # Initialize environment
    env = TrafficEnv(args.config, mode="gui" if args.render else "binary", tripinfo_file=tripinfo_file)
    state = env.reset()
    state_dim = env.get_state_dim()
    action_dim = env.get_action_dim()
    n_agents = len(traci.trafficlight.getIDList())

    # Initialize two MADDPG agents
    normal_agent = MADDPG(n_agents, state_dim, action_dim, lr=1e-4, tau=1e-4, gamma=0.99, batch_size=128)
    emergency_agent = MADDPG(n_agents, state_dim, action_dim, lr=3e-4, tau=2e-3, gamma=0.99, batch_size=128)

    n_episode = 250
    metrics = {
        "average_travel_time": [],
        "emergency_waiting_time": [],
        "emergency_travel_time": [],
        "normal_travel_time": []  # Added to match the four returned values
    }
    episode_durations = []
    fastest_duration = float('inf')
    progress_interval = 50
    timeout = 99999999999999999999
    emergency_step_count = 0  # Counter for emergency agent steps

    try:
        for episode in range(n_episode):
            state = env.reset()
            reward_epi_normal = 0.0
            reward_epi_emergency = 0.0
            actions = [None for _ in range(n_agents)]
            action_probs = [None for _ in range(n_agents)]
            done = False
            step_count = 0

            print(f"\nStarting Episode {episode + 1}")

            while not done:
                active_agent = emergency_agent if env.emergency_detected else normal_agent
                for i in range(n_agents):
                    action, action_prob = active_agent.select_action(state[i], i)  # Pass state[i] directly
                    actions[i] = action
                    action_probs[i] = action_prob
                before_state = state
                state, reward_normal, reward_emergency, done = env.step(actions, timeout)
                reward_epi_normal += sum(reward_normal)
                reward_epi_emergency += sum(reward_emergency)
                step_count += 1
                transition_normal = [before_state, action_probs, state, reward_normal, done]
                transition_emergency = [before_state, action_probs, state, reward_emergency, done]
                normal_agent.push(transition_normal)


                emergency_agent.push(transition_emergency)
                if env.emergency_detected:
                    if emergency_agent.train_start():
                        for i in range(n_agents):
                            emergency_agent.train_model(i)
                        if env.emergency_detected:
                            emergency_step_count += 1
                            if emergency_step_count % 30 == 0:
                                emergency_agent.update_eps_emergency()
                else:
                    if normal_agent.train_start():
                        for i in range(n_agents):
                            normal_agent.train_model(i)
                        normal_agent.update_eps()
                if step_count % (5 * env.decision_time) == 0:
                    print(f"Emergency active: {env.emergency_detected} "
                          f"Episode {episode + 1} - Step {step_count}: "
                          f"Normal Reward: {reward_epi_normal:.2f}, "
                          f"Emergency Reward: {reward_epi_emergency:.2f}, "
                          f"Normal Eps: {normal_agent.eps:.3f}, Emergency Eps: {emergency_agent.eps:.3f}, "
                          f"Emergency Steps: {emergency_step_count}")

                if done:
                    break

            duration = step_count
            episode_durations.append(duration)
            if duration < timeout:
                fastest_duration = duration
                timeout = 2 * fastest_duration

            env.close()
            # Updated unpacking to handle four values
            avg_travel_time, avg_emergency_waiting, avg_emergency_travel, avg_normal_travel = get_traffic_metrics(
                tripinfo_file)
            metrics["average_travel_time"].append(avg_travel_time)
            metrics["emergency_waiting_time"].append(avg_emergency_waiting)
            metrics["emergency_travel_time"].append(avg_emergency_travel)
            metrics["normal_travel_time"].append(avg_normal_travel)

            # agent_type = "Emergency" if env.emergency_detected else "Normal"
            print(f"Episode {episode + 1} Completed: "
                  f"Avg Travel Time: {avg_travel_time:.2f}, "
                  f"Emergency Waiting Time: {avg_emergency_waiting:.2f}, "
                  f"Emergency Travel Time: {avg_emergency_travel:.2f}, "
                  f"Normal Travel Time: {avg_normal_travel:.2f}, "
                  f"Duration: {duration}, Timeout: {timeout or 'None'} "
                  f"Normal Eps: {normal_agent.eps:.3f}, Emergency Eps: {emergency_agent.eps:.3f}, "
                  f"Emergency Steps: {emergency_step_count}")

    except KeyboardInterrupt:
        print("\nKeyboard Interrupt detected! Plotting current performance...")
        plot_performance(metrics, args.output_dir, "Interim Performance of Dual-MADDPG System")
        print("Training resumed. Press Ctrl+C again to stop.")

    # Save models and plot final performance
    normal_agent.save_model(os.path.join(args.output_dir, "normal_agent.th"))
    emergency_agent.save_model(os.path.join(args.output_dir, "emergency_agent.th"))
    plot_performance(metrics, args.output_dir, "Performance of Dual-MADDPG System")
