import os
import sys
import argparse
import matplotlib.pyplot as plt
from env import TrafficEnv
from utils import get_traffic_metrics
import libsumo as traci

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", type=str, required=True, help="Path to the SUMO config file (*.sumocfg)")
parser.add_argument("-R", "--render", action="store_true", help="Render the simulation with GUI")
parser.add_argument("-o", "--output-dir", type=str, default="results", help="Directory for output files")
args = parser.parse_args()


def plot_performance(metrics_dict, output_dir, title_prefix):
    plt.style.use('ggplot')

    # Individual plots for selected metrics
    for metric_name, values in metrics_dict.items():
        if metric_name not in ["emergency_travel_time", "normal_travel_time"]:
            plt.figure(figsize=(10.8, 7.2), dpi=120)
            plt.plot(values)
            plt.xlabel('# of Episodes')
            plt.ylabel(metric_name.replace('_', ' ').title())
            plt.title(f"{title_prefix} - {metric_name.replace('_', ' ').title()}")
            plt.savefig(os.path.join(output_dir, f"{title_prefix.lower().replace(' ', '_')}_{metric_name}.png"))
            plt.close()

    # Comparison plot: Emergency vs Normal Travel Time
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

    n_episode = 1  # Number of episodes to run for testing
    metrics = {
        "average_travel_time": [],
        "emergency_waiting_time": [],
        "emergency_travel_time": [],
        "normal_travel_time": []
    }
    episode_durations = []
    timeout = 100000  # Set a reasonable timeout (in steps) for each episode

    try:
        for episode in range(n_episode):
            state = env.reset()
            done = False
            step_count = 0

            print(f"\nStarting Episode {episode + 1} (SUMO Default Behavior)")

            while not done:
                # Simply step the simulation without applying any actions
                # SUMO will use its default traffic light logic
                for _ in range(env.decision_time):
                    traci.simulationStep()
                    env.time += 1
                    step_count += 1

                state = env.get_state()  # Update state for emergency detection if needed
                done = env.get_done() or (env.time >= timeout)

            duration = env.time
            episode_durations.append(duration)

            env.close()
            # Collect metrics from the tripinfo file
            avg_travel_time, avg_emergency_waiting, avg_emergency_travel, avg_normal_travel = get_traffic_metrics(
                tripinfo_file)
            metrics["average_travel_time"].append(avg_travel_time)
            metrics["emergency_waiting_time"].append(avg_emergency_waiting)
            metrics["emergency_travel_time"].append(avg_emergency_travel)
            metrics["normal_travel_time"].append(avg_normal_travel)

            avg_travel_time, avg_emergency_waiting, avg_emergency_travel, avg_normal_travel = get_traffic_metrics(
                tripinfo_file)
            avg_emergency_speed = env.get_avg_emergency_speed()  # Lấy tốc độ trung bình của xe emergency
            print(f"Traffic Metrics:")
            print(f"  Average Travel Time: {avg_travel_time:.2f}")
            print(f"  Emergency Waiting Time: {avg_emergency_waiting:.2f}")
            print(f"  Emergency Travel Time: {avg_emergency_travel:.2f}")
            print(f"  Normal Travel Time: {avg_normal_travel:.2f}")
            print(f"  Average Emergency Speed: {avg_emergency_speed:.2f}")

            print(f"Episode {episode + 1} Completed: "
                  f"Avg Travel Time: {avg_travel_time:.2f}, "
                  f"Emergency Waiting Time: {avg_emergency_waiting:.2f}, "
                  f"Emergency Travel Time: {avg_emergency_travel:.2f}, "
                  f"Normal Travel Time: {avg_normal_travel:.2f}, ",
                  f"Duration: {duration}")

    except KeyboardInterrupt:
        print("\nKeyboard Interrupt detected! Plotting current performance...")
        plot_performance(metrics, args.output_dir, "Interim Performance of SUMO Default")
        env.close()
        sys.exit()

    # Plot final performance
    plot_performance(metrics, args.output_dir, "Performance of SUMO Default")
    print("Simulation completed.")