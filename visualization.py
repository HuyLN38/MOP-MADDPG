import os
import sys
import numpy as np
import torch
import time
import traci
import matplotlib.pyplot as plt
from io import BytesIO
import base64

from env import TrafficEnv
from maddpg import MADDPG
from utils import get_traffic_metrics


def plot_metrics(vehicle_counts_regular, vehicle_counts_emergency, normal_steps,
                 emergency_steps, intersection_vehicle_counts, output_dir):
    """
    Generate visualization plots for vehicle counts, activation times, intersection heatmap,
    and reward equation description.
    Returns an HTML string with embedded plots.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Traffic Simulation Metrics", fontsize=16)

    # Plot 1: Vehicle counts
    steps = range(len(vehicle_counts_regular))
    axes[0, 0].plot(steps, vehicle_counts_regular, label="Regular Vehicles", color="green")
    axes[0, 0].plot(steps, vehicle_counts_emergency, label="Emergency Vehicles", color="orange")
    axes[0, 0].set_title("Number of Vehicles Over Time")
    axes[0, 0].set_xlabel("Step")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Plot 2: Activation times as pie chart
    # Multiply normal_steps by 10 to account for 10 simulation steps per model step
    activation_times = [normal_steps * 10, emergency_steps]
    labels = ["Normal Mode", "Emergency Mode"]
    colors = ["blue", "red"]
    axes[0, 1].pie(activation_times, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
    axes[0, 1].set_title("Activation Time of Normal vs Emergency Modes")
    axes[0, 1].axis("equal")  # Equal aspect ratio ensures pie is circular

    # Plot 3: Heatmap of vehicles per intersection, excluding zero counts
    # Filter out intersections with zero vehicle counts
    filtered_intersection_counts = {k: v for k, v in intersection_vehicle_counts.items() if v > 1000}
    intersection_ids = list(filtered_intersection_counts.keys())
    vehicle_counts = list(filtered_intersection_counts.values())

    if len(intersection_ids) > 0 and sum(vehicle_counts) > 0:
        heatmap_data = np.array(vehicle_counts).reshape(1, -1)
        cax = axes[1, 0].imshow(heatmap_data, cmap="hot", interpolation="nearest")
        fig.colorbar(cax, ax=axes[1, 0], label="Vehicle Count")
        axes[1, 0].set_xticks(range(len(intersection_ids)))
        axes[1, 0].set_xticklabels(intersection_ids, rotation=45)
        axes[1, 0].set_yticks([])
        axes[1, 0].set_title("Heatmap of Vehicles per Intersection (Non-Zero Counts)")
    else:
        axes[1, 0].text(0.5, 0.5, "No intersections with non-zero vehicle counts",
                        horizontalalignment="center", verticalalignment="center")
        axes[1, 0].set_title("Heatmap of Vehicles per Intersection")

    # Plot 4: Reward equation description
    reward_text = (
        "Reward Equations:\n"
        "Normal Mode (per traffic light i):\n"
        "r_normal,i = -0.5 * queue_length_i - 0.3 * waiting_time_i + 0.2 * throughput_i\n\n"
        "Emergency Mode (per traffic light i):\n"
        "r_emergency,i = -0.6 * emergency_waiting_time_i + 0.3 * emergency_speed_i - 0.1 * normal_waiting_time_i\n\n"
        "Total Rewards:\n"
        "reward_normal = sum(r_normal,i)\n"
        "reward_emergency = sum(r_emergency,i)"
    )
    axes[1, 1].text(0.5, 0.5, reward_text, horizontalalignment="center", verticalalignment="center",
                    wrap=True, fontsize=10)
    axes[1, 1].set_title("Reward Equations")
    axes[1, 1].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save plot to file
    plot_file = os.path.join(output_dir, "simulation_metrics.png")
    plt.savefig(plot_file)
    plt.close()

    # Embed plot in HTML
    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    encoded_image = base64.b64encode(image_png).decode("utf-8")
    html_str = f'<img src="data:image/png;base64,{encoded_image}" alt="Simulation Metrics"/>'
    return html_str


if __name__ == "__main__":
    # Check SUMO_HOME environment variable
    if "SUMO_HOME" not in os.environ:
        sys.exit("Please declare environment variable 'SUMO_HOME'")
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)

    # Configuration
    config_file = "2025-04-09-10-48-15/osm.sumocfg"
    output_dir = "results"
    tripinfo_file = os.path.join(output_dir, "tripinfo.xml")
    render = True
    model_name = "Full - Scenerio No Emergency"
    num_runs = 1
    use_model = True

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # File to save metrics
    metrics_file = os.path.join(output_dir, f"{model_name}_metrics.txt")

    # Initialize metrics storage
    all_metrics = {
        "avg_travel_time": [],
        "avg_emergency_waiting": [],
        "avg_emergency_travel": [],
        "avg_normal_travel": [],
        "avg_normal_waiting": [],
        "avg_emergency_speed": [],
        "avg_non_emergency_speed": []
    }

    print("Starting Multiple Simulation Runs")

    for run in range(num_runs):
        print(f"\nStarting Run {run + 1}/{num_runs}")

        # Set seed for this run
        seed = str(run + 1)
        env = TrafficEnv(
            config_file,
            mode="gui" if render else "binary",
            tripinfo_file=tripinfo_file,
            seed=seed,
        )

        # Reset environment
        state = env.reset()
        state_dims = env.get_state_dim()
        action_dim = env.get_action_dim()
        n_agents = len(state)

        # Initialize and load agents
        normal_agent = MADDPG(n_agents, state_dims, action_dim, lr=1e-3, tau=0.005, gamma=0.99, batch_size=128)
        emergency_agent = MADDPG(n_agents, state_dims, action_dim, lr=1e-4, tau=0.001, gamma=0.99, batch_size=256)
        normal_agent.load_model(os.path.join(output_dir, "normal_agent_full_old.th"))
        emergency_agent.load_model(os.path.join(output_dir, "normal_agent_full_old.th"))
        normal_agent.eps = 0.0
        emergency_agent.eps = 0.0

        # Initialize data collection
        vehicle_counts_regular = []
        vehicle_counts_emergency = []
        intersection_vehicle_counts = {}
        reward_epi_normal = 0.0
        reward_epi_emergency = 0.0
        actions = [None for _ in range(n_agents)]
        action_probs = [None for _ in range(n_agents)]
        done = False
        step_count = 0
        timeout = 99999999999999999999
        action_selection_times = []
        emergency_step = 0
        normal_step = 0

        while not done:
            # Collect vehicle counts
            try:
                regular_count, emergency_count = env.get_vehicle_counts()
            except AttributeError:
                print("Warning: get_vehicle_counts not implemented in TrafficEnv")
                regular_count, emergency_count = 0, 0
            vehicle_counts_regular.append(regular_count)
            vehicle_counts_emergency.append(emergency_count)

            # Select actions
            if use_model:
                action_selection_time = 0.0
                if env.emergency_detected:
                    for i in range(n_agents):
                        start_time = time.time()
                        action, action_prob = normal_agent.select_action(state[i], i)
                        end_time = time.time()
                        action_selection_time += (end_time - start_time)
                        actions[i] = action
                        action_probs[i] = action_prob
                    emergency_step += 1
                else:
                    for i in range(n_agents):
                        start_time = time.time()
                        action, action_prob = normal_agent.select_action(state[i], i)
                        end_time = time.time()
                        action_selection_time += (end_time - start_time)
                        actions[i] = action
                        action_probs[i] = action_prob
                    normal_step += 1

                action_selection_times.append(action_selection_time)

                # Step environment
                state, reward_normal, reward_emergency, done = env.step(actions, timeout)
                reward_epi_normal += sum(reward_normal)
                reward_epi_emergency += sum(reward_emergency)
                step_count += 1
            else:
                state, reward_normal, reward_emergency, done = env.normal_step()
                env.time += 1
                step_count += 1

            # Collect intersection vehicle counts
            try:
                intersection_counts = env.get_intersection_vehicle_counts()
                for intersection_id, count in intersection_counts.items():
                    intersection_vehicle_counts[intersection_id] = (
                            intersection_vehicle_counts.get(intersection_id, 0) + count
                    )
            except AttributeError:
                print("Warning: get_intersection_vehicle_counts not implemented in TrafficEnv")
                intersection_vehicle_counts = {}

            if step_count % 100 == 0:
                print(
                    f"Step {step_count}: Normal Reward: {reward_epi_normal:.2f}, "
                    f"Emergency Reward: {reward_epi_emergency:.2f}"
                )

        # Print run statistics
        avg_action_selection_time = np.mean(action_selection_times) if action_selection_times else 0.0
        print(f"Run {run + 1} Average Model Action Selection Time: {avg_action_selection_time:.6f} seconds")
        print(f"Run {run + 1} Normal Steps: {normal_step}, Emergency Steps: {emergency_step}")

        # Close environment
        env.close()

        # Get and store metrics
        avg_travel_time, avg_emergency_waiting, avg_emergency_travel, avg_normal_travel, avg_normal_waiting = get_traffic_metrics(
            tripinfo_file
        )
        avg_emergency_speed = env.get_avg_emergency_speed()
        avg_non_emergency_speed = env.get_avg_non_emergency_speed()

        all_metrics["avg_travel_time"].append(avg_travel_time)
        all_metrics["avg_emergency_waiting"].append(avg_emergency_waiting)
        all_metrics["avg_emergency_travel"].append(avg_emergency_travel)
        all_metrics["avg_normal_travel"].append(avg_normal_travel)
        all_metrics["avg_normal_waiting"].append(avg_normal_waiting)
        all_metrics["avg_emergency_speed"].append(avg_emergency_speed)
        all_metrics["avg_non_emergency_speed"].append(avg_non_emergency_speed)

        print(
            f"Run {run + 1} Completed: Total Normal Reward: {reward_epi_normal:.2f}, "
            f"Total Emergency Reward: {reward_epi_emergency:.2f}"
        )
        print(f"Traffic Metrics for Run {run + 1}:")
        print(f"  Average Travel Time: {avg_travel_time:.2f}")
        print(f"  Emergency Waiting Time: {avg_emergency_waiting:.2f}")
        print(f"  Emergency Travel Time: {avg_emergency_travel:.2f}")
        print(f"  Normal Travel Time: {avg_normal_travel:.2f}")
        print(f"  Normal Waiting Time: {avg_normal_waiting:.2f}")
        print(f"  Average Emergency Speed: {avg_emergency_speed:.2f}")
        print(f"  Average Non-Emergency Speed: {avg_non_emergency_speed:.2f}")

        # Generate visualizations
        html_content = plot_metrics(
            vehicle_counts_regular,
            vehicle_counts_emergency,
            normal_step,
            emergency_step,
            intersection_vehicle_counts,
            output_dir
        )

        # Save HTML visualization
        html_file = os.path.join(output_dir, "simulation_metrics.html")
        with open(html_file, "w") as f:
            f.write("<html><body>")
            f.write("<h2>Traffic Simulation Visualization</h2>")
            f.write(html_content)
            f.write("</body></html>")
        print(f"Visualization saved to {html_file}")

    # Save average metrics
    with open(metrics_file, "w") as f:
        f.write(f"Traffic Metrics for {model_name} (Average of {num_runs} runs):\n")
        for key, values in all_metrics.items():
            f.write(
                f"  {key.replace('_', ' ').title()}: {np.mean(values):.2f} ± {np.std(values):.2f}\n"
            )
        f.write("\nIndividual Run Data:\n")
        for run in range(num_runs):
            f.write(f"\nRun {run + 1}:\n")
            for key, values in all_metrics.items():
                f.write(f"  {key.replace('_', ' ').title()}: {values[run]:.2f}\n")

    print(f"\nAll metrics have been saved to {metrics_file}")