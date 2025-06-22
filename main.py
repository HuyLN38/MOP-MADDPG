import os
import sys
import argparse
import matplotlib.pyplot as plt
import libsumo as traci
from env import TrafficEnv
from maddpg import MADDPG
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
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
    # Debug: Print configuration details
    print(f'Tripinfo file: {tripinfo_file}')
    print(f'Sumo command: {env.sumoCmd}')

    # Verify GPU availability
    print(f'CUDA Available: {torch.cuda.is_available()}')
    print(f'Device: {torch.device("cuda" if torch.cuda.is_available() else "cpu")}')

    # Reset environment
    try:
        traci.close()
    except:
        pass
    state = env.reset()
    state_dims = env.get_state_dim()
    action_dim = env.get_action_dim()
    n_agents = len(traci.trafficlight.getIDList())

    # Initialize agents
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    normal_agent = MADDPG(n_agents, state_dims, action_dim, lr=1e-3, tau=0.005, gamma=0.99, batch_size=128).to(device)
    emergency_agent = MADDPG(n_agents, state_dims, action_dim, lr=1e-4, tau=0.001, gamma=0.99, batch_size=256).to(device)

    # Pre-Training for Emergency Agent
    pretrain_episodes = 50
    pretrain_metrics = {
        'average_travel_time': [],
        'emergency_waiting_time': [],
        'emergency_travel_time': [],
        'normal_travel_time': []
    }
    pretrain_durations = []
    pretrain_timeout = float('inf')
    emergency_step_count = 0
    best_emergency_waiting_time = float('inf')

    print('\nStarting Emergency Agent Pre-Training')
    for episode in range(pretrain_episodes):
        try:
            traci.close()
        except:
            pass
        state = env.reset()
        reward_epi_emergency = 0.0
        actions = [None for _ in range(n_agents)]
        action_probs = [None for _ in range(n_agents)]
        done = False
        step_count = 0

        print(f'\nPre-Training Episode {episode + 1}')
        env.emergency_detected = True  # Force emergency mode

        while not done:
            for i in range(n_agents):
                action, action_prob = emergency_agent.select_action(state[i], i)
                actions[i] = action
                action_probs[i] = action_prob
            before_state = state
            state, reward_normal, reward_emergency, done = env.step(actions, pretrain_timeout)
            reward_epi_emergency += sum(reward_emergency)
            step_count += 1

            emergency_agent.set_lr(True, episode)
            transition_emergency = [before_state, action_probs, state, reward_emergency, done]
            emergency_agent.push(transition_emergency)
            if emergency_agent.train_start():
                for i in range(n_agents):
                    emergency_agent.train_model(i)
                emergency_step_count += 1
                emergency_agent.update_eps_emergency(episode)

            if step_count % 100 == 0:
                current_lr = emergency_agent.actor_optimizers[0].param_groups[0]['lr']
                print(f'Pre-Training Episode {episode + 1} - Step {step_count}: '
                      f'Emergency Reward: {reward_epi_emergency:.2f}, '
                      f'Emergency Eps: {emergency_agent.eps:.3f}, '
                      f'Current LR: {current_lr:.6f}, '
                      f'Emergency Steps: {emergency_step_count} '
                      f'Interval: {env.decision_time}')
            if done:
                print(f'Total Emergency Reward of Pre-Training Episode {episode + 1} : {reward_epi_emergency:.2f}')
                break

        duration = env.get_episode_duration()
        pretrain_durations.append(duration)
        if duration < pretrain_timeout:
            pretrain_timeout = 2 * duration

        env.close()
        avg_travel_time, avg_emergency_waiting, avg_emergency_travel, avg_normal_travel = get_traffic_metrics(tripinfo_file)
        pretrain_metrics['average_travel_time'].append(avg_travel_time)
        pretrain_metrics['emergency_waiting_time'].append(avg_emergency_waiting)
        pretrain_metrics['emergency_travel_time'].append(avg_emergency_travel)
        pretrain_metrics['normal_travel_time'].append(avg_normal_travel)

        # Save models
        emergency_agent.save_model(os.path.join(args.output_dir, 'emergency_agent_pretrain_last.th'))
        if avg_emergency_waiting < best_emergency_waiting_time:
            best_emergency_waiting_time = avg_emergency_waiting
            emergency_agent.save_model(os.path.join(args.output_dir, 'emergency_agent_pretrain_best.th'))
            print(f'New best model saved with Emergency Waiting Time: {best_emergency_waiting_time:.2f}')

        print(f'Pre-Training Episode {episode + 1} Completed: '
              f'Avg Travel Time: {avg_travel_time:.2f}, '
              f'Emergency Waiting Time: {avg_emergency_waiting:.2f}, '
              f'Emergency Travel Time: {avg_emergency_travel:.2f}, '
              f'Normal Travel Time: {avg_normal_travel:.2f}, '
              f'Duration: {duration}, Timeout: {pretrain_timeout or "None"} '
              f'Emergency Eps: {emergency_agent.eps:.3f}, '
              f'Emergency Steps: {emergency_step_count}')

    # Plot pre-training performance
    plot_performance(pretrain_metrics, args.output_dir, 'Pre-Training Performance of Emergency Agent')

    # Initialize environment for main training
    env = TrafficEnv(args.config, mode='gui' if args.render else 'binary', tripinfo_file=tripinfo_file)

    # Debug: Print configuration details
    print(f'\nMain Training Config file: {args.config}')
    print(f'Tripinfo file: {tripinfo_file}')
    print(f'Sumo command: {env.sumoCmd}')

    # Reset environment
    try:
        traci.close()
    except:
        pass
    state = env.reset()

    # Main Training Loop
    n_episode = 400
    metrics = {
        'average_travel_time': [],
        'emergency_waiting_time': [],
        'emergency_travel_time': [],
        'normal_travel_time': [],
        'emergency_reward': []
    }
    episode_durations = []
    fastest_duration = float('inf')
    timeout = float('inf')
    emergency_step_count = 0
    best_emergency_waiting_time = float('inf')

    for episode in range(n_episode):
        try:
            traci.close()
        except:
            pass
        state = env.reset()
        reward_epi_normal = 0.0
        reward_epi_emergency = 0.0
        actions = [None for _ in range(n_agents)]
        action_probs = [None for _ in range(n_agents)]
        done = False
        step_count = 0

        print(f'\nStarting Main Training Episode {episode + 1}')

        while not done:
            for i in range(n_agents):
                if env.emergency_detected:
                    action, action_prob = emergency_agent.select_action(state[i], i)
                else:
                    action, action_prob = normal_agent.select_action(state[i], i)
                actions[i] = action
                action_probs[i] = action_prob
            before_state = state
            state, reward_normal, reward_emergency, done = env.step(actions, timeout)
            reward_epi_normal += sum(reward_normal)
            reward_epi_emergency += sum(reward_emergency)
            step_count += 1

            if env.emergency_detected:
                emergency_agent.set_lr(True, episode)
            else:
                normal_agent.set_lr(False, episode)

            transition_normal = [before_state, action_probs, state, reward_normal, done]
            transition_emergency = [before_state, action_probs, state, reward_emergency, done]
            if env.emergency_detected:
                emergency_agent.push(transition_emergency)
                if emergency_agent.train_start():
                    for i in range(n_agents):
                        emergency_agent.train_model(i)
                    emergency_step_count += 1
                    emergency_agent.update_eps_emergency(episode)
            else:
                normal_agent.push(transition_normal)
                if normal_agent.train_start():
                    for i in range(n_agents):
                        normal_agent.train_model(i)
                    normal_agent.update_eps(episode)

            if step_count % 100 == 0:
                current_lr = normal_agent.actor_optimizers[0].param_groups[0]['lr']
                print(f'Main Training Episode {episode + 1} - Step {step_count}: '
                      f'Normal Reward: {reward_epi_normal:.2f}, '
                      f'Emergency Reward: {reward_epi_emergency:.2f}, '
                      f'Normal Eps: {normal_agent.eps:.3f}, Emergency Eps: {emergency_agent.eps:.3f}, '
                      f'Current LR: {current_lr:.6f}, '
                      f'Emergency Steps: {emergency_step_count} '
                      f'Interval: {env.decision_time}')
            if done:
                print(f'Total Normal Reward of Main Training Episode {episode + 1} : {reward_epi_normal:.2f}')
                break

        duration = env.get_episode_duration()
        episode_durations.append(duration)
        if duration < timeout:
            fastest_duration = duration
            timeout = 2 * fastest_duration

        env.close()
        avg_travel_time, avg_emergency_waiting, avg_emergency_travel, avg_normal_travel = get_traffic_metrics(tripinfo_file)
        metrics['average_travel_time'].append(avg_travel_time)
        metrics['emergency_waiting_time'].append(avg_emergency_waiting)
        metrics['emergency_travel_time'].append(avg_emergency_travel)
        metrics['normal_travel_time'].append(avg_normal_travel)
        metrics['emergency_reward'].append(reward_epi_emergency)

        # Save models
        normal_agent.save_model(os.path.join(args.output_dir, 'normal_agent_last_big_map.th'))
        emergency_agent.save_model(os.path.join(args.output_dir, 'emergency_agent_last_big_map.th'))
        if avg_emergency_waiting < best_emergency_waiting_time:
            best_emergency_waiting_time = avg_emergency_waiting
            normal_agent.save_model(os.path.join(args.output_dir, 'normal_agent_best_big_map.th'))
            emergency_agent.save_model(os.path.join(args.output_dir, 'emergency_agent_best_big_map.th'))
            print(f'New best model saved with Emergency Waiting Time: {best_emergency_waiting_time:.2f}')

        print(f'Main Training Episode {episode + 1} Completed: '
              f'Avg Travel Time: {avg_travel_time:.2f}, '
              f'Emergency Waiting Time: {avg_emergency_waiting:.2f}, '
              f'Emergency Travel Time: {avg_emergency_travel:.2f}, '
              f'Normal Travel Time: {avg_normal_travel:.2f}, '
              f'Duration: {duration}, Timeout: {timeout or "None"} '
              f'Normal Eps: {normal_agent.eps:.3f}, Emergency Eps: {emergency_agent.eps:.3f}, '
              f'Emergency Steps: {emergency_step_count}')

    # Plot performance
    plot_performance(metrics, args.output_dir, 'Performance of Dual-MADDPG System')
