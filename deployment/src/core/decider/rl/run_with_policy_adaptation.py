import numpy as np
import torch

from crowd_sim.envs.utils.info import *
import csv
import os

from collections import Counter

from scipy.stats import gaussian_kde

def compute_quantile(value, data):
    data.append(value)
    sorted_data = sorted(data) # TODO: store the sorted list so that it don't have to be sorted for many times
    count_less = sum(1 for i in sorted_data if i <= value)
    quantile = count_less / len(sorted_data)
    return quantile

def get_kde_density(value, kde):
    return kde(value)[0]
    

def run_with_policy_adaptation(actor_critics, score_lists, eval_envs, num_processes, device, test_size, logging, config, args, gif_save_path, mode, kde_smoothness):
    """ function to run all testing episodes and log the testing metrics """
    # initializations
    eval_episode_rewards = []
    safety_density_list = [gaussian_kde(s_l, bw_method=kde_smoothness) for s_l in score_lists]
    if config.robot.policy not in ['orca', 'social_force']:
        eval_recurrent_hidden_states = {}

        node_num = 1
        edge_num = actor_critics[-1].base.human_num + 1 # actor_critics are sorted from conservative to aggressive
        eval_recurrent_hidden_states['human_node_rnn'] = torch.zeros(num_processes, node_num, actor_critics[-1].base.human_node_rnn_size,
                                                                     device=device)

        eval_recurrent_hidden_states['human_human_edge_rnn'] = torch.zeros(num_processes, edge_num,
                                                                           actor_critics[-1].base.human_human_edge_rnn_size,
                                                                           device=device)

    eval_masks = torch.zeros(num_processes, 1, device=device)

    success_times = []
    collision_times = []
    timeout_times = []

    success = 0
    collision = 0
    timeout = 0
    too_close_ratios = []
    min_dist = []

    collision_cases = []
    timeout_cases = []

    all_path_len = []

    # to make it work with the virtualenv in sim2real
    if hasattr(eval_envs.venv, 'envs'):
        baseEnv = eval_envs.venv.envs[0].env
    else:
        baseEnv = eval_envs.venv.unwrapped.envs[0].env
    
    time_limit = baseEnv.time_limit
    
    baseEnv.collect_failure_samples = True
    baseEnv.collect_nearest_score = True

    # start the testing episodes
    total_policy_indices = []
    for k in range(test_size):
        baseEnv.episode_k = k
        done = False
        rewards = []
        stepCounter = 0
        episode_rew = 0
        obs = eval_envs.reset()
        global_time = 0.0
        path_len = 0.
        too_close = 0.
        last_pos = obs['robot_node'][0, 0, :2].cpu().numpy()
        
        num_safe_ttc = 0
        last_safety_score = 0
        
        acting_actor_critic_index = 0 # start with conservative policies
        
        policy_indices = []

        while not done:
            stepCounter = stepCounter + 1                # run inference on the NN policy
            with torch.no_grad():
                # print(acting_actor_critic_index)
                _, action, _, eval_recurrent_hidden_states = actor_critics[acting_actor_critic_index].act(
                    obs,
                    eval_recurrent_hidden_states,
                    eval_masks,
                    deterministic=True)
                policy_indices.append(acting_actor_critic_index)
                total_policy_indices.append(acting_actor_critic_index)

            if not done:
                global_time = baseEnv.global_time

            # if the vec_pretext_normalize.py wrapper is used, send the predicted traj to env
            if args.env_name == 'CrowdSimPredRealGST-v0' and config.env.use_wrapper:
                out_pred = obs['spatial_edges'][:, :, 2:].to('cpu').numpy()
                # send manager action to all processes
                ack = eval_envs.talk2Env(out_pred)
                assert all(ack)

            baseEnv.plot_step(gif_save_path) # yjp mark

            # Obser reward and next obs
            obs, rew, done, infos = eval_envs.step(action)
                        
            safety_score = infos[0]['nearest_score']
            if safety_score > 1000:
                num_safe_ttc += 1
                if num_safe_ttc < 3:
                    safety_score = last_safety_score
                else:
                    safety_score = safety_score
                    num_safe_ttc = 0
            else:
                num_safe_ttc = 0
            
            last_safety_score = safety_score
            # if safety_score < 100 and stepCounter > 0:
            safety_quantile = compute_quantile(safety_score, score_lists[acting_actor_critic_index])
            
            if mode == "gradual":
                if safety_quantile >= 0.95: # move to be more aggressive
                    if acting_actor_critic_index < len(actor_critics) - 1:
                        acting_actor_critic_index += 1
                        print(f"move to be more aggressive with safety score: {safety_score} in step: {stepCounter}")
                elif safety_quantile < 0.50: # move to be more conservative
                    if acting_actor_critic_index > 0:
                        acting_actor_critic_index -= 1
                        print(f"move to be more conservative with safety score: {safety_score} in step: {stepCounter}")
            elif mode == "optimal":
                if safety_quantile >= 0.95 or safety_quantile < 0.50:
                    # safety_score = 0
                    # safety_quantile = compute_quantile(safety_score, score_lists[acting_actor_critic_index])
                    safety_quantiles = [compute_quantile(safety_score, s_l) for s_l in score_lists]
                    max_q_index = 0
                    for i, q in enumerate(safety_quantiles):
                        if q - safety_quantiles[max_q_index] > 0.02:
                            max_q_index = i
                        elif abs(q - safety_quantiles[max_q_index]) <= 0.001 and safety_quantiles[max_q_index] > 0.99:
                            max_q_index = max(i, max_q_index)
                            
                    acting_actor_critic_index = max_q_index#np.argmax(safety_quantiles)
                    
                    print(f"safety score: {safety_score:.1f}, change to aggressiveness: {acting_actor_critic_index}")
            elif mode == "density":
                if safety_quantile >= 0.95:
                    safety_quantiles = [compute_quantile(safety_score, s_l) for s_l in score_lists]
                    max_q_index = acting_actor_critic_index
                    for i, q in enumerate(safety_quantiles):
                        if q - safety_quantiles[max_q_index] > 0.05:
                            max_q_index = i
                        elif abs(q - safety_quantiles[max_q_index]) <= 0.001 and safety_quantiles[max_q_index] > 0.99:
                            max_q_index = max(i, max_q_index)
                    print("quantile mode")        
                    acting_actor_critic_index = max_q_index#np.argmax(safety_quantiles)
                elif safety_quantile < 0.50:
                    safety_densities = [get_kde_density(safety_score, safety_kde) for safety_kde in safety_density_list]
                    min_d_index  = acting_actor_critic_index
                    for i, d in enumerate(safety_densities):
                        if d < safety_densities[min_d_index] - 0.02:
                            min_d_index = i
                    acting_actor_critic_index = min_d_index  
                    print("density mode")  
                else:
                    print("stay")
                print(f"safety score: {safety_score:.1f}, change to aggressiveness: {acting_actor_critic_index}")
            else:
                acting_actor_critic_index = int(mode) - 1
            
            
            # record the info for calculating testing metrics
            rewards.append(rew)

            path_len = path_len + np.linalg.norm(obs['robot_node'][0, 0, :2].cpu().numpy() - last_pos)
            last_pos = obs['robot_node'][0, 0, :2].cpu().numpy()

            if isinstance(infos[0]['info'], Danger):
                too_close = too_close + 1
                min_dist.append(infos[0]['info'].min_dist)

            episode_rew += rew[0]


            eval_masks = torch.tensor(
                [[0.0] if done_ else [1.0] for done_ in done],
                dtype=torch.float32,
                device=device)

            for info in infos:
                if 'episode' in info.keys():
                    eval_episode_rewards.append(info['episode']['r'])

        # an episode ends!
        print('')
        print('Reward={}'.format(episode_rew))
        print('Episode', k, 'ends in', stepCounter)
        all_path_len.append(path_len)
        too_close_ratios.append(too_close/stepCounter*100)


        if isinstance(infos[0]['info'], ReachGoal):
            success += 1
            success_times.append(global_time)
            result = 'Success'
            print(result)
        elif isinstance(infos[0]['info'], Collision):
            collision += 1
            collision_cases.append(k)
            collision_times.append(global_time)
            result = 'Collision'
            print(result)
        elif isinstance(infos[0]['info'], Timeout):
            timeout += 1
            timeout_cases.append(k)
            timeout_times.append(time_limit)
            result = 'Timeout'
            print(result)
        elif isinstance(infos[0]['info'] is None):
            result = 'None'
            pass
        else:
            raise ValueError('Invalid end signal from environment')

        print(f"Current SR: {success/(k+1)}, CR: {collision/(k+1)}")
        policy_index_frequency_count = Counter(policy_indices)
        sorted_frequency = sorted(policy_index_frequency_count.items(), key=lambda item: item[1])
        for item, frequency in sorted_frequency:
            percentage = (frequency / len(policy_indices)) * 100
            print(f"Policy Index: {item}, Frequency: {frequency}, Percentage: {percentage:.2f}%")
        baseEnv.animate_episode(gif_save_path, f"{k}_{result}")
        
    # all episodes end
    success_rate = success / test_size
    collision_rate = collision / test_size
    timeout_rate = timeout / test_size
    assert success + collision + timeout == test_size
    avg_nav_time = sum(success_times) / len(
        success_times) if success_times else time_limit  # baseEnv.env.time_limit

    # logging
    print(
        'Testing success rate: {:.2f}, collision rate: {:.2f}, timeout rate: {:.2f}, '
        'nav time: {:.2f}, path length: {:.2f}, average intrusion ratio: {:.2f}%, '
        'average minimal distance during intrusions: {:.2f}'.
            format(success_rate, collision_rate, timeout_rate, avg_nav_time, np.mean(all_path_len),
                   np.mean(too_close_ratios), np.mean(min_dist)))

    print('Collision cases: ' + ' '.join([str(x) for x in collision_cases]))
    print('Timeout cases: ' + ' '.join([str(x) for x in timeout_cases]))
    print(" Evaluation using {} episodes: mean reward {:.5f}\n".format(
        len(eval_episode_rewards), np.mean(eval_episode_rewards)))

    file_path = os.path.join("cp_models", 'mixed_gradual', 'evaluation_data_best_model.csv')
    os.makedirs(os.path.join("cp_models", 'mixed_gradual'), exist_ok = True)
    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Success Times', 'Collision Times', 'Timeout Times', 'Path Length', 'Min Distance'])
        
        # Determine the maximum length of the lists
        max_length = max(len(success_times), len(collision_times), len(timeout_times), len(all_path_len), len(min_dist))
        
        # Write data to CSV, handling missing values directly
        for i in range(max_length):
            row = [
                success_times[i] if i < len(success_times) else '',
                collision_times[i] if i < len(collision_times) else '',
                timeout_times[i] if i < len(timeout_times) else '',
                all_path_len[i] if i < len(all_path_len) else '',
                min_dist[i] if i < len(min_dist) else ''
            ]
            writer.writerow(row)
    policy_selection_file_path = os.path.joinf("{env_config.unsafe_samples_path}", 'mixed_gradual', 'policy_selection.csv')
    
    with open(policy_selection_file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['policy_selection'])
        for i in range(len(total_policy_indices)):
            writer.writerow([total_policy_indices[i]])

    eval_envs.close()