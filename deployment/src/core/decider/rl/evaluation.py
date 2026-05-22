import numpy as np
import torch

from crowd_sim.envs.utils.info import *


def evaluate(actor_critic, eval_envs, num_processes, device, test_size, logging, config, args, visualize=False):
    """ function to run all testing episodes and log the testing metrics """
    # initializations
    eval_episode_rewards = []

    if config.robot.policy not in ['orca', 'social_force']:
        eval_recurrent_hidden_states = {}

        node_num = 1
        edge_num = actor_critic.base.human_num + 1
        eval_recurrent_hidden_states['human_node_rnn'] = torch.zeros(num_processes, node_num, actor_critic.base.human_node_rnn_size,
                                                                     device=device)

        eval_recurrent_hidden_states['human_human_edge_rnn'] = torch.zeros(num_processes, edge_num,
                                                                           actor_critic.base.human_human_edge_rnn_size,
                                                                           device=device)

    eval_masks = torch.zeros(num_processes, 1, device=device)

    success_times = []
    collision_human_times = []
    collision_obs_times = []
    target_lost_times = []

    success = 0
    collision_human = 0
    collision_obs = 0
    target_lost = 0
    too_close_ratios = []
    min_dist = []

    collision_human_cases = []
    collision_obs_cases = []
    target_lost_cases = []

    all_target_distances = []

    # to make it work with the virtualenv in sim2real
    if hasattr(eval_envs.venv, 'envs'):
        baseEnv = eval_envs.venv.envs[0].env
    else:
        baseEnv = eval_envs.venv.unwrapped.envs[0].env
    time_limit = baseEnv.time_limit

    # start the testing episodes
    for k in range(test_size):
        baseEnv.episode_k = k
        done = False
        rewards = []
        stepCounter = 0
        episode_rew = 0
        obs = eval_envs.reset()
        global_time = 0.0
        too_close = 0.
        target_distances = []

        while not done:
            stepCounter = stepCounter + 1
            
            # Record target_dist BEFORE step (to avoid VecEnv auto-reset issue)
            if hasattr(baseEnv, 'target_human'):
                target_pos = np.array([baseEnv.target_human.px, baseEnv.target_human.py])
                robot_pos = obs['robot_node'][0, 0, :2].cpu().numpy()
                target_dist = np.linalg.norm(target_pos - robot_pos)
                target_distances.append(target_dist)
            
            if config.robot.policy not in ['orca', 'social_force']:
                # run inference on the NN policy
                with torch.no_grad():
                    _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                        obs,
                        eval_recurrent_hidden_states,
                        eval_masks,
                        deterministic=True)
            else:
                action = torch.zeros([1, 2], device=device)

            # if the vec_pretext_normalize.py wrapper is used, send the predicted traj to env
            if args.env_name == 'CrowdSimPredRealGST-v0' and config.env.use_wrapper:
                out_pred = obs['spatial_edges'][:, :, 2:].to('cpu').numpy()
                # send manager action to all processes
                ack = eval_envs.talk2Env(out_pred)
                assert all(ack)
            # render
            if visualize:
                eval_envs.render()

            # Obser reward and next obs
            obs, rew, done, infos = eval_envs.step(action)

            # Update global_time using stepCounter to avoid VecEnv reset issue
            global_time = stepCounter * baseEnv.time_step

            # record the info for calculating testing metrics
            rewards.append(rew)

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
        too_close_ratios.append(too_close/stepCounter*100)

        avg_target_dist = np.mean(target_distances) if target_distances else float('inf')
        all_target_distances.append(avg_target_dist)

        if isinstance(infos[0]['info'], Success):
            success += 1
            success_times.append(global_time)
            print('Success')
        elif isinstance(infos[0]['info'], HumanCollision):
            collision_human += 1
            collision_human_cases.append(k)
            collision_human_times.append(global_time)
            print('Collision with human')
        elif isinstance(infos[0]['info'], ObstacleCollision):
            collision_obs += 1
            collision_obs_cases.append(k)
            collision_obs_times.append(global_time)
            print('Collision with obstacle')
        elif isinstance(infos[0]['info'], TargetLost):
            target_lost += 1
            target_lost_cases.append(k)
            target_lost_times.append(global_time)
            print('Target Lost')
        else:
            raise ValueError('Invalid end signal from environment')

    # all episodes end
    success_rate = success / test_size
    collision_human_rate = collision_human / test_size
    collision_obs_rate = collision_obs / test_size
    target_lost_rate = target_lost / test_size
    assert success + collision_human + collision_obs + target_lost == test_size
    
    avg_nav_time = sum(success_times) / len(success_times) if success_times else time_limit
    avg_target_distance = np.mean(all_target_distances)
    # logging
    logging.info(
        'Testing success rate: {:.4f}, human collision rate: {:.4f}, obstacle collision rate: {:.4f}, target lost rate: {:.4f}, '
        'average following distance: {:.4f}, average intrusion ratio: {:.4f}%, '
        'average minimal distance during intrusions: {:.4f}'.
            format(success_rate, collision_human_rate, collision_obs_rate, target_lost_rate, avg_target_distance, 
                   np.mean(too_close_ratios), np.mean(min_dist) if min_dist else float('inf')))

    logging.info('Human collision cases: ' + ' '.join([str(x) for x in collision_human_cases]))
    logging.info('Obstacle collision cases: ' + ' '.join([str(x) for x in collision_obs_cases]))
    logging.info('Target lost cases: ' + ' '.join([str(x) for x in target_lost_cases]))
    # print(" Evaluation using {} episodes: mean reward {:.5f}\n".format(
    #     len(eval_episode_rewards), np.mean(eval_episode_rewards)))

    eval_envs.close()