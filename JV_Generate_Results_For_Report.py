import numpy as np
import gym
import random
import gc
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from collections import deque, namedtuple

import matplotlib.pyplot as plt
from tqdm import tqdm

import matplotlib
matplotlib.use('agg') ## using a non-interactive background framework for matplotlib

from JV_Duelling_DQN import DDQN,ReplayBuffer,DDQN_Agent_egreedy
from JV_REINFORCE import PolicyNetwork,ValueNetwork,ReplayBuffer,ReinforceAgent

from JV_DDQN_CartPole_Best_Config import *
from JV_DDQN_Acrobot_Best_Config import *

from JV_REINFORCE_CartPole_Best_Config import *
from JV_REINFORCE_Acrobot_Best_Config import *

def plot_mean_std_dev(data_list,message,xlabel,ylabel,scale_down = 1,results_dir=""):

    """
    
    Method to plot the column wise mean of a 2D data with a band of stddev around it

    data : 2D np array.
    message : title of the plot.
    xlabel : label of the x axis.
    ylabel : label of the y axis.
    scale_down : factor by which x axis is shrinked

    Returns : None
    
    """
    
    plt.figure(figsize=(7,7))

    for ii in range(len(data_list)):

        data = data_list[ii]

        mean = np.mean(data,axis=0)/scale_down
        stddev = np.std(data,axis=0)
        
        plt.plot(np.arange(mean.shape[0]),mean, label='Type'+str(ii+1))
        y_min = mean - stddev
        y_max = mean + stddev

        plt.fill_between(np.arange(mean.shape[0]), y_min, y_max, alpha=0.5)

        scale_label = ""
        if scale_down>1:
            scale_label = "X"+str(scale_down)
    
    plt.legend()
    plt.xlabel(xlabel+scale_label)
    plt.ylabel(ylabel)
    plt.title(message)

    if("regret" in message.lower()):
        plt.savefig(results_dir+message+"Regret_Plot.png")
    else:
        plt.savefig(results_dir+message+"Reward_Plot.png")

    plt.show()
    plt.clf()
    plt.close('all')
    gc.collect()



def Run_ddqn_for_Report(environment,config,seeds):

    device_to_use = config["device_to_use"]

    if config["activation"] == "silu":

        device_to_use = "cpu" ## silu is computationally expensive and slows down the GPU operations
    
    aggregation = config["aggregation"]
    
    env = gym.make(environment)
    
    state_shape = env.observation_space.shape[0]
    action_shape = env.action_space.n
    
    """
    Setting the Hyperparameters
    """
    
    ## creating a hyper param dict, with default values, which can be changed during hyper parameter tuning

    if  config["activation"] == "relu":
        activation_fn  = nn.ReLU()

    elif  config["activation"] == "silu":
        activation_fn  = nn.SiLU()

    elif  config["activation"] == "selu":
        activation_fn  = nn.SELU()

    elif  config["activation"] == "tanh":
        activation_fn  = nn.Tanh()
    
    hyperparm_dict = {
        "replay_buffer_size" : config["replay_buffer_size"],
        
        "batch_size" : config["batch_sizes"],
    
        "lr" : config["lr"],
    
        "target_net_update_freq" : config["target_net_update_freq"],
    
        "gamma" : 0.99,
    
        "activation" : activation_fn,
    
        "l2_regularization" : config["weight_decay"],
    
        "num_shared_layers" : config["num_shared_layers"], 
        
        "shared_hidden_sizes" : [config["shared_hidden_size"]]*config["num_shared_layers"],
        
        "num_value_layers" : config["num_value_layers"],
        
        "value_hidden_sizes" : [config["value_hidden_size"]]*(config["num_value_layers"]-1)+[1],
        
        "num_advantage_layers" : config["num_advantage_layers"], 
        
        "advantage_hidden_sizes" : [config['advantage_hidden_size']]*(config["num_advantage_layers"]-1)+[action_shape],
    
        "optimiser" : config["optimiser"]
        
    }
    
    
    device = torch.device(device_to_use)

    if environment == "CartPole-v1":
        n_episodes = 200 ##based on the convergence of the best config of Type 1 and 2 Duelling DQN for cartpole
        max_regret = 500

    else:

        n_episodes = 820 ##based on the convergence of the best config of Type 1 and 2 Duelling DQN for Acrobot
        max_regret = -500

    list_of_experiment_wise_episodic_rewards = []
    list_of_experiment_wise_episodic_smooth_rewards = []
    list_of_experiment_wise_total_regrets = [] 
    list_of_experiment_wise_total_smooth_regrets = []

    for seed in seeds:

        agent = DDQN_Agent_egreedy(hyperparm_dict,state_size=state_shape,action_size = action_shape,seed = seed,aggregation_type=aggregation)
        
        env.reset(seed=seed)

        eps = config['start_eps']
        eps_decay = config['eps_decay']
        
        list_of_episode_wise_rewards = []
        list_of_episode_wise_smooth_rewards = []

        list_of_episode_wise_regrets = []
        list_of_episode_wise_smooth_regrets = []

        regret_window = deque(maxlen=50) ##last 100 regret values for smoothening.
        scores_window = deque(maxlen=50) ##last 100 score values for smoothening.

        for i_episode in tqdm(range(1, n_episodes+1)):
            
            state = env.reset(seed=seed)[0]
            score = 0
            eps_end = 0.01

            for t in range(500): ## max episode length in both
                action = agent.act(state, eps)
                next_state, reward, done, _, _ = env.step(action)
                agent.step(state, action, reward, next_state, done)
                state = next_state
                score += reward
                if done:
                    break
            eps = max(eps_end, eps_decay*eps)

            cur_episode_regret = max_regret - score

            scores_window.append(score)
            regret_window.append(cur_episode_regret)

            list_of_episode_wise_rewards.append(score)
            list_of_episode_wise_regrets.append(cur_episode_regret)

            list_of_episode_wise_smooth_rewards.append(np.mean(scores_window))
            list_of_episode_wise_smooth_regrets.append(np.mean(regret_window))

        list_of_experiment_wise_episodic_rewards.append(list_of_episode_wise_rewards)
        list_of_experiment_wise_total_regrets.append(list_of_episode_wise_regrets)
        list_of_experiment_wise_episodic_smooth_rewards.append(list_of_episode_wise_smooth_rewards)
        list_of_experiment_wise_total_smooth_regrets.append(list_of_episode_wise_smooth_regrets)

    return list_of_experiment_wise_episodic_rewards,list_of_experiment_wise_episodic_smooth_rewards,list_of_experiment_wise_total_regrets,list_of_experiment_wise_total_smooth_regrets




def make_dir(dir):
    try:
        os.mkdir(dir)
        print(f"{dir} creation success!")
    except Exception as e:
        #print(e)
        pass


def generate_ddqn_report():

    seeds = [23,76,18,16,55]

    results_dir = "Results/"
    make_dir(results_dir)

    ######################## Duelling DQN for CarPole ########################

    expt = "Duelling-DQN-Cartpole"
    expt_dir = results_dir+expt+"/"
    make_dir(expt_dir)

    type1_experiment_wise_episodic_rewards,type1_experiment_wise_episodic_smoothened_rewards,type1_experiment_wise_total_regrets,type1_experiment_wise_smoothened_regrets = Run_ddqn_for_Report(environment='CartPole-v1',config=cp_type1_best_config,seeds=seeds)
    type2_experiment_wise_episodic_rewards,type2_experiment_wise_episodic_smoothened_rewards,type2_experiment_wise_total_regrets,type2_experiment_wise_smoothened_regrets = Run_ddqn_for_Report(environment='CartPole-v1',config=cp_type2_best_config,seeds=seeds)

    np.save(expt_dir+expt+"-type1-rewards.npy",np.array(type1_experiment_wise_episodic_smoothened_rewards))
    np.save(expt_dir+expt+"-type1-regrets.npy",np.array(type1_experiment_wise_smoothened_regrets))

    np.save(expt_dir+expt+"-type2-rewards.npy",np.array(type2_experiment_wise_episodic_smoothened_rewards))
    np.save(expt_dir+expt+"-type2-regrets.npy",np.array(type2_experiment_wise_smoothened_regrets))


    #plot_mean_std_dev([type1_experiment_wise_episodic_rewards,type2_experiment_wise_episodic_rewards],message=expt+"Episode Wise Total Reward",xlabel="Episode",ylabel="Total Reward",results_dir=expt_dir)
    plot_mean_std_dev([type1_experiment_wise_episodic_smoothened_rewards,type2_experiment_wise_episodic_smoothened_rewards],message=expt+"Episode Wise Total Reward",xlabel="Episode",ylabel="Total Reward",results_dir=expt_dir)

    #plot_mean_std_dev([type1_experiment_wise_total_regrets,type2_experiment_wise_total_regrets],message=expt+"Episode Wise Regret",xlabel="Episode",ylabel="Regret",results_dir=expt_dir)
    plot_mean_std_dev([type1_experiment_wise_smoothened_regrets,type2_experiment_wise_smoothened_regrets],message=expt+"Episode Wise Total Regret",xlabel="Episode",ylabel="Regret",results_dir=expt_dir)


    ######################## Duelling DQN for Acrobot ########################

    expt = "Duelling-DQN-Acrobot"
    expt_dir = results_dir+expt+"/"
    make_dir(expt_dir)

    type1_experiment_wise_episodic_rewards,type1_experiment_wise_episodic_smoothened_rewards,type1_experiment_wise_total_regrets,type1_experiment_wise_smoothened_regrets = Run_ddqn_for_Report(environment='Acrobot-v1',config=ab_type1_best_config,seeds=seeds)
    type2_experiment_wise_episodic_rewards,type2_experiment_wise_episodic_smoothened_rewards,type2_experiment_wise_total_regrets,type2_experiment_wise_smoothened_regrets = Run_ddqn_for_Report(environment='Acrobot-v1',config=ab_type2_best_config,seeds=seeds)

    #plot_mean_std_dev([type1_experiment_wise_episodic_rewards,type2_experiment_wise_episodic_rewards],message=expt+"Episode Wise Total Reward",xlabel="Episode",ylabel="Total Reward",results_dir=expt_dir)
    plot_mean_std_dev([type1_experiment_wise_episodic_smoothened_rewards,type2_experiment_wise_episodic_smoothened_rewards],message=expt+"Episode Wise Total Reward",xlabel="Episode",ylabel="Total Reward",results_dir=expt_dir)

    #plot_mean_std_dev([type1_experiment_wise_total_regrets,type2_experiment_wise_total_regrets],message=expt+"Episode Wise Regret",xlabel="Episode",ylabel="Regret",results_dir=expt_dir)
    plot_mean_std_dev([type1_experiment_wise_smoothened_regrets,type2_experiment_wise_smoothened_regrets],message=expt+"Episode Wise Total Regret",xlabel="Episode",ylabel="Regret",results_dir=expt_dir)

    np.save(expt_dir+expt+"-type1-rewards.npy",np.array(type1_experiment_wise_episodic_smoothened_rewards))
    np.save(expt_dir+expt+"-type1-regrets.npy",np.array(type1_experiment_wise_smoothened_regrets))

    np.save(expt_dir+expt+"-type2-rewards.npy",np.array(type2_experiment_wise_episodic_smoothened_rewards))
    np.save(expt_dir+expt+"-type2-regrets.npy",np.array(type2_experiment_wise_smoothened_regrets))


def Run_reinforce_for_Report(environment,config,seeds):

    device_to_use = config["device_to_use"]
    
    env = gym.make(environment)
    
    state_shape = env.observation_space.shape[0]
    action_shape = env.action_space.n
    
    """
    Setting the Hyperparameters
    """
    
    ## creating a hyper param dict, with default values, which can be changed during hyper parameter tuning

    if  config["policy_activation"] == "relu":
        policy_activation_fn  = nn.ReLU()

    elif  config["policy_activation"] == "silu":
        policy_activation_fn  = nn.SiLU()

    elif  config["policy_activation"] == "selu":
        policy_activation_fn  = nn.SELU()

    elif  config["policy_activation"] == "tanh":
        policy_activation_fn  = nn.Tanh()

    
    if  config["value_activation"] == "relu":
        value_activation_fn  = nn.ReLU()

    elif  config["value_activation"] == "silu":
        value_activation_fn  = nn.SiLU()

    elif  config["value_activation"] == "selu":
        value_activation_fn  = nn.SELU()

    elif  config["value_activation"] == "tanh":
        value_activation_fn  = nn.Tanh()
    
    hyperparm_dict = {

        "replay_buffer_size" : int(1e5),
        
        "batch_size" : config["batch_sizes"],
    
        "target_net_update_freq" : config["target_net_update_freq"],
        
        "policy_lr" : config["policy_lr"],

        "value_lr" : config["value_lr"],
    
        "gamma" : 0.99,
    
        "policy_activation" : policy_activation_fn,

        "value_activation" : value_activation_fn,
    
        "l2_regularization" : config["weight_decay"],
        
        "num_value_layers" : config["num_value_layers"],
        
        "value_hidden_sizes" : [config["value_hidden_size"]]*(config["num_value_layers"]),
        
        "num_policy_layers" : config["num_policy_layers"], 
        
        "policy_hidden_sizes" : [config['policy_hidden_size']]*(config["num_policy_layers"]),
    
        "policy_optimiser" : config["policy_optimiser"],

        "value_optimiser" : config["value_optimiser"],

        "baseline" : config["baseline"],
        
    }
    
    
    device = torch.device(device_to_use)
    
    if environment == "Acrobot-v1":
        max_regret = -500
        n_episodes = 1500
        
    elif environment == "CartPole-v1":
        max_regret = 500
        n_episodes = 1500

    #list_of_experiment_wise_episodic_rewards = []
    #list_of_experiment_wise_total_regrets = []
    
    list_of_experiment_wise_episodic_smooth_rewards = []
    list_of_experiment_wise_total_smooth_regrets = []

    for seed in seeds:

        env.reset(seed=seed)
        agent = ReinforceAgent(hyperparm_dict,state_size=state_shape,action_size = action_shape,seed = seed,baseline = config["baseline"])

        ## for book keeping
        list_of_episode_wise_smooth_rewards = []
        list_of_episode_wise_smooth_regrets = []
        regret_window = deque(maxlen=100) ##last 100 regret values for smoothening.
        scores_window = deque(maxlen=100) ##last 100 score values for smoothening.

        for i_episode in tqdm(range(1, n_episodes+1)):
            ## for training
            list_of_states = []
            list_of_actions = []
            list_of_rewards = []
            
            state = env.reset(seed=seed)[0]
            score = 0
            
            for t in range(500): ## 500 is maximum epsiode length,as per the gym environment documentation
                action = agent.act(state)
                next_state, reward, done, _, _ = env.step(action)

                if agent.baseline:
                    agent.value_step(state, action, reward, next_state, done)
                
                
                list_of_states.append(state)
                list_of_actions.append(action)
                list_of_rewards.append(reward)
                
                state = next_state
                score += reward
                
                if done:
                    break

            agent.learn_policy([list_of_states, list_of_actions, list_of_rewards])
                

            cur_episode_regret = max_regret - score
            regret_window.append(cur_episode_regret)
            scores_window.append(score)

            list_of_episode_wise_smooth_rewards.append(np.mean(scores_window))
            list_of_episode_wise_smooth_regrets.append(np.mean(regret_window))

        list_of_experiment_wise_episodic_smooth_rewards.append(list_of_episode_wise_smooth_rewards)
        list_of_experiment_wise_total_smooth_regrets.append(list_of_episode_wise_smooth_regrets)

    return list_of_experiment_wise_episodic_smooth_rewards,list_of_experiment_wise_total_smooth_regrets


def generate_reinforce_report():

    seeds = [23,76,18,16,55]

    results_dir = "Results/"
    make_dir(results_dir)

     ######################## REINFORCE for Acrobot ########################

    expt = "REINFORCE-Acrobot"
    expt_dir = results_dir+expt+"/"
    make_dir(expt_dir)

    type1_experiment_wise_episodic_smoothened_rewards,type1_experiment_wise_smoothened_regrets = Run_reinforce_for_Report(environment='Acrobot-v1',config=reinforce_ab_baseline_best_config,seeds=seeds)
    type2_experiment_wise_episodic_smoothened_rewards,type2_experiment_wise_smoothened_regrets = Run_reinforce_for_Report(environment='Acrobot-v1',config=reinforce_ab_no_baseline_best_config,seeds=seeds)

    #plot_mean_std_dev([type1_experiment_wise_episodic_rewards,type2_experiment_wise_episodic_rewards],message=expt+"Episode Wise Total Reward",xlabel="Episode",ylabel="Total Reward",results_dir=expt_dir)
    plot_mean_std_dev([type1_experiment_wise_episodic_smoothened_rewards,type2_experiment_wise_episodic_smoothened_rewards],message=expt+"Episode Wise Total Reward",xlabel="Episode",ylabel="Total Reward",results_dir=expt_dir)

    #plot_mean_std_dev([type1_experiment_wise_total_regrets,type2_experiment_wise_total_regrets],message=expt+"Episode Wise Regret",xlabel="Episode",ylabel="Regret",results_dir=expt_dir)
    plot_mean_std_dev([type1_experiment_wise_smoothened_regrets,type2_experiment_wise_smoothened_regrets],message=expt+"Episode Wise Total Regret",xlabel="Episode",ylabel="Regret",results_dir=expt_dir)

    np.save(expt_dir+expt+"-BASELINE-rewards.npy",np.array(type1_experiment_wise_episodic_smoothened_rewards))
    np.save(expt_dir+expt+"-BASELINE-regrets.npy",np.array(type1_experiment_wise_smoothened_regrets))

    np.save(expt_dir+expt+"-NO_BASELINE-rewards.npy",np.array(type2_experiment_wise_episodic_smoothened_rewards))
    np.save(expt_dir+expt+"-NO_BASELINE-regrets.npy",np.array(type2_experiment_wise_smoothened_regrets))

    ######################## REINFORCE for CarPole ########################

    expt = "REINFORCE-Cartpole"
    expt_dir = results_dir+expt+"/"
    make_dir(expt_dir)

    type1_experiment_wise_episodic_smoothened_rewards,type1_experiment_wise_smoothened_regrets = Run_reinforce_for_Report(environment='CartPole-v1',config=reinforce_cp_baseline_best_config,seeds=seeds)
    type2_experiment_wise_episodic_smoothened_rewards,type2_experiment_wise_smoothened_regrets = Run_reinforce_for_Report(environment='CartPole-v1',config=reinforce_cp_no_baseline_best_config,seeds=seeds)

    np.save(expt_dir+expt+"-BASELINE-rewards.npy",np.array(type1_experiment_wise_episodic_smoothened_rewards))
    np.save(expt_dir+expt+"-BASELINE-regrets.npy",np.array(type1_experiment_wise_smoothened_regrets))

    np.save(expt_dir+expt+"-NO_BASELINE-rewards.npy",np.array(type2_experiment_wise_episodic_smoothened_rewards))
    np.save(expt_dir+expt+"-NO_BASELINE-regrets.npy",np.array(type2_experiment_wise_smoothened_regrets))


    #plot_mean_std_dev([type1_experiment_wise_episodic_rewards,type2_experiment_wise_episodic_rewards],message=expt+"Episode Wise Total Reward",xlabel="Episode",ylabel="Total Reward",results_dir=expt_dir)
    plot_mean_std_dev([type1_experiment_wise_episodic_smoothened_rewards,type2_experiment_wise_episodic_smoothened_rewards],message=expt+"Episode Wise Total Reward",xlabel="Episode",ylabel="Total Reward",results_dir=expt_dir)

    #plot_mean_std_dev([type1_experiment_wise_total_regrets,type2_experiment_wise_total_regrets],message=expt+"Episode Wise Regret",xlabel="Episode",ylabel="Regret",results_dir=expt_dir)
    plot_mean_std_dev([type1_experiment_wise_smoothened_regrets,type2_experiment_wise_smoothened_regrets],message=expt+"Episode Wise Total Regret",xlabel="Episode",ylabel="Regret",results_dir=expt_dir)


#generate_ddqn_report()
#generate_reinforce_report()