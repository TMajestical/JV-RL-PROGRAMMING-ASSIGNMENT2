#!/usr/bin/env python
# coding: utf-8

# In[1]:


#JV


# In[2]:


import numpy as np
import gym
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from collections import deque, namedtuple

import matplotlib.pyplot as plt
from tqdm import tqdm

import wandb

import matplotlib


# In[3]:


gym.__version__


# In[ ]:





# ## REINFORCE Code

# ### Policy Network Class

# In[4]:


class PolicyNetwork(nn.Module):

    """
    
    A neural network based parameteric representation of the policy is being done. This class provides methods to create a neural network.
    
    """

    def __init__(self, state_size, action_size, seed = 76, activation = nn.ReLU(), num_hidden_layers = 1, hidden_sizes = [64]):
        """I
        nitialize parameters and build model.
        Params:
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            num_hidden_layers(int) : Default 1,  it is the number of hidden layers in the policy network.
            hidden_sizes (list of int) : default [64], List of number of neurons per hidden layer.
        """
        
        super(PolicyNetwork, self).__init__()

        self.activation = activation
        
        self.seed = torch.manual_seed(seed)

        hidden_layers = self._create_layers(state_size, hidden_sizes)
        self.hidden_layers = nn.Sequential(*hidden_layers)

        self.output_layer = nn.Linear(hidden_sizes[-1],action_size)

    def _create_layers(self, state_size, hidden_sizes):
        
        layers = []
        sizes = [state_size] + hidden_sizes
        
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i+1]))
            layers.append(self.activation)
            
        return layers
        

    def forward(self, state):
        """Build a network that maps state -> action values."""
        
        x = self.hidden_layers(state)
        
        output = F.softmax(self.output_layer(x),dim=1)

        return output

    def initalize_weights_biases(self,m):
            """
    
            Method to initialize weights given a torch module.
    
            Using "Xavier" Initialization.
    
            """
            if isinstance(m, nn.Linear):  ## it its a fully connected layer
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01) ##a small non-zero value


# ### Value Network Class

# In[5]:


class ValueNetwork(nn.Module):

    """
    
    A variant of REINFOCE algorithm uses parameterized representation of the state value function as the base line. This class creates a neural network for the value funciton.
    
    """

    def __init__(self, state_size, seed = 76, activation = nn.ReLU(), num_hidden_layers = 1, hidden_sizes = [64]):
        """I
        nitialize parameters and build model.
        Params:
            state_size (int): Dimension of each state
            seed (int): Random seed
            num_hidden_layers(int) : Default 1,  it is the number of hidden layers in the policy network.
            hidden_sizes (list of int) : default [64], List of number of neurons per hidden layer.
        """
        
        super(ValueNetwork, self).__init__()

        self.activation = activation
        
        self.seed = torch.manual_seed(seed)

        hidden_layers = self._create_layers(state_size, hidden_sizes)
        self.hidden_layers = nn.Sequential(*hidden_layers)

        self.output_layer = nn.Linear(hidden_sizes[-1],1) ## output is always the state's value hence the output layer size is 1.

    def _create_layers(self, state_size, hidden_sizes):
        
        layers = []
        sizes = [state_size] + hidden_sizes
        
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i+1]))
            layers.append(self.activation)
            
        return layers
        

    def forward(self, state):
        """Build a network that maps state -> action values."""
        
        x = self.hidden_layers(state)

        output = self.output_layer(x)
        
        ##no activation on the output, because it is the state's value
        
        return output

    def initalize_weights_biases(self,m):
            """
    
            Method to initialize weights given a torch module.
    
            Using "Xavier" Initialization.
    
            """
            if isinstance(m, nn.Linear):  ## it its a fully connected layer
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01) ##a small non-zero value


# ### Replay Buffer Class for Value Network's TD(0) based updates.

# In[6]:


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size,hyperparm_dict, seed, device = "cpu"):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            hyperparm_dict : dictionary containing hyperparameters.
            seed (int): random seed
            device (str) :  device to be used for execution.
        """
        
        self.action_size = action_size
        self.memory = deque(maxlen=hyperparm_dict["replay_buffer_size"])
        self.batch_size = hyperparm_dict["batch_size"]
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.device = device

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


# ### REINFORCE AGENT CLASS

# In[7]:


class ReinforceAgent:

    """
    Class providing methods to take actions and learn at the end of episodes, as per the standard (textbook) algorithm.
    
    """

    def __init__(self,hyperparm_dict, state_size, action_size,seed,device="cpu",baseline=False):

        ## Agent Environment Interaction
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.device = device
        self.hyperparm_dict = hyperparm_dict
        self.baseline = baseline

        ## Policy-Network 
        self.policy_net = PolicyNetwork(state_size=self.state_size,action_size=self.action_size,activation=self.hyperparm_dict["policy_activation"],
                                        num_hidden_layers=self.hyperparm_dict["num_policy_layers"],hidden_sizes=self.hyperparm_dict["policy_hidden_sizes"]).to(self.device)
        self.policy_net.apply(self.policy_net.initalize_weights_biases)
        
        ## Value-Network -- Baseline : Has target network as it follows TD(0) updates
        if self.baseline:     
            
            ## create value net
            self.value_net = ValueNetwork(state_size=self.state_size,activation=self.hyperparm_dict["value_activation"],
                                          num_hidden_layers=self.hyperparm_dict["num_value_layers"],hidden_sizes=self.hyperparm_dict["value_hidden_sizes"]).to(self.device)
            self.value_net.apply(self.value_net.initalize_weights_biases)

            ## create target value net
            self.target_value_net = ValueNetwork(state_size=self.state_size,activation=self.hyperparm_dict["value_activation"],
                                                 num_hidden_layers=self.hyperparm_dict["num_value_layers"],hidden_sizes=self.hyperparm_dict["value_hidden_sizes"]).to(self.device)
            self.target_value_net.apply(self.target_value_net.initalize_weights_biases)
    
            ## Replay Memory
            self.memory = ReplayBuffer(action_size,self.hyperparm_dict, seed, self.device) 
            ## Initialize time step- Needed for Value Target sync
            self.t_step = 0

        if self.hyperparm_dict["policy_optimiser"] == "adam":
            self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=self.hyperparm_dict["policy_lr"],weight_decay=self.hyperparm_dict["l2_regularization"])
            
        elif self.hyperparm_dict["policy_optimiser"] == "nadam":
            self.policy_optimizer = optim.NAdam(self.policy_net.parameters(), lr=self.hyperparm_dict["policy_lr"],weight_decay=self.hyperparm_dict["l2_regularization"])
            
        elif self.hyperparm_dict["policy_optimiser"] == "rmsprop":
            self.policy_optimizer = optim.RMSprop(self.policy_net.parameters(), lr=self.hyperparm_dict["policy_lr"],weight_decay=self.hyperparm_dict["l2_regularization"])

        if baseline:

            if self.hyperparm_dict["value_optimiser"] == "adam":
                self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=self.hyperparm_dict["value_lr"],weight_decay=self.hyperparm_dict["l2_regularization"])
                
            elif self.hyperparm_dict["value_optimiser"] == "nadam":
                self.value_optimizer = optim.NAdam(self.value_net.parameters(), lr=self.hyperparm_dict["value_lr"],weight_decay=self.hyperparm_dict["l2_regularization"])
                
            elif self.hyperparm_dict["value_optimiser"] == "rmsprop":
                self.value_optimizer = optim.RMSprop(self.value_net.parameters(), lr=self.hyperparm_dict["value_lr"],weight_decay=self.hyperparm_dict["l2_regularization"])
    
    def act(self, state):

        """
        Method to sample action for the give state according to the current policy.
        """

        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        
        self.policy_net.eval()
        with torch.no_grad():
            action_probs = self.policy_net(state)
        self.policy_net.train()

        ## Act according to the policy...
        action_probs = action_probs.detach().cpu().numpy()[0]
        action = np.random.choice(np.arange(self.action_size), p=action_probs)
        
        return action

    def value_step(self, state, action, reward, next_state, done):

        ''' Save experience in replay memory '''
        self.memory.add(state, action, reward, next_state, done)

        ''' If enough samples are available in memory, get random subset and learn '''
        if len(self.memory) >= self.hyperparm_dict["batch_size"]:
            experiences = self.memory.sample()
            self.learn_value(experiences, self.hyperparm_dict["gamma"])

        """ +V TARGETS PRESENT """
        ''' Updating the Network every 'UPDATE_EVERY' steps taken '''
        self.t_step = (self.t_step + 1) % self.hyperparm_dict["target_net_update_freq"]
        if self.t_step == 0:

            self.target_value_net.load_state_dict(self.value_net.state_dict())

    
    def learn_value(self, experiences, gamma=0.99):
        """ +E EXPERIENCE REPLAY PRESENT """
        states, actions, rewards, next_states, dones = experiences

        ''' Get max predicted V values (for next states) from target model'''
        V_targets_next = self.target_value_net(next_states).detach()#.unsqueeze(0)

        ''' Compute V targets for current states '''
        V_targets = rewards + (gamma * V_targets_next * (1 - dones))

        ''' Get expected V values from local model '''
        V_expected = self.value_net(states)

        ''' Compute loss '''
        loss = F.mse_loss(V_expected, V_targets)

        ''' Minimize the loss '''
        self.value_optimizer.zero_grad()
        loss.backward()

        ''' Gradiant Clipping '''
        """ +T TRUNCATION PRESENT """
        for param in self.value_net.parameters():
            param.grad.data.clamp_(-1, 1)

        self.value_optimizer.step()

    def learn_policy(self, episode_history, gamma=0.99):

        """
        Method to use the episode history to compute loss as per policy gradient theorem and update the parameters
        """
        
        ##unpacking
        list_of_states, list_of_actions, list_of_rewards = episode_history

        episode_len = len(list_of_actions)
        returns = [0]*episode_len ## to store true return at each step in the episode
        cur_return = 0 ## bcz return at last step is 0
        
        ## iterate over the experiences in the reverse order this makes it easy to compute the discounted return, in one pass
        for step in range(episode_len-1,-1,-1):
            cur_return = list_of_rewards[step] + gamma*cur_return
            returns[step] = cur_return

        ## list_of_states, list_of_actions, list_of_rewards are lists make them tensors to compute probs and then log probs
        ## log probs are used in weight update rule.

        episode_states_tensor = torch.FloatTensor(list_of_states).to(self.device)
        episode_actions_tensor = torch.LongTensor(list_of_actions).to(self.device).view(-1,1) ## index of gather must be a longtensor
        episode_returns_tensor = torch.FloatTensor(returns).to(self.device).view(-1,1)

        ## The update depends on grad(log(pi(St,At))).
        ## Hence we need to gather probabilities corresponding to actions taken.
        episode_chosen_action_probs_tensor = self.policy_net(episode_states_tensor).gather(1,episode_actions_tensor)

        delta = episode_returns_tensor
        
        if self.baseline: ##if its baseline variant of REINFORCE
            episode_state_values_tensor = self.value_net(episode_states_tensor).to(self.device)
            with torch.no_grad(): ## since value (baseline) is a constant w.r.t the policy parameters.
                delta = episode_returns_tensor - episode_state_values_tensor

        policy_update = torch.mean(torch.log(episode_chosen_action_probs_tensor) * delta)
        
        policy_loss = -policy_update ## our optimizers do gradient descent, but we need gradient ascent so do gradient descent on the negative of the objective


        ## PolicyNet Params update
        self.policy_net.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        
        """if self.baseline:

            ## because the update rule interms of delta is essentially a derivative of MSE loss b/w return and predicted value. 
            ## This makes sense because value is expected return, so tuning it w.r.t true return is correct.

            ## As given in the question this is the TD(0) update
            metric  = nn.MSELoss()

            #list_of_next_states = list_of_states[1:]
            #episode_next_states_tensor = torch.FloatTensor(list_of_next_states).to(self.device)

            value_loss = metric(episode_state_values_tensor,episode_returns_tensor)
            
            self.value_net.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()"""



# In[8]:


def test_model(env,agent,max_regret,seed=76,log_wandb=False,episodes=100):
    
    total_regret = 0
    episode_wise_regrets = []
    for i_episode in tqdm(range(1, episodes)):
            state = env.reset(seed=seed)[0]
            score = 0
            for t in range(500): ## max_t is maximum epsiode length.
                action = agent.act(state)
                next_state, reward, done, _, _ = env.step(action)
                state = next_state
                score += reward
                if done:
                    break
                    
            cur_episode_regret = max_regret - score
            episode_wise_regrets.append(cur_episode_regret)
            total_regret += cur_episode_regret 
    if not log_wandb:
        print(f"Average Regret: {np.mean(episode_wise_regrets)} in {i_episode+1} episodes")
    else:
        wandb.log({'Average Test Regret' : np.mean(episode_wise_regrets)})


# In[9]:


# In[10]:


def Reinforce_Experiment(env,agent,max_regret = 500,cutoff = 30,n_episodes=5000,max_t=500,seed=76, log_wandb = False):

    early_stopping = 0

    total_regret = 0

    regret_window = deque(maxlen=100) ##last 100 regret values for checking if the avg is less than 20.
    
    
    list_of_episodic_rewards = []
    
    for i_episode in tqdm(range(1, n_episodes+1)):

        list_of_states = []
        list_of_actions = []
        list_of_rewards = []
        
        state = env.reset(seed=seed)[0]
        
        score = 0
        
        for t in range(max_t): ## max_t is maximum epsiode length.
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

        #print(len(list_of_states), len(list_of_actions), len(list_of_rewards))

        agent.learn_policy([list_of_states, list_of_actions, list_of_rewards])
            

        cur_episode_regret = max_regret - score
        regret_window.append(cur_episode_regret)
        total_regret += cur_episode_regret

        list_of_episodic_rewards.append(score)

        #average_episodic_regret =  (cur_episode_regret - average_episodic_regret)/i_episode + average_episodic_regret

        if i_episode % 100 == 0:
            
            if not log_wandb:
                print('\rEpisode {}\tTotal Regret: {:.2f}\t Regret in Window: {:.2f}'.format(i_episode, total_regret,np.mean(regret_window)))
            else:
                wandb.log({'total_regret':total_regret,'regret_in_window': np.mean(regret_window),'Total Episodes':i_episode})
        
        if (max_regret>0 and np.mean(regret_window)<=cutoff):
            
            if not log_wandb:
                print('\nEnvironment solved in {:d} episodes!\tRegret in Window: {:.2f}'.format(i_episode, np.mean(regret_window)))            
            else:
                wandb.log({'final_total_regret':total_regret,'final_regret_in_window': np.mean(regret_window),'Total Episodes':i_episode})
            
            early_stopping = 1
            break

        elif max_regret<0 and ((np.mean(regret_window)<=-420) or (i_episode==1500 and np.mean(regret_window)>-300) or (i_episode==1000 and np.mean(regret_window)>-150) or (i_episode==500 and np.mean(regret_window)>-50) or (i_episode==150 and np.mean(regret_window)==0)):
            if not log_wandb:
                print('\nEnvironment solved in {:d} episodes!\tRegret in Window :  {:.2f}'.format(i_episode, np.mean(regret_window)))            
            else:
                wandb.log({'final_total_regret':total_regret,'final_regret_in_window': np.mean(regret_window),'Total Episodes':i_episode})
            
            early_stopping = 1
            break

    if not early_stopping:
        if not log_wandb:
            pass
        else:
            wandb.log({'final_total_regret':total_regret,'final_regret_in_window': np.mean(regret_window),'Total Episodes':i_episode})

    test_model(env,agent,max_regret,seed=seed,log_wandb=log_wandb) ## test this model to see the regret now.
    
    f = plt.figure(figsize=(7,7))
    plt.plot(np.arange(len(list_of_episodic_rewards)),list_of_episodic_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward (score)")
    
    if log_wandb:
        wandb.log({"Episode Wise Total Reward Plot": [wandb.Image(plt, caption="")]})
    else:
        plt.show()
    
    f.clf()
    plt.close(f)
    return True


# In[11]:


def setup_and_start_expt(config,environment = "CartPole-v1",max_episodes=5000):
    
    seed = 76

    device_to_use = config["device_to_use"]
    
    env = gym.make(environment)
    env.reset(seed=seed)
    
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
        
    
    agent = ReinforceAgent(hyperparm_dict,state_size=state_shape,action_size = action_shape,seed = seed,baseline = config["baseline"])
    
    if environment == "Acrobot-v1":
        max_regret = -500
        cutoff = -100
    elif environment == "CartPole-v1":
        max_regret = 500
        cutoff = 30

    matplotlib.use('agg') ## using a non-interactive background framework for matplotlib, to enable smooth hyperparameter tuning.
    
    Reinforce_Experiment(env,agent,max_regret,n_episodes = max_episodes,log_wandb=True)
    


# In[12]:


def custom_testing():

    seed = 76
    
    device_to_use = "mps"
    
    environment = "Acrobot-v1"
    
    env = gym.make(environment)
    
    state_shape = env.observation_space.shape[0]
    action_shape = env.action_space.n
    
    hyperparm_dict = {

        "replay_buffer_size" : int(1e5),
        
        "batch_size" : 128,
    
        "target_net_update_freq" : 20,
        
        "policy_lr" : 5e-4,

        "value_lr" : 5e-4,
    
        "gamma" : 0.99,
    
        "policy_activation" : nn.ReLU(),

        "value_activation" : nn.ReLU(),
    
        "l2_regularization" : 0,
    
        "num_value_layers" : 2,
        
        "value_hidden_sizes" : [64,32],
        
        "num_policy_layers" : 1, 
        
        "policy_hidden_sizes" : [64],

        "value_optimiser" : "adam",
    
        "policy_optimiser" : "adam",

        "baseline" : True,

        'device_to_use' : "mps",

    }
    
    device = torch.device(device_to_use)
        
    #begin_time = datetime.datetime.now()
    
    
    agent = ReinforceAgent(hyperparm_dict,state_size=state_shape,action_size = action_shape,seed = seed,baseline = hyperparm_dict["baseline"])
    
    if environment == "Acrobot-v1":
        max_regret = -500
        cutoff = -100
    elif environment == "CartPole-v1":
        max_regret = 500
        cutoff = 40
    
    Reinforce_Experiment(env,agent,max_regret,n_episodes = 3000,seed=seed,log_wandb=False)



# In[64]:


#custom_testing()


# In[13]:


#wandb.login(key="")

sweep_config = {
    'method': 'random',
    'name' : 'REINFORCE CartPole',
    'metric': {
      'name': 'final_total_regret',
      'goal': 'minimize'
    },
    'parameters': {
        
        'num_value_layers': {
            'values': [1,2,3]
        },    
         'value_hidden_size':{
            'values':[32,64,128]
        },
        'num_policy_layers': {
            'values': [1,2,3]
        },    
         'policy_hidden_size':{
            'values':[32,64,128]
        },
        
        'policy_activation': {
            'values': ['relu','silu','selu','tanh']
        },

        'value_activation': {
            'values': ['relu','silu','selu','tanh']
        },

        
        'policy_optimiser': {
            'values': ["adam","rmsprop","nadam"]
        },

        'value_optimiser': {
            'values': ["adam","rmsprop","nadam"]
        },
        
        'policy_lr': {
            'values': [1e-4,3e-4,5e-4,1e-5]
        },
        'value_lr': {
            'values': [1e-4,3e-4,5e-4,1e-5]
        },
        
        'weight_decay': {
            'values': [0,0.005,1e-4,0.5]
        },

        "device_to_use" : {
            "values" : ["mps"]
        },

        "baseline" : {
            "values" : [True,False],
        },

        'batch_sizes': {
            'values': [32,64,128,256]
        },
            
        "target_net_update_freq" : {
            'values' : [5,10,15,20,25]
        
        }
    }

}

#sweep_id = wandb.sweep(sweep=sweep_config, project='JV_RL_PA2_HyperOpt_REINFORCE')


# In[14]:


"""def main():
    '''
    WandB calls main function each time with differnet combination.

    We can retrive the same and use the same values for our hypermeters.

    '''


    with wandb.init() as run:

        run_name="-vhl_"+str(wandb.config.num_value_layers)+"-vhs_"+str(wandb.config.value_hidden_size)+"-phl_"+str(wandb.config.num_policy_layers)+"-ahs_"+str(wandb.config.policy_hidden_size)+"-Pac_"+str(wandb.config.policy_activation)+"-Vac_"+str(wandb.config.value_activation)

        run_name = run_name+"-Poptim_"+str(wandb.config.policy_optimiser)+"-Voptim_"+str(wandb.config.value_optimiser)+"-Vlr_"+str(wandb.config.value_lr)+"-Plr_"+str(wandb.config.policy_lr)+"-reg_"+str(wandb.config.weight_decay)+"-Baseline_"+str(wandb.config.baseline)

        wandb.run.name=run_name

        setup_and_start_expt(wandb.config,environment='CartPole-v1',max_episodes=1500)
        

wandb.agent(sweep_id, function=main,count=100) # calls main function for count number of times.
wandb.finish()"""


# In[ ]:





# In[18]:


#wandb.login(key="")

sweep_config = {
    'method': 'random',
    'name' : 'REINFORCE Acrobot',
    'metric': {
      'name': 'final_total_regret',
      'goal': 'minimize'
    },
    'parameters': {
        
        'num_value_layers': {
            'values': [1,2,3]
        },    
         'value_hidden_size':{
            'values':[32,64,128]
        },
        'num_policy_layers': {
            'values': [1,2,3]
        },    
         'policy_hidden_size':{
            'values':[32,64,128]
        },
        
        'policy_activation': {
            'values': ['relu','silu','selu','tanh']
        },

        'value_activation': {
            'values': ['relu','silu','selu','tanh']
        },

        
        'policy_optimiser': {
            'values': ["adam","rmsprop","nadam"]
        },

        'value_optimiser': {
            'values': ["adam","rmsprop","nadam"]
        },
        
        'policy_lr': {
            'values': [1e-4,3e-4,5e-4,1e-5]
        },
        'value_lr': {
            'values': [1e-4,3e-4,5e-4,1e-5]
        },
        
        'weight_decay': {
            'values': [0,0.005,1e-4,0.5]
        },

        "device_to_use" : {
            "values" : ["mps"]
        },

        "baseline" : {
            "values" : [True,False],
        },

        'batch_sizes': {
            'values': [32,64,128,256]
        },
            
        "target_net_update_freq" : {
            'values' : [5,10,15,20,25]
        
        }
    }

}

#sweep_id = wandb.sweep(sweep=sweep_config, project='JV_RL_PA2_HyperOpt_REINFORCE')


# In[20]:


"""def main():
    '''
    WandB calls main function each time with differnet combination.

    We can retrive the same and use the same values for our hypermeters.

    '''
    with wandb.init() as run:

        run_name="-vhl_"+str(wandb.config.num_value_layers)+"-vhs_"+str(wandb.config.value_hidden_size)+"-phl_"+str(wandb.config.num_policy_layers)+"-ahs_"+str(wandb.config.policy_hidden_size)+"-Pac_"+str(wandb.config.policy_activation)+"-Vac_"+str(wandb.config.value_activation)

        run_name = run_name+"-Poptim_"+str(wandb.config.policy_optimiser)+"-Voptim_"+str(wandb.config.value_optimiser)+"-Vlr_"+str(wandb.config.value_lr)+"-Plr_"+str(wandb.config.policy_lr)+"-reg_"+str(wandb.config.weight_decay)+"-Baseline_"+str(wandb.config.baseline)

        wandb.run.name=run_name

        setup_and_start_expt(wandb.config,environment='Acrobot-v1',max_episodes=1500)
   

wandb.agent(sweep_id, function=main,count=100) # calls main function for count number of times.
wandb.finish()"""


# In[ ]:

# In[ ]:




