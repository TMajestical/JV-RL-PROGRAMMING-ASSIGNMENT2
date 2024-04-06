#!/usr/bin/env python
# coding: utf-8

# In[1]:


#JV


# In[2]:


import numpy as np
import gym
import random
import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from collections import deque, namedtuple
import datetime

import matplotlib.pyplot as plt
from tqdm import tqdm

import wandb

import matplotlib

# In[3]:


gym.__version__


# ## Deep Dueling Q-Network (DDQN)

# In[4]:


class DDQN(nn.Module):

    def __init__(self, state_size, action_size, seed,aggregation_type="mean", activation = nn.ReLU(), num_shared_layers = 1, shared_hidden_sizes = [64],
                 num_value_layers = 2, value_hidden_sizes = [256,1],num_advantage_layers = 2, advantage_hidden_sizes = [256,2]):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            aggregation_type(str) : "mean" or "max", to enforce identifiability.
            num_shared_layers(int) : Default 1,  it is the number of hidden layers common to both value and advantage.
            shared_hidden_sizes (list of int) : default [64], List of number of neurons per hidden layer.
            num_value_layers(int) : Default, 2. Number of hidden layers for value function. 
            value_hidden_sizes (list of int) : default [256,1],List of number of neurons per hidden layer.
            num_advantage_layers (int) : Default 2. Number of hidden layers for value function.
            advantage_hidden_sizes (list of int) : [256,2], List of number of neurons per hidden layer.
        """
        
        super(DDQN, self).__init__()

        self.activation = activation
        
        self.seed = torch.manual_seed(seed)

        shared_layers = self._create_layers(state_size, shared_hidden_sizes)
        self.shared_layers = nn.Sequential(*shared_layers)
        #self.shared_layers.apply(self.initalize_weights_biases)

        value_layers = self._create_layers(shared_hidden_sizes[-1],value_hidden_sizes)
        self.value_layers = nn.Sequential(*value_layers)
        #self.value_layers.apply(self.initalize_weights_biases)

        advantage_layers = self._create_layers(shared_hidden_sizes[-1],advantage_hidden_sizes)
        self.advantage_layers = nn.Sequential(*advantage_layers)
        #self.advantage_layers.apply(self.initalize_weights_biases)
        
        self.aggregation_type = aggregation_type

    def _create_layers(self, state_size, hidden_sizes):
        
        layers = []
        sizes = [state_size] + hidden_sizes
        
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i+1]))
            layers.append(self.activation)
            
        return layers

    def forward(self, state):
        """Build a network that maps state -> action values."""
        
        x = self.shared_layers(state)
        
        x_val = self.value_layers(x)
        
        x_adv = self.advantage_layers(x)

        if self.aggregation_type == "mean":
            
            aggregation =  torch.add(x_val,torch.sub(x_adv,x_adv.mean())) 

        elif self.aggregation_type == "max":

            aggregation =  torch.add(x_val,torch.sub(x_adv,x_adv.max()))
        
        return aggregation
    
    def initalize_weights_biases(self,m):
        """
        
        Method to initialize weights given a torch module.

        Using "Xavier" Initialization, as it goes well with ReLU in CNN.
        
        """
        if isinstance(m, nn.Linear):  ## it its a fully connected layer
            torch.nn.init.xavier_normal_(m.weight)
            m.bias.data.fill_(0.01) ##a small non-zero value


# In[5]:


## Running on apple silicon GPU

#device = torch.device("mps") #if torch.cuda.is_available() else "cpu")

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


# In[57]:


class DDQN_Agent_egreedy():

    def __init__(self,hyperparm_dict, state_size, action_size,seed,aggregation_type="mean",device="cpu"):

        ''' Agent Environment Interaction '''
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.device = device
        self.hyperparm_dict = hyperparm_dict

        ''' Q-Network '''
        self.qnetwork_local = DDQN(state_size, action_size, seed, aggregation_type, activation = self.hyperparm_dict["activation"], 
                                   num_shared_layers = self.hyperparm_dict["num_shared_layers"], shared_hidden_sizes = self.hyperparm_dict["shared_hidden_sizes"], 
                                   num_value_layers = self.hyperparm_dict["num_value_layers"], value_hidden_sizes = self.hyperparm_dict["value_hidden_sizes"],
                                   num_advantage_layers = self.hyperparm_dict["num_advantage_layers"], advantage_hidden_sizes = self.hyperparm_dict["advantage_hidden_sizes"]).to(self.device)
        self.qnetwork_target = DDQN(state_size, action_size, seed,aggregation_type, activation = self.hyperparm_dict["activation"], 
                                    num_shared_layers = self.hyperparm_dict["num_shared_layers"], shared_hidden_sizes = self.hyperparm_dict["shared_hidden_sizes"], 
                                    num_value_layers = self.hyperparm_dict["num_value_layers"], value_hidden_sizes = self.hyperparm_dict["value_hidden_sizes"],
                                    num_advantage_layers = self.hyperparm_dict["num_advantage_layers"], advantage_hidden_sizes = self.hyperparm_dict["advantage_hidden_sizes"]).to(self.device)
        
        self.qnetwork_local.apply(self.qnetwork_local.initalize_weights_biases)
        self.qnetwork_target.apply(self.qnetwork_target.initalize_weights_biases)

        self.qnetwork_local.to(self.device)
        self.qnetwork_target.to(self.device)

        if self.hyperparm_dict["optimiser"] == "adam":
            self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.hyperparm_dict["lr"],weight_decay=self.hyperparm_dict["l2_regularization"])
        elif self.hyperparm_dict["optimiser"] == "nadam":
            self.optimizer = optim.NAdam(self.qnetwork_local.parameters(), lr=self.hyperparm_dict["lr"],weight_decay=self.hyperparm_dict["l2_regularization"])
        elif self.hyperparm_dict["optimiser"] == "rmsprop":
            self.optimizer = optim.RMSprop(self.qnetwork_local.parameters(), lr=self.hyperparm_dict["lr"],weight_decay=self.hyperparm_dict["l2_regularization"])

        ''' Replay memory '''
        self.memory = ReplayBuffer(action_size,self.hyperparm_dict, seed, self.device)

        ''' Initialize time step (for updating every UPDATE_EVERY steps)           -Needed for Q Targets '''
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):

        ''' Save experience in replay memory '''
        self.memory.add(state, action, reward, next_state, done)

        ''' If enough samples are available in memory, get random subset and learn '''
        if len(self.memory) >= self.hyperparm_dict["batch_size"]:
            experiences = self.memory.sample()
            self.learn(experiences, self.hyperparm_dict["gamma"])

        """ +Q TARGETS PRESENT """
        ''' Updating the Network every 'UPDATE_EVERY' steps taken '''
        self.t_step = (self.t_step + 1) % self.hyperparm_dict["target_net_update_freq"]
        if self.t_step == 0:

            self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())

    def act(self, state, eps=0.):

        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        ''' Epsilon-greedy action selection (Already Present) '''
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """ +E EXPERIENCE REPLAY PRESENT """
        states, actions, rewards, next_states, dones = experiences

        ''' Get max predicted Q values (for next states) from target model'''
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)

        ''' Compute Q targets for current states '''
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        ''' Get expected Q values from local model '''
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        ''' Compute loss '''
        loss = F.mse_loss(Q_expected, Q_targets)

        loss.to(self.device)

        ''' Minimize the loss '''
        self.optimizer.zero_grad()
        loss.backward()

        ''' Gradiant Clipping '''
        """ +T TRUNCATION PRESENT """
        for param in self.qnetwork_local.parameters():
            param.grad.data.clamp_(-1, 1)

        self.optimizer.step()


# In[58]:


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


# In[59]:


def ddqn_egreedy(env,agent,max_regret = 1000,cutoff = 30,n_episodes=5000,max_t=500, eps_start=0.5, eps_end=0.01, eps_decay=0.995,seed=76, log_wandb = False):


    if log_wandb:
        matplotlib.use('agg') ## using a non-interactive background framework for matplotlib, to enable smooth hyperparameter tuning.

    early_stopping = 0

    list_of_episodic_returns = []

    average_episodic_regret = 0 ##initialize, use running average

    total_regret = 0

    regret_window = deque(maxlen=100) ##last 100 regret values for checking if the avg is less than 20.

    scores_window = deque(maxlen=100)
    
    eps = eps_start ## Initialize epsilon

    
    for i_episode in tqdm(range(1, n_episodes+1)):
        state = env.reset(seed=seed)[0]
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, done, _, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break

        cur_episode_regret = max_regret - score
        scores_window.append(score)
        regret_window.append(cur_episode_regret)
        total_regret += cur_episode_regret

        list_of_episodic_returns.append(score)

        #average_episodic_regret =  (cur_episode_regret - average_episodic_regret)/i_episode + average_episodic_regret

        eps = max(eps_end, eps_decay*eps)
        ''' decrease epsilon '''

        if i_episode % 100 == 0:
            
            if not log_wandb:
                print('\rEpisode {}\tTotal Regret: {:.2f}\t Average Regret: {:.2f}\t Average Score: {:.2f}'.format(i_episode, total_regret,np.mean(regret_window),np.mean(scores_window)))
            else:
                wandb.log({'total_regret':total_regret,'regret_in_window': np.mean(regret_window),'Total Episodes':i_episode})
        
        if (max_regret>0 and np.mean(regret_window)<=cutoff):
            
            if not log_wandb:
                print('\nEnvironment solved in {:d} episodes!\tRegret in Window: {:.2f}'.format(i_episode, np.mean(regret_window)))            
            else:
                wandb.log({'final_total_regret':total_regret,'final_regret_in_window': np.mean(regret_window),'Total Episodes':i_episode})
            
            early_stopping = 1
            break

        elif max_regret<0 and ((np.mean(regret_window)<=-400)or (i_episode==1500 and np.mean(regret_window)>-300) or (i_episode==1000 and np.mean(regret_window)>-150) or (i_episode==500 and np.mean(regret_window)>-50) or (i_episode==150 and np.mean(regret_window)==0)):

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
    plt.plot(np.arange(len(list_of_episodic_returns)),list_of_episodic_returns)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward (score)")
    #plt.title("Episode Wise Total Reward Plot")

    if log_wandb:
        wandb.log({"Episode Wise Total Reward Plot": [wandb.Image(plt, caption="")]})
    else:
        plt.show()

    f.clf()
    plt.close(f)

    del list_of_episodic_returns
    del regret_window
    del agent.memory
    gc.collect()

    return True


# In[ ]:





# In[60]:


def setup_and_start_expt(config,environment = "Acrobot-v1",max_episodes=5000):
    
    seed = 76

    device_to_use = config["device_to_use"]

    if config["activation"] == "silu":

        device_to_use = "cpu" ## silu is computationally expensive and slows down the GPU operations
    
    aggregation = config["aggregation"]
    
    env = gym.make(environment)
    env.reset(seed=seed)
    
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
    
    #begin_time = datetime.datetime.now()
    
    
    agent = DDQN_Agent_egreedy(hyperparm_dict,state_size=state_shape,action_size = action_shape,seed = seed,aggregation_type=aggregation)
    
    if environment == "Acrobot-v1":
        max_regret = -500
        cutoff = -470
    elif environment == "CartPole-v1":
        max_regret = 500
        cutoff = 30
    
    ddqn_egreedy(env,agent,max_regret,n_episodes = max_episodes, eps_start=config["start_eps"],eps_decay=config["eps_decay"],seed=seed,log_wandb=True)
    
    #time_taken = datetime.datetime.now() - begin_time
    
    #print(time_taken)


# In[63]:

def custom_testing():

    seed = 76
    
    device_to_use = "mps"
    
    aggregation = "mean"
    
    environment = "Acrobot-v1"
    
    env = gym.make(environment)
    #env.re(seed)
    
    state_shape = env.observation_space.shape[0]
    action_shape = env.action_space.n
    
    hyperparm_dict = {
        "replay_buffer_size" : int(5e3),
        
        "batch_size" : 32,
    
        "lr" : 1e-3,
    
        "target_net_update_freq" : 20,
    
        "gamma" : 0.99,
    
        "activation" : nn.ReLU(),
    
        "l2_regularization" : 5e-4,
    
        "num_shared_layers" : 3, 
        
        "shared_hidden_sizes" : [128]*3,
        
        "num_value_layers" : 2,
        
        "value_hidden_sizes" : [32]+[1],
        
        "num_advantage_layers" : 1, 
        
        "advantage_hidden_sizes" : [128]+[action_shape],
    
        "optimiser" : "rmsprop"


        
    }
    
    device = torch.device(device_to_use)
        
    #begin_time = datetime.datetime.now()
    
    
    agent = DDQN_Agent_egreedy(hyperparm_dict,state_size=state_shape,action_size = action_shape,seed = seed,aggregation_type=aggregation)
    
    if environment == "Acrobot-v1":
        max_regret = -500
        cutoff = -100
    elif environment == "CartPole-v1":
        max_regret = 500
        cutoff = 30
    
    ddqn_egreedy(env,agent,max_regret,n_episodes = 2000, eps_start=0.9,eps_decay=0.996 ,seed=seed,log_wandb=False)





# In[64]:


#custom_testing()


# In[10]:


#n(key="38853ce9d1432bd40bf80d2d27657183fc335aeb")


# In[11]:


sweep_config = {
    'method': 'bayes',
    'name' : 'DDQN Mean Acrobot',
    'metric': {
      'name': 'Average Test Regret',
      'goal': 'minimize'
    },
    'parameters': {
        'num_shared_layers': {
            'values': [1,2,3]
        },    
         'shared_hidden_size':{
            'values':[32,64,128]
        },
        'num_value_layers': {
            'values': [1,2,3]
        },    
         'value_hidden_size':{
            'values':[32,64,128]
        },
        'num_advantage_layers': {
            'values': [1,2,3]
        },    
         'advantage_hidden_size':{
            'values':[32,64,128]
        },
        
        'activation': {
            'values': ['relu','silu','tanh']
        },

        
        'optimiser': {
            'values': ["adam","rmsprop","nadam"]
        },
        
        
        'batch_sizes': {
            'values': [32,64,128,256]
        },
        
        'lr': {
            'values': [1e-3,1e-4,3e-4,5e-4,1e-5]
        },
        'weight_decay': {
            'values': [0,0.005,5e-4,0.05,0.5]
        },

        "target_net_update_freq" : {
            'values' : [5,10,15,20,25]
        },

        "replay_buffer_size" : {
            'values' : [int(5e3),int(1e4),int(1e5)]
            },

        "device_to_use" : {
            "values" : ["mps"]
        },

        "aggregation" : {
            "values" : ["mean"]
        },

        "start_eps" : {
            "values" : [0.7,0.8,0.9]
        },

        "eps_decay" : {
            "values" : [0.996,0.998]
        }
        
        }
    }


# In[12]:


#sweep_id = wandb.sweep(sweep=sweep_config, project='RL_PA2_HyperOpt_DDQN')


# In[17]:


"""def main():
    '''
    WandB calls main function each time with differnet combination.

    We can retrive the same and use the same values for our hypermeters.

    '''


    with wandb.init() as run:

        run_name="-shl_"+str(wandb.config.num_shared_layers)+"-shs_"+str(wandb.config.shared_hidden_size)+"-vhl_"+str(wandb.config.num_value_layers)+"-vhs_"+str(wandb.config.value_hidden_size)+"-ahl_"+str(wandb.config.num_advantage_layers)+"-ahs_"+str(wandb.config.advantage_hidden_size)+"-ac_"+str(wandb.config.activation)

        run_name = run_name+"-optim_"+str(wandb.config.optimiser)+"-lr_"+str(wandb.config.lr)+"-bs_"+str(wandb.config.batch_sizes)+"-reg_"+str(wandb.config.weight_decay)+"-tUP_"+str(wandb.config.target_net_update_freq)

        run_name = run_name+"-Strt_eps_"+str(wandb.config.start_eps)+"-Eps_DC_"+str(wandb.config.eps_decay)

        wandb.run.name=run_name

        setup_and_start_expt(wandb.config,max_episodes=2000)
        

wandb.agent(sweep_id, function=main,count=200) # calls main function for count number of times.
wandb.finish()"""

# In[ ]:





# In[14]:


def print_network_architecture(model):
    print(model)
    print("Network Architecture:")
    for name, module in model.named_children():
        print("-" * 40)
        print(f"Layer Name: {name}")
        print(module)
        print("-" * 40)


