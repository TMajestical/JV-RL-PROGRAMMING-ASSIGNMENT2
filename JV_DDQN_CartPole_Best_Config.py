cp_type1_best_config = {
        'num_shared_layers': 2,

        'shared_hidden_size':128,

        'num_value_layers': 3,

        'value_hidden_size':32,

        'num_advantage_layers': 1,

        'advantage_hidden_size':128,
        
        'activation': 'selu',

        'optimiser': "rmsprop",
        
        'batch_sizes': 256,
        
        'lr': 3e-4,

        'weight_decay': 5e-3,

        "target_net_update_freq" : 25,

        "replay_buffer_size" : int(1e5),

        "device_to_use" : "mps",

        "aggregation" : "mean",

        "start_eps" : 0.5,

        "eps_decay" : 0.995
        }

cp_type2_best_config = {
        'num_shared_layers': 2,

        'shared_hidden_size':64,

        'num_value_layers': 3,

        'value_hidden_size':128,

        'num_advantage_layers': 3,

        'advantage_hidden_size':128,
        
        'activation': 'selu',

        'optimiser': "rmsprop",
        
        'batch_sizes': 128,
        
        'lr': 1e-4,

        'weight_decay': 5e-3,

        "target_net_update_freq" : 25,

        "replay_buffer_size" : int(1e5),

        "device_to_use" : "mps",

        "aggregation" : "max",

        "start_eps" : 0.5,

        "eps_decay" : 0.995
        }

from tabulate import tabulate

table = [(key, val) for key, val in sorted(cp_type1_best_config.items())]
latex_table = tabulate(table, headers=['HyperParameter', 'Value'], tablefmt='latex')

algo = "Duelling DQN Type1"
env = "CartPosle-v1"
with open("./DDQN_CartPole_Type1.tex", "w") as f:
        for i in latex_table:
                f.write(i)

#########

table = [(key, val) for key, val in sorted(cp_type2_best_config.items())]
latex_table = tabulate(table, headers=['HyperParameter', 'Value'], tablefmt='latex')

algo = "Duelling DQN Type2"
env = "CartPosle-v1"
with open("./DDQN_CartPole_Type2.tex", "w") as f:
        for i in latex_table:
                f.write(i)