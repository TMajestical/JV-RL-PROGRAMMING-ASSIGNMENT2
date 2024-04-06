from tabulate import tabulate

search_space = {
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
            "values" : ["mean","max"]
        },

        "start_eps" : {
            "values" : [0.5,0.7,0.8,0.9]
        },

        "eps_decay" : {
            "values" : [0.995,0.996,0.998]
        }
        
        }


ab_type1_best_config = {
        'num_shared_layers': 1,

        'shared_hidden_size':32,

        'num_value_layers': 3,

        'value_hidden_size':64,

        'num_advantage_layers': 3,

        'advantage_hidden_size':32,
        
        'activation': 'relu',

        'optimiser': "rmsprop",
        
        'batch_sizes': 128,
        
        'lr': 5e-4,

        'weight_decay': 0,

        "target_net_update_freq" : 20,

        "replay_buffer_size" : int(1e5),

        "device_to_use" : "mps",

        "aggregation" : "mean",

        "start_eps" : 0.5,

        "eps_decay" : 0.995
        }

ab_type2_best_config = {
        'num_shared_layers': 3,

        'shared_hidden_size':64,

        'num_value_layers': 2,

        'value_hidden_size':128,

        'num_advantage_layers': 2,

        'advantage_hidden_size':64,
        
        'activation': 'silu',

        'optimiser': "nadam",
        
        'batch_sizes': 256,
        
        'lr': 5e-4,

        'weight_decay': 5e-4,

        "target_net_update_freq" : 20,

        "replay_buffer_size" : int(1e4),

        "device_to_use" : "mps",

        "aggregation" : "max",

        "start_eps" : 0.8,

        "eps_decay" : 0.996
        }

table = [(key, val) for key, val in sorted(ab_type1_best_config.items())]
latex_table = tabulate(table, headers=['HyperParameter', 'Value'], tablefmt='latex')

algo = "Duelling DQN Type1"
env = "Acrobot-v1"
with open("./DDQN_Acrobot_Type1.tex", "w") as f:
        for i in latex_table:
                f.write(i)

########################################################################

table = [(key, val) for key, val in sorted(ab_type2_best_config.items())]
latex_table = tabulate(table, headers=['HyperParameter', 'Value'], tablefmt='latex')

algo = "Duelling DQN Type2"
env = "Acrobot-v1"
with open("./DDQN_Acrobot_Type2.tex", "w") as f:
        for i in latex_table:
                f.write(i)
        
########################################################################
table = [(key, val["values"]) for key, val in sorted(search_space.items())]
latex_table = tabulate(table, headers=['HyperParameter', 'Values'], tablefmt='latex')

algo = "Search Space for DDQN Algorithm"
with open("./DDQN_Search_Space.tex", "w") as f:
        for i in latex_table:
                f.write(i)
        


