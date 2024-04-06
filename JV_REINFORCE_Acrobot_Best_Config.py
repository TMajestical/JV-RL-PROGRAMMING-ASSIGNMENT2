from tabulate import tabulate

search_space = {
        
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


reinforce_ab_baseline_best_config = {

        "baseline" : True,

        'batch_sizes': 32,

        "device_to_use" : "mps",

        'num_policy_layers': 3,
        
        'num_value_layers': 3,
        
        'policy_activation': 'selu',

        'policy_hidden_size':32,

        'policy_lr': 5e-4,

        'policy_optimiser': "nadam",

        "target_net_update_freq" : 10,

        'value_activation': 'tanh',

        'value_hidden_size':32,

        'value_lr': 1e-4,

        'value_optimiser': "nadam",
        
        'weight_decay': 0.005

}

reinforce_ab_no_baseline_best_config = {

        "baseline" : False,

        'batch_sizes': 64,

        "device_to_use" : "mps",

        'num_policy_layers': 2,
        
        'num_value_layers': 2,
        
        'policy_activation': 'tanh',

        'policy_hidden_size':64,

        'policy_lr': 5e-4,

        'policy_optimiser': "rmsprop",

        "target_net_update_freq" : 20,

        'value_activation': 'tanh',

        'value_hidden_size':32,

        'value_lr': 3e-4,

        'value_optimiser': "nadam",
        
        'weight_decay': 1e-4
}


table = [(key, val) for key, val in sorted(reinforce_ab_baseline_best_config.items())]
latex_table = tabulate(table, headers=['HyperParameter', 'Value'], tablefmt='latex')

algo = "REINFORCE with Baseline"
env = "Acrobot-v1"
with open("./Reinforce_Acrobot_Baseline.tex", "w") as f:
        for i in latex_table:
                f.write(i)

########################################################################

table = [(key, val) for key, val in sorted(reinforce_ab_no_baseline_best_config.items())]
latex_table = tabulate(table, headers=['HyperParameter', 'Value'], tablefmt='latex')

algo = "REINFORCE without Baseline"
env = "Acrobot-v1"
with open("./Reinforce_Acrobot_No_Baseline.tex", "w") as f:
        for i in latex_table:
                f.write(i)

########################################################################
table = [(key, val["values"]) for key, val in sorted(search_space.items())]
latex_table = tabulate(table, headers=['HyperParameter', 'Values'], tablefmt='latex')

algo = "Search Space for REINFORCE ALgorithm"
with open("./REINFORCE_Search_Space.tex", "w") as f:
        for i in latex_table:
                f.write(i)
        

