from tabulate import tabulate

reinforce_cp_baseline_best_config = {

        "baseline" : True,

        'batch_sizes': 256,

        "device_to_use" : "mps",

        'num_policy_layers': 3,
        
        'num_value_layers': 1,
        
        'policy_activation': 'selu',

        'policy_hidden_size':64,

        'policy_lr': 3e-4,

        'policy_optimiser': "nadam",

        "target_net_update_freq" : 25,

        'value_activation': 'selu',

        'value_hidden_size':64,

        'value_lr': 1e-4,

        'value_optimiser': "rmsprop",
        
        'weight_decay': 0

}

reinforce_cp_no_baseline_best_config = {

        "baseline" : False,

        'batch_sizes': 32,

        "device_to_use" : "mps",

        'num_policy_layers': 1,
        
        'num_value_layers': 3,
        
        'policy_activation': 'selu',

        'policy_hidden_size':128,

        'policy_lr': 5e-4,

        'policy_optimiser': "rmsprop",

        "target_net_update_freq" : 10,

        'value_activation': 'tanh',

        'value_hidden_size':32,

        'value_lr': 1e-5,

        'value_optimiser': "nadam",
        
        'weight_decay': 0
}


table = [(key, val) for key, val in sorted(reinforce_cp_baseline_best_config.items())]
latex_table = tabulate(table, headers=['HyperParameter', 'Value'], tablefmt='latex')

algo = "REINFORCE with Baseline"
env = "CartPole-v1"
with open("./Reinforce_CartPole_Baseline.tex", "w") as f:
        for i in latex_table:
                f.write(i)


########################################################################

table = [(key, val) for key, val in sorted(reinforce_cp_no_baseline_best_config.items())]
latex_table = tabulate(table, headers=['HyperParameter', 'Value'], tablefmt='latex')

algo = "REINFORCE without Baseline"
env = "CartPole-v1"
with open("./Reinforce_CartPole_No_Baseline.tex", "w") as f:
        for i in latex_table:
                f.write(i)