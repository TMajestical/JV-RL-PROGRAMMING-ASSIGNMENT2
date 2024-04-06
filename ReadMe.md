JV

CS23M036, Malladi Tejasvi

This is the repo for the CS6700, Reinforcement Learning, IIT Madras, Programming assignment 2.

The file JV_REINFORCE.py has all the core REINFORCE algorithm code and also can perform hyperparameter tuning using WandB. Please add your login key and uncomment the wandb.login,wandb.sweep and main() method to use wandb.
To train with a custom hyperparameter configuration, configure the specific hyper parameters the custom_testing() method and invoke it, make sure that the main and wandb.login and wandb.sweep remain commented.

Simliar steps apply to the JV_Duelling_DQN.py code, the heart of which is the Duelling DQN algorithm.

JV_Generate_Results_For_Report.py reads picks the best hyperparmeter configuration from *_Best_Config.py (that was found after hyperparameter tuning) and averages the rewards and regrets over a list of seeds. This code recreates the plots used in the report.

Uncomment the invocation to generate_ddqn_report() and/or generate_reinforce_report(), as required.
