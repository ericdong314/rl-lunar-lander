# Lunar Lander DQN

## Folder Structure
`data` - Training data saved in each training for plotting graphs

`models` - The models of agents saved before and after training

`plotting` - Graphs plotted in the project

`epoch` - supplymentary code for training and saving data in epoch


plotting-DQNs.ipynb is the jupyter notebook used for plotting all the DQN related and DQN-PPO comparison graphs.

The major code to run is lunar_lander_dqn.py where different DQN variants are implemented.

## Dependencies
Install the following libraries using `pip install`

- torch==2.0.0
- gymnasium==0.28.1
- gymnasium["LunarLander]
- argparse==1.4.0


## Arguments
Run the file by `python lunar_lander_dqn.py`. With optional parameters:

- `num_episodes`: number of episodes to train in a training loop (500 by default)
- `num_agents`: number of agents to train for each category (4 by default)
