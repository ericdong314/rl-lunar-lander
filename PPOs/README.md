# Lunar Lander PPO

## Folder Structure
- the 2 algorithms that you will run are `lunar_lander_ppo_gae_lambda.py` and `lunar_lander_ppo_td_residual.py`
- `utils/` - contains helper function(s) used by the script
- `saved_models/` - where the model is saved during and at the end of training
- `data` - where the output data (rewards, episode length etc) of training are saved

## Dependencies
Install the following libraries using `pip install`

- torch==2.0.0
- gymnasium==0.28.1
- gymnasium["LunarLander]
- argparse==1.4.0

## Arguments
Run the file simply by `python [file_name]` e.g. `python lunar_lander_ppo_gae_lambda.py`. Change the paramters by passing in commmand line arguments. The useful ones are listed below:

- `load_model`: load pre-trained model to the agent
- `eval_mode`: if specified as 1, the algorithm will run the env indefinitely without learning
- `num_agent`: number of agents to train. 
- `lr`: learning rate
- `total_timesteps`: the total timesteps to sample from throughout the entire training for an agent
- `seed`: seed for pseurandom generateors. To make the results reproducible, you also have to pass the seed to the calls for envs.reset()
- `num_envs`: number of environments to run in parallel
- `num_steps`: number of steps to step for each parallel env in a rollout phase
- `anneal_lr`: learning rate decay
- `gae_lambda`: lambda for GAE computation
- `gamma`: gamma used in computing return and GAE
- `num_minibatches`: number of minibatches
- `training_iter`: how many times the sample steps are reused for training
- `clip_coef`: clipping range in loss function