import numpy as np
import matplotlib as mpl
from random import seed
from reservoir_env import Reservoir_continuous, plot_continuous, Reservoir_discrete, plot_discrete
from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MultiInputPolicy

mpl.use('tkAgg')

#############################################################################
### Train PPO continuous policy for reservoir operations
#############################################################################

### set random seed for consistent results
rseed = 5
seed(rseed)

### initialize reservoir & set initial state
reservoir = Reservoir_continuous()
obs = reservoir.reset()

### Implement PPO with default parameterization
model = PPO("MultiInputPolicy", reservoir, verbose=1)

### return a trained model that is trained over 100,000 timesteps
model.learn(total_timesteps=100_000)
policy = model.policy

### save trained policy to disk
policy.save('trained_models/ppo_policy.pkl')


### get trained policy from disk
loaded_policy = MultiInputPolicy.load('trained_models/ppo_policy.pkl')

### run 5 seeds of alternative inflows & plot behavior
for rseed in range(5):
    seed(rseed)

    ### lists for storing state variables, actions, & rewards
    obs = reservoir.reset()
    storages, targets, inflows, releases, rewards = [obs['storage'][0]], [obs['target'][0]], [], [], []

    ### loop through daily simulation until the reservoir environment terminates after set number of time steps
    terminated = False
    while not terminated:
        ### use a constant release that is slightly different from inflow
        release, _states = loaded_policy.predict(obs, deterministic=True)

        ### update state of reservoir with release & get reward
        obs, reward, terminated, info = reservoir.step(release)
        ### save state variables, actions, & rewards for plotting
        storages.append(obs['storage'][0])
        targets.append(obs['target'][0])
        inflows.append(info['inflow'])
        releases.append(info['release'])
        rewards.append(reward)

    ### plot reservoir storage, target, inflow, release, & reward time series
    plot_continuous(reservoir, storages, targets, inflows, releases, rewards, f'figs/ppo_continuous_{rseed}.png')

    reservoir.close()


