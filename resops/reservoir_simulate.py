import numpy as np
import matplotlib as mpl
from random import seed
from reservoir_env import Reservoir_continuous, plot_continuous, Reservoir_discrete, plot_discrete

# mpl.use('tkAgg')

#############################################################################
### first simulate continuous-version reservoir under constant releases
#############################################################################

### set random seed for consistent results
rseed = 5
seed(rseed)

### initialize reservoir & set initial state
reservoir = Reservoir_continuous()
obs = reservoir.reset()

### lists for storing state variables, actions, & rewards
storages, targets, inflows, releases, rewards = [obs['storage'][0]], [obs['target'][0]], [], [], []

### loop through daily simulation until the reservoir environment terminates after set number of time steps
terminated = False
while not terminated:
    ### use a constant release that is slightly different from inflow
    release = [reservoir.inflow_base * 1.01]
    ### update state of reservoir with release & get reward
    obs, reward, terminated, info = reservoir.step(release)
    ### save state variables, actions, & rewards for plotting
    storages.append(obs['storage'][0])
    targets.append(obs['target'][0])
    inflows.append(info['inflow'])
    releases.append(info['release'])
    rewards.append(reward)

### plot reservoir storage, target, inflow, release, & reward time series
plot_continuous(reservoir, storages, targets, inflows, releases, rewards, 'figs/constant_continuous.png')

reservoir.close()




#############################################################################
### now simulate continuous-version reservoir under random releases
#############################################################################

### set random seed for consistent results
seed(rseed)

### initialize reservoir & set initial state
reservoir = Reservoir_continuous()
obs = reservoir.reset()

### lists for storing state variables, actions, & rewards
storages, targets, inflows, releases, rewards = [obs['storage'][0]], [obs['target'][0]], [], [], []

### loop through daily simulation until the reservoir environment terminates after set number of time steps
terminated = False
while not terminated:
    ### generate a random release
    release = reservoir.action_space.sample()
    ### update state of reservoir with release & get reward
    obs, reward, terminated, info = reservoir.step(release)
    ### save state variables, actions, & rewards for plotting
    storages.append(obs['storage'][0])
    targets.append(obs['target'][0])
    inflows.append(info['inflow'])
    releases.append(info['release'])
    rewards.append(reward)

### plot reservoir storage, target, inflow, release, & reward time series
plot_continuous(reservoir, storages, targets, inflows, releases, rewards, 'figs/random_continuous.png')

reservoir.close()




#############################################################################
### now simulate discrete-version reservoir under constant releases
#############################################################################

### set random seed for consistent results
seed(rseed)

### initialize reservoir & set initial state
reservoir = Reservoir_discrete()
obs = reservoir.reset()

### lists for storing state variables, actions, & rewards
storages_discrete, targets_discrete, storages_cont, targets_cont, inflows, releases, rewards = \
    [obs['storage_discrete'][0]], [obs['storage_discrete'][0]], [], [], [], [], []

### loop through daily simulation until the reservoir environment terminates after set number of time steps
terminated = False
while not terminated:
    ### use a constant release that is slightly different from inflow
    release_discrete = np.argmin(np.abs(reservoir.grid_release - 1.))
    ### update state of reservoir with release & get reward
    obs, reward, terminated, info = reservoir.step(release_discrete)
    ### save state variables, actions, & rewards for plotting
    storages_discrete.append(obs['storage_discrete'][0])
    targets_discrete.append(obs['target_discrete'][0])
    storages_cont.append(info['storage_cont'])
    targets_cont.append(info['target_cont'])
    inflows.append(info['inflow'])
    releases.append(info['release'])
    rewards.append(reward)

### plot reservoir storage, target, inflow, release, & reward time series
plot_discrete(reservoir, storages_cont, targets_cont, storages_discrete, targets_discrete, inflows, releases, rewards, 'figs/constant_discrete.png')

reservoir.close()




#############################################################################
### now simulate discrete-version reservoir under random releases
#############################################################################

### set random seed for consistent results
seed(rseed)

### initialize reservoir & set initial state
reservoir = Reservoir_discrete()
obs = reservoir.reset()

### loop through daily simulation until the reservoir environment terminates after set number of time steps
storages_discrete, targets_discrete, storages_cont, targets_cont, inflows, releases, rewards = [obs['storage_discrete']], [obs['storage_discrete']], [], [], [], [], []

terminated = False
while not terminated:
    ### generate a random release
    release_discrete = reservoir.action_space.sample()
    ### update state of reservoir with release & get reward
    obs, reward, terminated, info = reservoir.step(release_discrete)
    ### save state variables, actions, & rewards for plotting
    storages_discrete.append(obs['storage_discrete'])
    targets_discrete.append(obs['target_discrete'])
    storages_cont.append(info['storage_cont'])
    targets_cont.append(info['target_cont'])
    inflows.append(info['inflow'])
    releases.append(info['release'])
    rewards.append(reward)

### plot reservoir storage, target, inflow, release, & reward time series
plot_discrete(reservoir, storages_cont, targets_cont, storages_discrete, targets_discrete, inflows, releases, rewards, 'figs/random_discrete.png')

reservoir.close()
