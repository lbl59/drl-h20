### A reservoir control environment to use with Gymnasium RL
from gymnasium import Env
from gymnasium.spaces import Dict, Discrete, Box
import numpy as np
from random import gauss, uniform
from math import pi, sin
import matplotlib.pyplot as plt


class Reservoir_base(Env):
    '''
    Basic reservoir class with shared functionality across continuous & discrete reservoir classes.
    Note this environment shouldnt be used directly; use Reservoir_continuous or Reservoir_discrete versions,
    which inherit from this base class.
    '''

    def __init__(self):

        ### inherit from generic Env class for environments in Gym
        super(Reservoir_base, self).__init__()

        ### general parameters
        self.ny = 6
        self.dt = 1/365
        self.time_steps = self.ny / self.dt

        ### inflow parameters
        self.inflow_base = 1
        self.use_inflow_sin = False
        self.inflow_sin_amp = 0.02
        self.inflow_sin2_amp = 0.01
        self.use_inflow_rand_walk = False
        self.inflow_rand_walk_sd = 0.0005
        self.use_inflow_jump = False
        self.inflow_jump_prob = 0.01
        self.inflow_jump_amp = 0.02

        ### reservoir parameters
        self.max_release = 3
        self.min_release = 0
        self.max_storage = 10
        self.min_storage = 0
        self.target_base = 5
        self.use_target_sin = True
        self.target_sin_amp = 2


    def reset_basic(self):
        '''
        function for resetting basic state variables each time the environment is reset
        :return: None
        '''
        self.inflow = 0.
        self.inflow_rand_walk = 0.
        self.storage = uniform(self.min_storage, self.max_storage)
        self.target = self.target_base
        self.reward = 0.
        self.t = 0
        self.terminate = False


    def update_inflow(self):
        '''
        Method for updating the inflow of the reservoir each time step (self.t). Depending on parameters,
        this can be (a) constant inflow, (b) combination of seasonally varying sinusoids,
        (c) b plus noise in the form of a Gaussian random walk, or (d) c plus a random jump process.
        :return: None
        '''

        inflow = self.inflow_base
        if self.use_inflow_sin:
            inflow += self.inflow_sin_amp * sin(self.t * 2 * pi) + \
                      self.inflow_sin2_amp * sin(self.t * pi)
        if self.use_inflow_rand_walk:
            self.inflow_rand_walk += gauss(0, self.inflow_rand_walk_sd)
            if self.use_inflow_jump:
                if uniform(0, 1) < self.inflow_jump_prob:
                    if uniform(0, 1) < 0.5:
                        self.inflow_rand_walk += self.inflow_jump_amp
                    else:
                        self.inflow_rand_walk -= self.inflow_jump_amp
            inflow += self.inflow_rand_walk
        self.inflow = max(inflow, 0)


    def update_target(self):
        '''
        Function for updating target if use_target_sin option turned on. Otherwise it just stays the same, don't do anything.
        Assume seasonal target is sin with opposite phase of inflows.
        :return: None
        '''

        if self.use_target_sin:
            self.target = self.target_base - self.target_sin_amp * sin(self.t * 2 * pi)


    def update_storage_release(self, release_target):
        '''
        Function for updating storage and release, based on target release, to preserve mass balance
        :param release_target: Target release proposed by RL agent.
        :return: None
        '''

        self.storage += (self.inflow - release_target)
        if self.storage < self.min_storage:
            self.release = release_target + self.storage
            self.storage = self.min_storage
        elif self.storage > self.max_storage:
            self.release = release_target + (self.storage - self.max_storage)
            self.storage = self.max_storage
        else:
            self.release = release_target


    def update_reward(self, storage, target):
        '''
        Function to calculate reward for RL agent. Reward diminishes based on abs value deviation from target.
        :param storage: Current storage at end of time step (either continuous or discrete)
        :param target: Current storage target (either continuous or discrete)
        :return: None
        '''

        self.reward = 10. - abs(storage - target)


    ### update time & termination status
    def update_time(self):
        '''
        Function to update the time step, & terminate simulation after ny years.
        :return: None
        '''

        self.terminate = self.t > self.ny - 0.00001
        self.t += self.dt




class Reservoir_continuous(Reservoir_base):
    '''
    Continuous-state/continuous-action version of reservoir environment.
    Inherits from Reservoir_base class, and adds specific functionality for continuous version.
    '''

    def __init__(self):

        ### inherit from Reservoir_base class for basic shared functionality
        super(Reservoir_continuous, self).__init__()

        ### define continuous 2D observation space: storage & target
        self.observation_shape = 2
        self.observation_space = Dict({'storage': Box(low=self.min_storage, high=self.max_storage, shape=(1,)),
                                       'target': Box(low=self.min_storage, high=self.max_storage, shape=(1,))})

        ### define continuous 1D action space: target reservoir release
        self.action_space = Box(low=self.min_release, high=self.max_release, shape=(1,))


    ### reset the environment to initial state each time the environment is reset
    def reset(self):
        '''
        function for resetting state variables each time the environment is reset
        :return: dictionary with the 2 observed states: storage & target.
        '''

        self.reset_basic()
        return {'storage': self.storage, 'target': self.target}


    def step(self, release_target):
        '''
        Method for updating the reservoir each time step based on prescribed release target.
        This involves updating the current inflow and seasonally-varying target,
        potentially modifying the target release to ensure mass balance, updating the storage,
        calculating the reward based on deviation from target,
        updating the time step, and terminating the simulation after a preset number of time steps.
        :release_target: The target release proposed by RL agent. generally stored in a single element list.
        :return: {'storage': self.storage, 'target': self.target}: dict of observed state that is returned to RL agent;
                 reward: decreases based on abs deviation of storage from target;
                 terminate: whether episode ended due to time horizon;
                 {'inflow':self.inflow, 'release':self.release}: additional state info not returned to agent
        '''

        ### assert that proposed action is valid
        assert self.action_space.contains(release_target), "Invalid Action"

        release_target = release_target[0]

        ### update inflow
        self.update_inflow()

        ### update release target
        self.update_target()

        ### update storage & release to preserve mass balance
        self.update_storage_release(release_target)

        ### provide negative reward (ie penalty) based on abs value deviation from target.
        self.update_reward(self.storage, self.target)

        ### update time and terminate episode when time is over
        self.update_time()

        return {'storage': self.storage, 'target': self.target}, self.reward, self.terminate, {'inflow':self.inflow, 'release':self.release}





class Reservoir_discrete(Reservoir_base):
    '''
    Discrete-state/discrete-action version of reservoir environment.
    Inherits from Reservoir_base class, and adds specific functionality for discrete version.
    '''

    def __init__(self):

        ### inherit from Reservoir_base class for basic shared functionality
        super(Reservoir_discrete, self).__init__()

        ### define discrete 2D observation space: storage & target
        self.nstep_storage = 15
        self.grid_storage = np.arange(self.min_storage, self.max_storage + 1e-9,
                                      (self.max_storage - self.min_storage) / (self.nstep_storage - 1))
        self.observation_space = Dict({'storage_discrete': Discrete(self.nstep_storage),
                                       'target_discrete': Discrete(self.nstep_storage)})

        ### define continuous 1D action space: target reservoir release
        self.nstep_release = 15
        self.grid_release = np.arange(self.min_release, self.max_release + 1e-9,
                                      (self.max_release - self.min_release) / (self.nstep_release - 1))
        self.action_space = Discrete(self.nstep_release)


    def reset(self):
        '''
        function for resetting state variables each time the environment is reset
        :return: dictionary with the 2 observed states: storage & target.
        '''

        self.reset_basic()

        ### get discrete indexes for storage and target to return to agent
        storage_discrete = np.argmin(np.abs(self.grid_storage - self.storage))
        target_discrete = np.argmin(np.abs(self.grid_storage - self.target))

        return {'storage_discrete':storage_discrete, 'target_discrete':target_discrete}


    def step(self, release_discrete):
        '''
        Method for updating the reservoir each time step based on prescribed release target.
        This involves updating the current inflow and seasonally-varying target,
        potentially modifying the target release to ensure mass balance, updating the storage,
        calculating the reward based on deviation from target,
        updating the time step, and terminating the simulation after a preset number of time steps.
        :release_target: The index of discrete release proposed by RL agent.
        :return: {'storage_discrete':storage_discrete, 'target_discrete':target_discrete}: dict of observed state that is returned to RL agent;
                 reward: decreases based on abs deviation of discretized storage from discretized target;
                 terminate: whether episode ended due to time horizon;
                 {'inflow': self.inflow, 'release': self.release,'storage_cont': self.storage, 'target_cont':self.target}:
                        additional state info not returned to agent
        '''

        ### assert that proposed action is valid
        assert self.action_space.contains(release_discrete), f"Invalid Action: {release_discrete}"

        ### map discrete action index to release_target
        release_target = self.grid_release[release_discrete]

        ### update inflow
        self.update_inflow()

        ### update release target
        self.update_target()

        ### update storage & release to preserve mass balance
        self.update_storage_release(release_target)

        ## get discrete indexes for storage and target & last release
        storage_discrete = np.argmin(np.abs(self.grid_storage - self.storage))
        target_discrete = np.argmin(np.abs(self.grid_storage - self.target))

        ### provide negative reward (ie penalty) based on abs value deviation from target.
        self.update_reward(storage_discrete, target_discrete)

        ### update time and truncate episode when time is over
        self.update_time()

        return {'storage_discrete':storage_discrete, 'target_discrete':target_discrete}, self.reward, self.terminate, \
                {'inflow': self.inflow, 'release': self.release,'storage_cont': self.storage, 'target_cont':self.target}





def plot_continuous(reservoir, storages, targets, inflows, releases, rewards):
    ny = reservoir.ny
    dt = reservoir.dt
    ### plot results
    fig, axs = plt.subplots(3, 1, figsize=(11, 4), gridspec_kw={'hspace': 0.05})
    t_flows = np.arange(0, ny + 0.0001, dt)
    t_storages = np.arange(-dt, ny + 0.0001, dt)
    ### inflow vs release
    axs[0].plot(t_flows, inflows, label='inflow')
    axs[0].plot(t_flows, releases, label='release')
    axs[0].set_xticks(np.arange(ny + 1), [''] * (ny + 1))
    axs[0].legend()
    axs[0].set_ylabel('Flows')
    ### target vs storage
    axs[1].plot(t_storages, targets, label='target')
    axs[1].plot(t_storages, storages, label='storage')
    axs[1].set_xlabel('years')
    axs[1].set_xticks(np.arange(ny + 1), np.arange(ny + 1))
    axs[1].legend()
    axs[1].set_ylabel('Storages')
    ### reward
    axs[2].plot(t_flows, rewards, label='rewards')
    axs[2].set_xticks(np.arange(ny + 1), [''] * (ny + 1))
    axs[2].legend()
    axs[2].set_ylabel('Rewards')
    plt.show()



def plot_discrete(reservoir, storages_cont, targets_cont, storages_discrete, targets_discrete, inflows, releases, rewards):
    ny = reservoir.ny
    dt = reservoir.dt
    ### plot results
    fig,axs = plt.subplots(4,1,figsize=(11,5),gridspec_kw={'hspace':0.05})
    t_flows = np.arange(0, ny + 0.001, dt)
    t_storages = np.arange(-dt, ny + 0.001, dt)
    ### inflow vs release
    axs[0].plot(t_flows, inflows, label='inflow')
    axs[0].plot(t_flows, releases, label='release')
    axs[0].set_xticks(np.arange(ny+1), ['']*(ny+1))
    axs[0].legend()
    axs[0].set_ylabel('Flows')
    ### target vs storage - continuous
    axs[1].plot(t_flows, targets_cont, label='target')
    axs[1].plot(t_flows, storages_cont, label='storage')
    axs[1].set_xlabel('years')
    axs[1].set_xticks(np.arange(ny+1), np.arange(ny+1))
    axs[1].legend()
    axs[1].set_ylabel('Storages\n(continuous)')
    ### target vs storage - discrete
    axs[2].plot(t_storages, targets_discrete, label='target obs')
    axs[2].plot(t_storages, storages_discrete, label='storage obs')
    axs[2].set_xlabel('years')
    axs[2].set_xticks(np.arange(ny+1), np.arange(ny+1))
    axs[2].legend()
    axs[2].set_ylabel('Storages\n(discrete)')
    ### reward
    axs[3].plot(t_flows, rewards, label='rewards')
    axs[3].set_xticks(np.arange(ny+1), ['']*(ny+1))
    axs[3].legend()
    axs[3].set_ylabel('Rewards')
    plt.show()
