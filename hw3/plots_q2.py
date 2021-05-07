import matplotlib.pyplot as plt
import numpy as np
import glob
import seaborn as sns 
from tensorflow.python.summary.summary_iterator import summary_iterator
import argparse
import random
from collections import namedtuple

import gym
import numpy as np
from torch import nn
import torch.optim as optim

from cs285.infrastructure.atari_wrappers import wrap_deepmind
from gym.envs.registration import register

import torch
#Plot Results


def linear_interpolation(l, r, alpha):
    return l + alpha * (r - l)

def lander_exploration_schedule(num_timesteps):
    return PiecewiseSchedule(
        [
            (0, 1),
            (num_timesteps * 0.1, 0.02),
        ], outside_value=0.02
    )

class PiecewiseSchedule(object):
    def __init__(self, endpoints, interpolation=linear_interpolation, outside_value=None):
        """Piecewise schedule.
        endpoints: [(int, int)]
            list of pairs `(time, value)` meanining that schedule should output
            `value` when `t==time`. All the values for time must be sorted in
            an increasing order. When t is between two times, e.g. `(time_a, value_a)`
            and `(time_b, value_b)`, such that `time_a <= t < time_b` then value outputs
            `interpolation(value_a, value_b, alpha)` where alpha is a fraction of
            time passed between `time_a` and `time_b` for time `t`.
        interpolation: lambda float, float, float: float
            a function that takes value to the left and to the right of t according
            to the `endpoints`. Alpha is the fraction of distance from left endpoint to
            right endpoint that t has covered. See linear_interpolation for example.
        outside_value: float
            if the value is requested outside of all the intervals sepecified in
            `endpoints` this value is returned. If None then AssertionError is
            raised when outside value is requested.
        """
        idxes = [e[0] for e in endpoints]
        assert idxes == sorted(idxes)
        self._interpolation = interpolation
        self._outside_value = outside_value
        self._endpoints      = endpoints

    def value(self, t):
        """See Schedule.value"""
        for (l_t, l), (r_t, r) in zip(self._endpoints[:-1], self._endpoints[1:]):
            if l_t <= t and t < r_t:
                alpha = float(t - l_t) / (r_t - l_t)
                return self._interpolation(l, r, alpha)

        # t does not belong to any of the pieces, so doom.
        assert self._outside_value is not None
        return self._outside_value


def get_section_results(file):
    """
        requires tensorflow==1.12.0
    """
    train_steps = []
    avg_return = []
    max_return = []
    for e in summary_iterator(file):

        for v in e.summary.value:
            if v.tag == 'Train_EnvstepsSoFar':
                train_steps.append(v.simple_value)
            elif v.tag == 'Train_AverageReturn':
                avg_return.append(v.simple_value)
            elif v.tag == 'Train_BestReturn':
                max_return.append(v.simple_value)
    return train_steps, avg_return, max_return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logs_dir',default=[], nargs='*')
    #parser.add_argument('--double_logs_dir',default=[], nargs='*')
    args = parser.parse_args()
    params = vars(args)
    #logdir = '/home/filippo/Projects/homework_fall2020/hw3/cs285/data/hw3_q1_doubleq_MsPacman-v0_06-05-2021_15-17-53/events.out.tfevents.1620307074.frittomisto'
    #eventfile = glob.glob(logdir)[0]
    #print(eventfile)
    max_returns=[]
    avg_returns=[]
    
    fig,ax = plt.subplots(figsize=(8,5))
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Return")

    for i,logdir in enumerate(params['logs_dir']):
        train_steps, avg_return, max_return = get_section_results(logdir)
        ##train_steps= train_steps[1:]
        #print(train_steps)
        #train_steps[0]=0
        max_return.insert(0,0)
        max_return.insert(0,0)
        
        avg_return.insert(0,0)
        max_returns.append(max_return)
        avg_returns.append(avg_return)
        if(i <=2):
            ax.plot(train_steps,avg_return,color='blue', alpha=0.1)
        else:
            ax.plot(train_steps,avg_return,color='orange', alpha=0.2)

        if ( i == 2):
            avg_returns=np.array(avg_returns)
            avg_return=np.average(avg_returns,0)
            ax.plot(train_steps,avg_return, label='DQN average return', color ='blue')
            avg_returns=[]
        if (i==5):
            avg_returns=np.array(avg_returns)
            avg_return=np.average(avg_returns,0)
            ax.plot(train_steps,avg_return, label='Double DQN average return', color ='orange')
        #print(len(train_steps), len(avg_return), len(max_return))
    '''
    if(len(avg_returns)>=1):
        max_returns=np.array(max_returns)
        #max_return=np.average(max_returns,0)
        avg_returns=np.array(avg_returns)
        avg_return=np.average(avg_returns,0)
'''
    
   
    lander_exploration_schedule=lander_exploration_schedule(500000)
    eps_list = []
    for el in train_steps:
        eps=lander_exploration_schedule.value(el)
        #print(eps)
        eps_list.append(eps)

    #sns.lineplot(train_steps,avg_return)
    #ax.plot(train_steps,max_return, label="max return")
    
    
    ax.set_title(f"DQN vs Double DQN Returns")
    #ax.set_xlim([100,1e6])
    ax2 = ax.twinx()
    ax2.set_ylabel("Epsilon")
    ax2.plot(train_steps,eps_list, label='Episilon Decay behaviour', alpha=0.35, color='r')
    plt.figlegend(loc='lower center')
    ax.grid(True)
    fig.tight_layout()
    #plt.show()
    plt.savefig(f"Lunar Lander DQN vs Double DQN Returns")



