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
    args = parser.parse_args()
    params = vars(args)
    #logdir = '/home/filippo/Projects/homework_fall2020/hw3/cs285/data/hw3_q1_doubleq_MsPacman-v0_06-05-2021_15-17-53/events.out.tfevents.1620307074.frittomisto'
    #eventfile = glob.glob(logdir)[0]
    #print(eventfile)
    max_returns=[]
    avg_returns=[]
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
        #print(len(train_steps), len(avg_return), len(max_return))
    if(len(avg_returns)>=1):
        max_returns=np.array(max_returns)
        #max_return=np.average(max_returns,0)
        avg_returns=np.array(avg_returns)
        avg_return=np.average(avg_returns,0)
    else:
        max_returns = max_return
        avg_returns = avg_return

    fig,ax = plt.subplots(figsize=(8,5))
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Return")
    #sns.lineplot(train_steps,avg_return)
    ax.plot(train_steps,max_return, label="max return")
    ax.plot(train_steps,avg_return, label='average return')
    ax.set_title(f"DQN Return")
    plt.figlegend(loc='lower center')
    ax.grid(True)
    fig.tight_layout()
    #plt.show()
    plt.savefig(f"Ms Pac-Man DQN Rewards")



