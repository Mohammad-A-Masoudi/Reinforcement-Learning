#!/usr/bin/env python
# coding: utf-8

# In[16]:


import gym
import numpy as np


# In[17]:


env = gym.make('MountainCar-v0')


# In[18]:


done = False
learning_rate = 0.1
discount = 0.95
epsilon = 0.5
decay_epsilon_start = 1
decay_epsilon_end = eps//2
epsilon_decay_v = epsilon / (decay_epsilon_end -decay_epsilon_start)
eps = 25000

show = 2000

discrete_os_size = [20] * len(env.observation_space.low)
discrete_os = (env.observation_space.high- env.observation_space.low)/ discrete_os_size
q_table = np.random.uniform(-2 , 0, (discrete_os_size + [env.action_space.n]))

def discrete_state(state):
    discrete_state = (state - env.observation_space.low)/discrete_os
    return tuple(discrete_state.astype(int))

for episode in range(eps):
    
    if episode % show == 0:
        render = True
        print(episode)
    else:
        render = False
    state_reset = discrete_state(env.reset())

    while not done:
        if np.random.random() > epsilon:
            action = np.argmax(q_table[state_reset])
        else:
            action = np.random.randint(0 ,env.action_space.n)
        new_state, reward, done, _ = env.step(action)
        new_dis_state = discrete_state(new_state)
        if render:
            env.render()
        if not done:
            max_next_q = max(q_table[new_dis_state])
            old_q = q_table[state_reset + (action,)]
            new_q = (1-learning_rate)* old_q + learning_rate * (reward+discount*max_next_q)
            q_table[state_reset+(action,)] = new_q
        elif new_state[0] >= env.goal_position:
            print(f'yes{episode}')
            q_table[state_reset+(action,)] = 0    
        state_reset = new_dis_state
        
    if decay_epsilon_end >= episode >= decay_epsilon_start:
        epsilon -= epsilon_decay_v
    env.close()


# In[ ]:




