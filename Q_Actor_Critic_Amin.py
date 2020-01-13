#!/usr/bin/env python
# coding: utf-8

# In[176]:


import numpy as np
import matplotlib.pyplot as plt
import gym
env = gym.make('Taxi-v1')
training_episode = 10000
plot_points = 100
episodes_per_point = training_episode//plot_points
#benchmark_data = np.zeros((1, plot_points))


# In[177]:



class Agent:
    def __init__(self, env , num_steps , gamma , alpha , beta , policy, VFA ):
        self.env = env
        self.num_steps = num_steps
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.nS = env.observation_space.n
        self.nA = env.action_space.n
        self.dim = self.nS , self.nA
        self.policy = policy
        self.VFA = VFA
    
    def learning(self):
        self.learn = True
    
    def notlearning(self):
        self.learn = False
            
    
    def train(self, numberofepisodes):
        self.learning()
        episode_rewards = 0
        state = self.env.reset()
        action = self.policy.take_action(FunctionApproximation , state)
        for i in range(numberofepisodes):
            state , action , reward, done = self.take_step(state , action)
            episode_rewards += reward
            if done: break
        return episode_rewards/numberofepisodes
    

    def benchmark(self ,numberofepisodes):   
        self.notlearning()
        episode_rewards = 0
        state = self.env.reset()
        action = self.policy.take_action(FunctionApproximation , state)
        for i in range(numberofepisodes):
            state , action , reward, done = self.take_step(state , action)
            episode_rewards += reward
            if done: break
        return episode_rewards/numberofepisodes
        
            
    def take_step(self, state , action):
        for j in range(num_steps):
            next_state, reward, done, info = self.env.step(action)
            next_action = self.policy.take_action(FunctionApproximation , next_state)
            if self.learn == True:
                features = self.VFA.feature_state_action(state , action)
                next_features = self.VFA.feature_state_action(next_state , next_action)

                value = self.VFA.get_value(features)
                next_value = self.VFA.get_value(next_features)
                delta = reward + self.gamma*next_value - value

                gradient = self.policy.take_gradient(state , action)
                delta_theta = self.alpha * gradient *value
                self.policy.update_weights(delta_theta)

                delta_weight = self.beta* delta *features
                self.VFA.update_weights(delta_weight)
            return next_state , next_action , reward , done
                  
            
class SoftmaxPolicy:
    def __init__(self, tau = 1):
        self.tau = tau
        self.nS = env.observation_space.n
        self.nA = env.action_space.n
        value = 1
        self.we = np.ones((self.nS,self.nA)) * value
        self.weights = self.we.reshape((self.nS*self.nA,1))
      #  self.dimensions = (self.nS, self.nA)

        
    def update_weights(self, delta_theta):
        self.weights += delta_theta

        
    def take_action(self,FunctionApproximation , state):
        values = np.zeros((self.nA,1))
        for action in range(self.nA):
            feature = VFA.feature_state_action(state,action)
            values[action] = np.dot(feature.T , self.weights)
        softmax_values = np.exp(values/self.tau - max(values))
        probability = (softmax_values / sum(softmax_values)).flatten()
        return np.random.choice(range(self.nA), p = probability)
    
    # def greedy_action():
     #   return np.argmax(probability)
        
    def take_gradient(self , state , action):
        features= VFA.feature_state_action(state , 0)
        for c in range(1 , self.nA):
            features = np.hstack([features , VFA.feature_state_action(state, c)])
        mean_feature = np.mean(features, 1).reshape(-1,1)
        gradient = (features[: , action].reshape(-1 , 1) - mean_feature)/ self.tau
        return gradient
    
class FunctionApproximation:
    def __init__(self):
        self.nS = env.observation_space.n
        self.nA = env.action_space.n
        value = 1
        self.we = np.ones((self.nS,self.nA)) * value
        self.weights = self.we.reshape((self.nS*self.nA,1))
    
   # def feature_state(self, state):
     #   feature = np.zeros((nS, 1))
      #  feature[state] = 1
      #  return feature
    
    def feature_state_action(self, state, action):
        feature = np.zeros((self.nS*self.nA, 1))
        feature[state * self.nA + action] = 1 
        return feature
    
    
    def update_weights(self , delta_weight):
        self.weights+= delta_weight
    
    def get_value(self, features):
        return np.dot(features.T, self.weights)
    
  #  def critic_gradient():
   #     return features
    
    
    
        
        
        
    
            
            
    
        
        


# In[178]:


policy = SoftmaxPolicy()
VFA = FunctionApproximation()
num_steps = 20
gamma = 1
alpha = 0.2
beta = 0.1
amin = Agent(env , num_steps , gamma , alpha , beta , policy , VFA )
benchmark_data = amin.benchmark(100)
for d in range(1, plot_points):
    t = amin.train(episodes_per_point)
    print(t)
 #   benchmark_data[2][0] = amin.benchmark(100)
    
#plt.figure(figsize=(16, 10))
#xaxis = [episodes_per_point for i in range(plot_points)]
#plt.plot(t,plot_points)
#plt.xlabel('Training episodes')
#plt.ylabel('Average reward per episode')
#plt.show()
    


# In[ ]:




