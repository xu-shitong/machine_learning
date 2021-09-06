# This program is about implementing reinforce learningbased on neural net policy, 
# This program does not involve training on real data, 
# written in sudo code

import numpy as np

epoch_num = 3

# play for one step, based on given network to decide next move
# This example is based on a binary choice, network output the probability of one action
def play_one_step(net, state, ):
  left_prob = net(state)
  action = 1 if random > left_prob else 0
  new_state, reward = play(state, action)
  grad = binary_cross_entropy(action, left_prob).grad
  return new_state, grad, reward

# calculate the discounted reward of a list of reward from one single trial
def discount_reward(rewards, discount_ratio):
  length = len(rewards)
  for i in range(length - 2, -1, -1):
    rewards[i] += rewards[i+1] * discount_ratio
  return rewards

# calulate discounted reward of multiple trials, and normalize result based on global mean and std
def discount_normalize_reward(all_reward, discount_ratio):
  discounted = [ discount_reward(rewards, discount_ratio) for rewards in all_reward]

  flatten_reward = np.concatenate(discounted)
  mean = np.mean(flatten_reward)
  std = np.std(flatten_reward)
  return (discounted - mean) / std

# play numtiple steps, record gradients and discount-normalize rewards
def play_multi_step(net, trial_num, state, discount_ratio):
  all_grads = []
  all_rewards = []
  for i in range(trial_num):
    one_trail_grads = []
    one_trail_rewards = []
    while playing:
      state, grad, reward = play_one_step(net, state)
      one_trail_grads.append(grad)
      one_trail_rewards.append(reward) # discounted reward?
    all_grads.append(one_trail_grads)
    all_rewards.append(one_trail_rewards)

  
  return all_grads, discount_normalize_reward(all_rewards, discount_ratio)

# training
for i in range(epoch_num):
  all_grad, all_reward = play_multi_step(net, 10, state, discount)
  
  for i in net.parameters():
    # get weighted mean of gradients, weight is discounted-normalizes weight of each trial step
    grad = np.mean([all_grad[trial_index][step_index][i] * reward
            for trial_index, rewards in all_reward
              for step_index, reward in rewards], axis=1)
    # apply gradient
    net.apply(grad)
