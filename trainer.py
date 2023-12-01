import torch
from utils import return_replay
import torch.nn.functional as F
import random
def learn(model,optimizer,episode,env):
    observation, dummy = env.reset()
    observation_replay, action_replay, reward_replay, total_reward = episode
    model.train()
    return_list = return_replay(reward_replay,0.95)
    for i in range(len(observation_replay)):
        observation_tensor = torch.tensor(observation_replay[i])

        pred = model(observation_tensor)
        print(pred)

        learn_result = pred
        learn_result[action_replay[i]]=return_list[i]
        print(learn_result)

        loss = F.l1_loss(pred, learn_result)
        print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def play(model,args,env,epsilon,epoch):
    observation , dummy = env.reset()
    cumulative_reward = 0
    done = False
    observation_replay=[]
    reward_replay=[]
    action_replay=[]
    while not done:
        with torch.no_grad():
            model.eval()
            observation_tensor=torch.tensor(observation)

            if random.random()>(epsilon*(1-epoch/args.epochs/args.epsilon_decay_epoch)):
                action = int(torch.argmax(model(observation_tensor)))
            else:
                action = int(torch.argmin(model(observation_tensor)))
                print('non-greedy chosen')
            observation_replay.append(observation)
            action_replay.append(action)

            observation, reward, done, info,dummy = env.step(action)
            cumulative_reward += reward

            reward_replay.append(reward)
    return observation_replay , action_replay, reward_replay , cumulative_reward

def train(model,optimizer,args,env):
    for epoch in range(1,args.epochs+1):
        episode = play(model,args,env,args.epsilon,epoch)
        learn(model,optimizer,episode,env)
    #test(model,env,args)

def test(model,env,args):
    print('Running Test...')
    return_list=[]
    for i in range(10):
        observation_replay , action_replay, reward_replay , cumulative_reward = play(model,args,env,0,0)
        return_list.append(cumulative_reward)
    print(return_list)