import gym
import os
import numpy as np
import torch as T
from double_dqn_agent import DoubleDQNAgent
from utils import plot_scores,save_frames_as_gif

if __name__ == '__main__':
    if T.cuda.is_available():
        print('...Using GPU...')
    else:
        print('...Using CPU...')

    env = gym.make('LunarLander-v2')
    load_checkpoint = False
    n_games = 2000
    agent = DoubleDQNAgent(gamma=0.99,epsilon=1.0,lr=0.0001,input_dims=env.observation_space.shape,n_actions=env.action_space.n,mem_size=50000,
                           eps_min=0.1,batch_size=32,replace=1000,eps_dec=1e-5,chkpt_dir='models_store/',algo='DoubleDQNAgent',env_name='LunarLander-v2')
    
    if load_checkpoint:
        agent.load_models()
    
    fname1 = agent.algo+'_'+agent.env_name+'_lr'+str(agent.lr)+'_'+str(n_games)+'games_v1'
    fname2 = agent.algo+'_'+agent.env_name+'_lr'+str(agent.lr)+'_'+str(n_games)+'games_v2'
    figure_file1 = 'plots/'+fname1+'.png'
    figure_file2 = 'plots/'+fname2+'.png'
    
    n_steps = 0
    scores,eps_history,steps_array,avg_scores,min_scores,max_scores = [],[],[],[],[],[]

    frames_start = []
    frames_end = []
    
    for i in range(n_games):
        done = False
        score = 0
        observation = env.reset()

        while not done:
            action = agent.choose_action(observation)
            observation_,reward,done,info = env.step(action)
            score += reward
            
            if not load_checkpoint:
                agent.store_transition(observation,action,reward,observation_,int(done))
                agent.learn()
            observation = observation_
            n_steps += 1

            if i <= 20:
                frames_start.append(env.render(mode='rgb_array'))
            elif i >= n_games-20:
                frames_end.append(env.render(mode='rgb_array'))
        
        scores.append(score)
        steps_array.append(n_steps)

        avg_score = np.mean(scores[-100:])
        min_score = np.min(scores[-100:])
        max_score = np.max(scores[-100:])

        avg_scores.append(avg_score)
        min_scores.append(min_score)
        max_scores.append(max_score)
        
        print('Episode {}, Score {}, Avg Score {}, Epsilon {}, Steps {}'.format(i,score,round(avg_score,2),round(agent.epsilon,2),n_steps))                                                          

        if (i%100 == 0 and i >= 100) or (i == n_games-1):
            if not load_checkpoint:
                agent.save_models()
        
        eps_history.append(agent.epsilon)
        
    episodes_array = [i+1 for i in range(n_games)]
    plot_scores(episodes_array,scores,avg_scores,min_scores,max_scores,eps_history,figure_file1)
    #plot_scores(steps_array,scores,avg_scores,min_scores,max_scores,eps_history,figure_file2)
    save_frames_as_gif(frames_start,filename='lunarlander_animation_start.gif')
    save_frames_as_gif(frames_end,filename='lunarlander_animation_end.gif')