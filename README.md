# OpenAI LunarLander-v2 with Reinforcement Learning 

## Summary

In the LunarLander-v2 environment, the agent’s goal is to land inside the landing zone without crashing (soft landing) by taking as few actions as possible (fuel-efficient). Although the action space is discrete, the state space is continuous as it describes the lander’s coordinates, velocities, etc. Therefore, Deep Q-Learning is a good starting point within the Q-Learning algorithm family. In addition, we also implemented the Double Deep Q-Learning algorithm to the LunarLander-v2 environment. Both implementations use the built-in numerical state values as inputs. Our objective is to check if there exists noticeable difference in the performance of the two models. Note that based on the action-reward menu, a total score of 200 per game is typically considered as the success threshold.We will adopt this criterion when comparing the performances of the models.

![alt text](https://github.com/johnyangyue/reinforcement-learning-lunar-lander/blob/1d27773feaf81259b2de541e645bcb30e0bc5aae/figs/lunar_dqn.png)

![alt text](https://github.com/johnyangyue/reinforcement-learning-lunar-lander/blob/1d27773feaf81259b2de541e645bcb30e0bc5aae/figs/lunar_ddqn.png)

Training results over 2,000 episodes are shown above. With the same hyperparameters, the Deep Q and the Double Deep Q algorithm yield similar performances with the latter being slightly better if we use the number of training episodes needed to cross the 200 average score threshold as the measure. Note that we can reduce the number of training episodes required for both models if we accelerate the epsilon decay, that is, reduce the amount of time the agents spend performing exploration. Also, without additional exercises like exploring the performances under different sets of hyperparameters, we cannot conclude that Double Deep Q is the superior algorithm for LunarLander-v2 when compared with Deep Q. The two graphs above simply confirm that our agents are indeed learning. Notice that there exists a curious dip in scores for the Deep Q model in the first graph. We hypothesize that it is due to consecutive back luck in either the action- selection policy or replay memory sampling.

![alt text](https://github.com/johnyangyue/reinforcement-learning-lunar-lander/blob/1d27773feaf81259b2de541e645bcb30e0bc5aae/figs/lunar_ddqn_image.png)

As shown above, our current image-based model is unsuccessful. Using the static one-frame screenshots as states, the agent fails to achieve meaningful learning over 10,000 episodes.


