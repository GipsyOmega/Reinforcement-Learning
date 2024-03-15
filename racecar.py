import gymnasium as gym
from gymnasium.utils.play import play
from gymnasium.utils.play import PlayPlot
from gymnasium.utils.step_api_compatibility import step_api_compatibility
from gymnasium.utils.seeding import np_random
import numpy as np
#env = gym.make("LunarLander-v2", render_mode="human")
np_random(123)
env = gym.make("CarRacing-v2", render_mode='rgb_array_list')
# action_space:


def compute_metrics(obs_t, obs_tp, action, reward, terminated, truncated, info):
    return [reward, ]


plotter = PlayPlot(compute_metrics, horizon_timesteps=200,
                   plot_names=["Immediate Rew.", ])

play(env, keys_to_action={"w": np.array([0, 0.8, 0]),
                          "a": np.array([-1, 0, 0]),
                          "s": np.array([0, 0, 0.3]),
                          "d": np.array([1, 0, 0]),
                          "wa": np.array([-1, 0.6, 0]),
                          "dw": np.array([1, 0.6, 0])},
     noop=np.array([0, 0, 0]))

observation, info = env.reset()

episodes = 5

for episode in range(1, episodes+1):
    state, _ = env.reset()
    done = False
    score = 0

    while not done:
        env.render()
        action = env.action_space.sample()
        # observation, reward, terminated, truncated, info = env.step(action)
        _, reward, done, trunc, _ = env.step(action)
        score += reward
        # print(score)
        if done or trunc:
            print("Break")
            break  # Exit the loop if episode is done

    print(f"Episode: {episode}, Score: {score}")


env.close()
