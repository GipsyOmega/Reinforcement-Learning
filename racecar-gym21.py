import gymnasium as gym
from gymnasium.utils.step_api_compatibility import step_api_compatibility
import numpy as np
#env = gym.make("LunarLander-v2", render_mode="human")
env = gym.make("CartPole-v1", render_mode="human")
# action_space:
episodes = 5

for episode in range(1, episodes+1):
    state = env.reset()
    done = False
    score = 0

    while not done:
        env.render()
        action = env.action_space.sample()
        # observation, reward, terminated, truncated, info = env.step(action)
        _, reward, done, _ = step_api_compatibility(
            env.step(action), output_truncation_bool=False)
        score += reward
        # print(score)
        if done:
            print("Break")
            break  # Exit the loop if episode is done

    print(f"Episode: {episode}, Score: {score}")


env.close()
