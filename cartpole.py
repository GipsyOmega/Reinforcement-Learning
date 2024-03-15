import gymnasium as gym
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

tf.compat.v1.disable_eager_execution()

env = gym.make("CartPole-v1")
states = env.observation_space.shape[0]
actions = 2  # env.action_space.shape[0]
print(states, actions)
# Tensorflow Model
model = Sequential()
model.add(Flatten(input_shape=(1, states)))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions, activation='linear'))


# Create a custom DQN loop

agent = DQNAgent(
    model=model,
    memory=SequentialMemory(limit=50000, window_length=1),
    policy=BoltzmannQPolicy(),
    nb_actions=actions,
    nb_steps_warmup=10,
    target_model_update=0.01
)

agent.compile(optimizer=Adam(), metrics=["mae"])
agent.fit(env, nb_steps=100000, visualize=False, verbose=1)

results = agent.test(env, nb_episodes=10, visualize=True)
print(results.history)
print()
print(np.mean(results.history["episode_reward"]))

'''
episodes = 10
for episode in range(1, episodes+1):
    state, _ = env.reset()
    done = False
    score = 0

    while not done:
        action = random.choice([0, 1])
        # observation, reward, terminated, truncated, info = env.step(action)
        _, reward, done, _, _ = env.step(action)
        score += reward
        env.render()

    print(f"Episode: {episode}, Score: {score}")
'''

env.close()
