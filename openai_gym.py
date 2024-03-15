import gymnasium as gym
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers.legacy import Adam
import keras
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
import random
import time

start = time.time()
env = gym.make("CartPole-v1")
states = env.observation_space.shape[0]
actions = env.action_space.n
# print(actions.n)
end = time.time()

# episodes = 20
# for episode in range(1, episodes+1):
#     observation, info = env.reset()
#     score = 0
#     done = False

#     while not done:
#         env.render()
#         action = random.choice([0, 1])
#         n_states, reward, done, info, _ = env.step(action)
#         score += reward

#     print(f"Episode {episode}, Score: {score}")
# #print(end - start)
# env.close()

model = None
model = Sequential()
model.add(Flatten(input_shape=(1, states)))
model.add(Dense(24, activation='relu'))
model.add(Dense(24, activation='relu'))
model.add(Dense(actions, activation='linear'))

model.summary()


def build_agent(model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy,
                   nb_actions=actions, nb_steps_warmup=10, target_model_update=1e-2)
    return dqn


dqn = build_agent(model, actions)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])
dqn.fit(env, nb_steps=50000, visualize=False, verbose=1)
#model.compile(loss = 'binary_crossentropy', optimizer = Adam, metrics = ['acc'])
# Deep learning Model
