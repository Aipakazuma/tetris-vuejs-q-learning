import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, BatchNormalization, Dropout
from keras.optimizers import Adam
from keras.callbacks import TensorBoard

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
import os

import sys
from game import Game
from time import time

import gym
import numpy as np
import gym.spaces

import sys,tty,termios


class _Getch:
    def __call__(self):
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(3)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch


def get():
    inkey = _Getch()
    selected = None
    while selected is None:
        k = inkey()
        # if k != '': break

        if k == '\x1b[A':
            selected = 2
        elif k=='\x1b[B':
            selected = 3
        elif k=='\x1b[C':
            selected = 1
        elif k=='\x1b[D':
            selected = 0
        else:
            print("not an arrow key!")

    return selected


class MyProcessor(Processor):
    def __init__(self):
        self.step = 0

    def process_action(self, action):
        if self.step <= 500:
            action = get()
            self.step += 1

        return action


class MyEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.env = Game()
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(200,)
        )
        self.reward_range = [-1, 1000]

    def reset(self):
        self.env.reset()
        observation, _, _, _ = self.step(np.random.randint(0, self.action_space.n))
        return observation

    def step(self, action):
        observation, reward, done, info = self.env.step(self.env.enable_actions[action])
        return observation, reward, done, info

    def render(self, mode='human', close=False):
        outfile = None
        return outfile

    def close(self):
        pass

    def seed(self, seed=None):
        return super().seed(seed)


if __name__ == '__main__':
    # Get the environment and extract the number of actions.
    env = MyEnv()
    np.random.seed(123)
    env.seed(123)
    nb_actions = env.action_space.n

    # Next, we build a very simple model.
    model = Sequential()
    # input
    model.add(Flatten(input_shape=(2, 200)))

    # 1 
    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    # 2
    model.add(Dense(128))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    # 3
    model.add(Dense(64))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    # output
    model.add(Dense(nb_actions))
    model.add(Activation('softmax'))
    print(model.summary())

    tb_log_dir = os.path.join('logs', str(time()))
    tb_callback = TensorBoard(log_dir=tb_log_dir)
    # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
    # even the metrics!
    memory = SequentialMemory(limit=100000, window_length=2)
    # policy = BoltzmannQPolicy()
    policy = EpsGreedyQPolicy(eps=0.3)
    dqn = DQNAgent(model=model, batch_size=128, nb_actions=nb_actions,
                   memory=memory, nb_steps_warmup=1000,
                   target_model_update=1e-2, policy=policy) #, processor=MyProcessor())
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])

    # Okay, now it's time to learn something! We visualize the training here for show, but this
    # slows down training quite a lot. You can always safely abort the training prematurely using
    # Ctrl + C.
    # dqn.load_weights('dqn_tetris_weights.h5f')
    dqn.fit(env, nb_steps=100000, visualize=False, verbose=2, callbacks=[tb_callback])

    # After training is done, we save the final weights.
    dqn.save_weights('dqn_{}_weights.h5f'.format('tetris'), overwrite=True)
    # dqn.load_weights('dqn_tetris_weights.h5f')

    # Finally, evaluate our algorithm for 5 episodes.
    # dqn.test(env, nb_episodes=5, visualize=False)
