from collections import deque
import os

import numpy as np
import tensorflow as tf


class DQNAgent:
    """
    Multi Layer Perceptron with Experience Replay
    """

    def __init__(self, enable_actions, environment_name, epsilon=1e-1):
        # parameters
        self.name = os.path.splitext(os.path.basename(__file__))[0]
        self.environment_name = environment_name
        self.enable_actions = enable_actions
        self.n_actions = len(self.enable_actions)
        self.minibatch_size = 32
        self.replay_memory_size = 10000
        self.learning_rate = 0.001
        self.discount_factor = 0.99
        self.exploration = 0.1
        self.model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        self.model_name = "{}.ckpt".format(self.environment_name)
        self.epsilon = epsilon

        # replay memory
        self.D = deque(maxlen=self.replay_memory_size)

        # model
        self.init_model()

        # variables
        self.current_loss = 0.0
        self.tmp_q_values = None

    def init_model(self):
        # input layer (20 x 10 = 200)
        self.x = tf.placeholder(tf.float32, [None, 200])

        # fully connected layer (200)
        W_fc1 = tf.Variable(tf.truncated_normal([200, 400], stddev=0.01))
        b_fc1 = tf.Variable(tf.zeros([400]))
        h_fc1 = tf.nn.relu(tf.matmul(self.x, W_fc1) + b_fc1)

        # output layer (n_actions)
        W_fc2 = tf.Variable(tf.truncated_normal([400, 100], stddev=0.01))
        b_fc2 = tf.Variable(tf.zeros([100]))
        h_fc2 = tf.matmul(h_fc1, W_fc2) + b_fc2

        W_out = tf.Variable(tf.truncated_normal([100, self.n_actions], stddev=0.01))
        b_out = tf.Variable(tf.zeros([self.n_actions]))
        self.y = tf.matmul(h_fc2, W_out) + b_out

        # loss function
        # 教師信号？
        self.a = tf.placeholder(tf.int64, [None])
        self.y_ = tf.placeholder(tf.float32, [None])

        a_one_hot = tf.one_hot(self.a, self.n_actions, 1.0, 0.0)  # 行動をone hot vectorに変換する
        m = tf.multiply(self.y, a_one_hot)
        q_value = tf.reduce_sum(m, reduction_indices=1)  # 行動のQ値の計算

        # エラークリップ
        error = tf.abs(self.y_ - q_value)
        quadratic_part = tf.clip_by_value(error, 0.0, 1.0)
        linear_part = error - quadratic_part
        self.loss = tf.reduce_mean(0.5 * tf.square(quadratic_part) + linear_part)  # 誤差関数

        # train operation
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.training = optimizer.minimize(self.loss)

        # saver
        self.saver = tf.train.Saver()

        # session
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def Q_values(self, state):
        # Q(state, action) of all actions
        return self.sess.run(self.y, feed_dict={self.x: [state]})[0]

    def select_action(self, state):
        self.tmp_q_values = self.Q_values(state)
        if np.random.rand() <= self.epsilon:
            # random
            return np.random.choice(self.enable_actions)
        else:
            # max_action Q(state, action)
            return self.enable_actions[np.argmax(self.tmp_q_values)]

    def store_experience(self, state, action, reward, state_1, terminal):
        self.D.append((state, action, reward, state_1, terminal))

    def experience_replay(self):
        state_minibatch = []
        y_minibatch = []

        # sample random minibatch
        minibatch_size = min(len(self.D), self.minibatch_size)
        minibatch_indexes = np.random.randint(0, len(self.D), minibatch_size)
        action_batch = []
        y_batch = []

        for j in minibatch_indexes:
            state_j, action_j, reward_j, state_j_1, terminal = self.D[j]
            action_j_index = self.enable_actions.index(action_j)
            action_batch.append(action_j_index)

            y_j = self.Q_values(state_j)

            if terminal:
                y_j[action_j_index] = reward_j
            else:
                # reward_j + gamma * max_action' Q(state', action')
                y_j[action_j_index] = reward_j + self.discount_factor * np.max(self.Q_values(state_j_1))  # NOQA

            state_minibatch.append(state_j)
            y_minibatch.append(y_j)
            y_batch.append(y_j[action_j_index])

        # training
        self.sess.run(self.training, feed_dict={self.x: state_minibatch,
                                                self.y_: y_batch,
                                                self.a: action_batch})

        # for log
        self.current_loss = self.sess.run(self.loss, feed_dict={self.x: state_minibatch,
                                                                self.y_: y_batch,
                                                                self.a: action_batch})

    def load_model(self, model_path=None):
        if model_path:
            # load from model_path
            self.saver.restore(self.sess, model_path)
        else:
            # load from checkpoint
            checkpoint = tf.train.get_checkpoint_state(self.model_dir)
            if checkpoint and checkpoint.model_checkpoint_path:
                self.saver.restore(self.sess, checkpoint.model_checkpoint_path)

    def save_model(self):
        self.saver.save(self.sess, os.path.join(self.model_dir, self.model_name))
