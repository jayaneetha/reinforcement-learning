import gym
import datetime

import gym
# from keras.layers import Dense, Conv2D, Flatten
# from keras.models import Sequential
# from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import pylab
import tensorflow as tf
from tensorflow.python.keras.engine.sequential import Sequential
from tensorflow.python.keras.layers import Conv2D, Flatten, Dense
from tensorflow.python.keras.optimizers import Adam

EPISODES = 1000


# A2C(Advantage Actor-Critic) agent
class A2CAgent:
    def __init__(self, state_size, action_size):
        self.render = False
        self.load_model = False
        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size
        self.value_size = 1

        # These are hyper parameters for the Policy Gradient
        self.discount_factor = 0.99
        self.actor_lr = 0.001
        self.critic_lr = 0.005

        # create model for policy network
        self.actor = self.build_actor()
        self.critic = self.build_critic()

        if self.load_model:
            self.actor.load_weights("./save_model/mnist_actor.h5")
            self.critic.load_weights("./save_model/mnist_critic.h5")

    # approximate policy and value using Neural Network
    # actor: state is input and probability of each action is output of model
    def build_actor(self):
        actor = Sequential()

        actor.add(Conv2D(filters=16, kernel_size=4, padding='same', activation='relu', input_shape=(28, 28, 1)))
        actor.add(Conv2D(filters=8, kernel_size=2, padding='same', activation="relu"))
        actor.add(Flatten())
        actor.add(Dense(64, activation="relu"))
        actor.add(Dense(self.action_size, activation="softmax"))

        actor.summary()

        actor.compile(loss='categorical_crossentropy', optimizer=Adam(lr=self.actor_lr))
        return actor

    # critic: state is input and value of state is output of model
    def build_critic(self):
        critic = Sequential()

        critic.add(Conv2D(filters=16, kernel_size=4, padding='same', activation='relu', input_shape=(28, 28, 1)))
        critic.add(Conv2D(filters=8, kernel_size=2, padding='same', activation="relu"))
        critic.add(Flatten())
        critic.add(Dense(64, activation="relu"))
        critic.add(Dense(self.value_size, activation="linear"))

        critic.summary()
        critic.compile(loss="mse", optimizer=Adam(lr=self.critic_lr))
        return critic

    # using the output of policy network, pick action stochastically
    def get_action(self, state):
        policy = self.actor.predict(state, batch_size=1).flatten()
        return np.random.choice(self.action_size, 1, p=policy)[0]

    # update policy network every episode
    def train_model(self, state, action, reward, next_state, done):
        target = np.zeros((1, self.value_size))
        advantages = np.zeros((1, self.action_size))

        value = self.critic.predict(state)[0]
        next_value = max(self.critic.predict(next_state)[0])

        if done:
            advantages[0][action] = reward - value
            target[0][0] = reward
        else:
            advantages[0][action] = reward + self.discount_factor * (next_value) - value
            target[0][0] = reward + self.discount_factor * next_value

        actor_log_dir = "actor-logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        actor_tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=actor_log_dir, histogram_freq=1)

        critic_log_dir = "critic-logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        critic_tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=critic_log_dir, histogram_freq=1)

        # self.actor.fit(state, advantages, epochs=1,  callbacks=[actor_tensorboard_callback])
        # self.critic.fit(state, target, epochs=1, callbacks=[critic_tensorboard_callback])

        self.actor.fit(state, advantages, epochs=1, verbose=0)
        self.critic.fit(state, target, epochs=1, verbose=0)


class MNISTEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self) -> None:
        super().__init__()
        self.itr = 0

        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

        filtered_x_temp = []
        filtered_y_temp = []

        num_classes = 3

        for i in range(len(y_train)):
            if y_train[i] < num_classes:
                filtered_x_temp.append(x_train[i])
                filtered_y_temp.append(y_train[i])

        x_train = np.array(filtered_x_temp)
        y_train = np.array(filtered_y_temp)

        self.X = x_train[:100]
        self.Y = y_train[:100]

        self.action_space = gym.spaces.Discrete(num_classes)
        self.observation_space = gym.spaces.Box(0, 225, [28, 28])

    def step(self, action):
        assert self.action_space.contains(action)
        reward = -0.1 + int(action == self.Y[self.itr])

        done = (len(self.X) - 2 <= self.itr)

        next_state = self.X[self.itr + 1]
        info = {
            "ground_truth": self.Y[self.itr],
            "itr": self.itr
        }
        self.itr += 1

        return next_state, reward, done, info

    def render(self, mode='human'):
        plt.figure(2)
        plt.imshow(self.X[self.itr])
        plt.title("Ground Truth: {}".format(self.Y[self.itr]))
        # self.plt.show()

    def reset(self):
        self.itr = 0
        return self.X[self.itr]


if __name__ == "__main__":

    env = MNISTEnv()
    # get size of state and action from environment
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # make A2C agent
    agent = A2CAgent(state_size, action_size)

    scores, episodes = [], []

    for e in range(EPISODES):
        done = False
        score = 0
        state = env.reset()
        state = np.reshape(state, [1, 28, 28, 1])

        while not done:
            if agent.render:
                env.render()

            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)

            ground_truth = info['ground_truth']

            print("{}\tGround Truth: {} \tPredicted: {}\tReward: {}".format(info['itr'], ground_truth, action, reward))

            next_state = np.reshape(next_state, [1, 28, 28, 1])

            agent.train_model(state, action, reward, next_state, done)

            score += reward
            state = next_state

            if done:
                # every episode, plot the play time
                # score = score if score == 500.0 else score + 100
                scores.append(score)
                episodes.append(e)
                pylab.plot(episodes, scores, 'b')
                pylab.savefig("./save_graph/mnist_a2c.png")
                print("episode:", e, "  score:", score)

        # save the model
        if e % 50 == 0:
            agent.actor.save_weights("./save_model/mnist_actor.h5")
            agent.critic.save_weights("./save_model/mnist_critic.h5")

    plt.figure(9)
    plt.plot(episodes, scores)
    plt.show()
    print("End")
