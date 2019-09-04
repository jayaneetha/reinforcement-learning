import sys

import gym
# from keras.layers import Dense, Conv2D, Flatten
# from keras.models import Sequential
# from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import pylab
import tensorflow as tf
from tensorflow.python.keras.engine.sequential import Sequential
from tensorflow.python.keras.layers import Flatten, Dense
from tensorflow.python.keras.optimizers import Adam

EPISODES = 1000


# This is Policy Gradient agent for the Cartpole
# In this example, we use REINFORCE algorithm which uses monte-carlo update rule
class REINFORCEAgent:
    def __init__(self, state_size, action_size):
        # if you want to see Cartpole learning, then change to True
        self.render = False
        self.load_model = False
        # get size of state and action
        # self.state_size = state_size
        self.action_size = action_size

        # These are hyper parameters for the Policy Gradient
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.hidden1, self.hidden2 = 24, 24

        # create model for policy network
        self.model = self.build_model()

        # lists for the states, actions and rewards
        self.states, self.actions, self.rewards = [], [], []

        if self.load_model:
            self.model.load_weights("./save_model/mnist_reinforce.h5")

    # approximate policy using Neural Network
    # state is input and probability of each action is output of network
    def build_model(self):
        model = Sequential()
        # model.add(Dense(self.hidden1, input_dim=self.state_size, activation='relu', kernel_initializer='glorot_uniform'))
        # model.add(Dense(self.hidden2, activation='relu', kernel_initializer='glorot_uniform'))
        # model.add(Dense(self.action_size, activation='softmax', kernel_initializer='glorot_uniform'))
        model.add(Flatten(input_shape=(28, 28)))
        model.add(Dense(256, activation="relu"))
        # model.add(Conv2D(filters=16, kernel_size=4, padding='same', activation='relu', input_shape=(28, 28, 1)))
        #
        # model.add(Conv2D(
        #     filters=8,
        #     kernel_size=2,
        #     padding='same',
        #     activation="relu"
        # ))
        # model.add(Flatten())
        model.add(Dense(64, activation="relu"))
        model.add(Dense(self.action_size, activation="softmax"))

        model.summary()
        # Using categorical crossentropy as a loss is a trick to easily
        # implement the policy gradient. Categorical cross entropy is defined
        # H(p, q) = sum(p_i * log(q_i)). For the action taken, a, you set 
        # p_a = advantage. q_a is the output of the policy network, which is
        # the probability of taking the action a, i.e. policy(s, a). 
        # All other p_i are zero, thus we have H(p, q) = A * log(policy(s, a))
        model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=self.learning_rate))
        return model

    # using the output of policy network, pick action stochastically
    def get_action(self, state):
        policy = self.model.predict(state, batch_size=1).flatten()
        return np.random.choice(self.action_size, 1, p=policy)[0]

    # In Policy Gradient, Q function is not available.
    # Instead agent uses sample returns for evaluating policy
    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.discount_factor + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    # save <s, a ,r> of each step
    def append_sample(self, state, action, reward):
        self.states.append(state)
        self.rewards.append(reward)
        self.actions.append(action)

    # update policy network every episode
    def train_model(self):
        episode_length = len(self.states)

        discounted_rewards = self.discount_rewards(self.rewards)
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)

        update_inputs = np.zeros((episode_length, 28, 28))
        advantages = np.zeros((episode_length, self.action_size))

        for i in range(episode_length):
            update_inputs[i] = self.states[i]
            advantages[i][self.actions[i]] = discounted_rewards[i]

        self.model.fit(update_inputs, advantages, epochs=1, verbose=0)
        self.states, self.actions, self.rewards = [], [], []


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
        plt.figure()
        plt.imshow(self.X[self.itr])
        plt.title("Ground Truth: {}".format(self.Y[self.itr]))
        plt.show()

    def reset(self):
        self.itr = 0
        return self.X[self.itr]


if __name__ == "__main__":
    # In case of CartPole-v1, you can play until 500 time step
    env = MNISTEnv()
    # get size of state and action from environment
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # make REINFORCE agent
    agent = REINFORCEAgent(state_size, action_size)

    scores, episodes = [], []

    for e in range(EPISODES):
        done = False
        score = 0
        state = env.reset()
        state = np.reshape(state, [1, 28, 28])
        # state = state.flatten()

        while not done:
            if agent.render:
                env.render()

            # get action for the current state and go one step in environment
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            ground_truth = info['ground_truth']

            print("{}\tGround Truth: {} \tPredicted: {}\tReward: {}".format(info['itr'], ground_truth, action, reward))

            next_state = np.reshape(next_state, [1, 28, 28])
            # next_state = next_state.flatten()
            # reward = reward if not done or score == 499 else -100

            # save the sample <s, a, r> to the memory
            agent.append_sample(next_state, action, reward)

            score += reward
            state = next_state

            if done:
                # every episode, agent learns from sample returns
                agent.train_model()

                # every episode, plot the play time
                # score = score if score == 500 else score + 100
                scores.append(score)
                episodes.append(e)
                pylab.plot(episodes, scores, 'b')
                pylab.savefig("./save_graph/mnist_reinforce.png")
                print("episode:", e, "  score:", score)

                # if the mean of scores of last 10 episode is bigger than 490
                # stop training
                if np.mean(scores[-min(10, len(scores)):]) > 490:
                    sys.exit()

        # save the model
        if e % 50 == 0:
            agent.model.save_weights("./save_model/mnist_reinforce.h5")
