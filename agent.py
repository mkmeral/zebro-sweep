from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt

import tensorflow as tf

from environment.environment import ZebroEnvironment
from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments import tf_py_environment
from tf_agents.networks import q_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

tf.compat.v1.enable_v2_behavior()


class ZebroAgent:

    def __init__(self,
                 train_environment,
                 eval_environment,
                 replay_buffer_capacity=1000,
                 fc_layer_params=(100,),
                 learning_rate=1e-3):
        # Use TF Environment Wrappers to translate them for TF
        self.train_env = tf_py_environment.TFPyEnvironment(train_environment)
        self.eval_env = tf_py_environment.TFPyEnvironment(eval_environment)

        # Define Q-Network
        q_net = q_network.QNetwork(
            self.train_env.observation_spec(),
            self.train_env.action_spec(),
            fc_layer_params=fc_layer_params)

        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

        train_step_counter = tf.compat.v2.Variable(0)
        # Define Agent
        self.agent = dqn_agent.DqnAgent(
            self.train_env.time_step_spec(),
            self.train_env.action_spec(),
            q_network=q_net,
            optimizer=optimizer,
            td_errors_loss_fn=dqn_agent.element_wise_squared_loss,
            train_step_counter=train_step_counter)

        self.agent.initialize()

        self.replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=self.agent.collect_data_spec,
            batch_size=train_env.batch_size,
            max_length=replay_buffer_capacity)

        self.eval_policy = self.agent.policy
        self.collect_policy = self.agent.collect_policy

        self.random_policy = random_tf_policy.RandomTFPolicy(self.train_env.time_step_spec(),
                                                             self.train_env.action_spec())

    def train(self,
              num_iterations=1000,
              initial_collect_steps=10,
              collect_steps_per_iteration=1,
              batch_size=64,
              log_interval=200,
              num_eval_episodes=5,
              eval_interval=1000
              ):
        # Collect initial steps
        for _ in range(initial_collect_steps):
            self.collect_step(self.train_env, self.random_policy)

        # Dataset generates trajectories with shape [Bx2x...]
        dataset = self.replay_buffer.as_dataset(
            num_parallel_calls=3, sample_batch_size=batch_size, num_steps=2).prefetch(3)

        iterator = iter(dataset)

        # (Optional) Optimize by wrapping some of the code in a graph using TF function.
        self.agent.train = common.function(self.agent.train)

        # Reset the train step
        self.agent.train_step_counter.assign(0)

        # Evaluate the agent's policy once before training.
        avg_return = self.compute_avg_return(self.eval_env, self.agent.policy, self.agent)
        returns = [avg_return]

        for _ in range(num_iterations):

            # Collect a few steps using collect_policy and save to the replay buffer.
            for _ in range(collect_steps_per_iteration):
                self.collect_step(self.train_env, self.agent.collect_policy)

            # Sample a batch of data from the buffer and update the agent's network.
            experience, unused_info = next(iterator)
            train_loss = self.agent.train(experience)

            step = self.agent.train_step_counter.numpy()

            if step % log_interval == 0:
                print('step = {0}: loss = {1}'.format(step, train_loss.loss))

            if step % eval_interval == 0:
                avg_return = self.compute_avg_return(self.eval_env, self.agent.policy, num_eval_episodes)
                print('step = {0}: Average Return = {1}'.format(step, avg_return))
                returns.append(avg_return)

    def compute_avg_return(self, environment, policy, num_episodes=1):

        total_return = 0.0
        for _ in range(num_episodes):

            time_step = environment.reset()
            episode_return = 0.0

            while not time_step.is_last():
                action_step = policy.action(time_step)
                time_step = environment.step(action_step.action)
                episode_return += time_step.reward
            total_return += episode_return

        avg_return = total_return / num_episodes
        return avg_return.numpy()[0]

    def collect_step(self, environment, policy):
        time_step = environment.current_time_step()
        action_step = policy.action(time_step)
        next_time_step = environment.step(action_step.action)
        traj = trajectory.from_transition(time_step, action_step, next_time_step)

        # Add trajectory to the replay buffer
        self.replay_buffer.add_batch(traj)

    def graph(self, num_iterations, eval_interval, returns):
        steps = range(0, num_iterations + 1, eval_interval)
        plt.plot(steps, returns)
        plt.ylabel('Average Return')
        plt.xlabel('Step')
        plt.ylim(top=250)
        plt.show()


if __name__ == "__main__":
    # Create Environments
    map_shape = (100, 100)
    number_of_zebros = 5
    zebro_step_size = 3
    visible_radius = 2

    train_py_env = ZebroEnvironment(map_shape, number_of_zebros, zebro_step_size, visible_radius)
    eval_py_env = ZebroEnvironment(map_shape, number_of_zebros, zebro_step_size, visible_radius)

    # Use TF Environment Wrappers to translate them for TF
    train_env = tf_py_environment.TFPyEnvironment(train_py_env)
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)
