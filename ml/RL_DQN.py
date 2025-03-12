from core.system import *
import gym
from gym import spaces
import numpy as np
from core.tp_parser import parse
from tqdm import tqdm
import matplotlib.pyplot as plt
from core.model import LowAlarm, HighAlarm
import logging
from datetime import datetime
import shutil
import copy

CONFIGURATION_PATH = "configurations/sample.tnp"
START_TIME_STRING = datetime.now().strftime('%Y%m%d-%H%M%S')


class SystemEnvironment(gym.Env):
    """
    A custom environment that simulates a system of tanks, valves, and pipes.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self, tnp, reward_step, reward_fail, reward_finish, breakage_prob=0.01
    ):
        """
        Initialize the environment with a configuration path.

        Args:
        tnp_path (str): Path to the configuration file.
        reward_step (int): Reward for each step.
        reward_fail (int): Reward for failure.
        reward_finish (int): Reward for finish.
        breakage_prob (float): Probability of valve or tap breakage at each step.
        """
        super(SystemEnvironment, self).__init__()

        self.breakage_prob = breakage_prob
        self.broken_valves = {}  # Dictionary to keep track of broken valves and their breakage steps

        if isinstance(tnp, str) and tnp != "":
            self.system_config = self.parse_tnp_file(tnp)
            self.setup_initial_state()
        else:
            self.system_config = copy.deepcopy(tnp)
            self.setup_initial_state()

        self.reward_step = reward_step
        self.reward_fail = reward_fail
        self.reward_finish = reward_finish
        self.action_space = spaces.Discrete(2 ** len(self.valves))
        self.observation_space = spaces.MultiDiscrete(
            [2] * len(self.valves) + [3] * (len(self.tanks) - 2)
        )

        self.current_step = 0
        self.max_steps = 2000
        self.state = None
        self.reset()

        self.balance_tanks = (301, 699)

        # # Setup logging
        # logging.basicConfig(filename=f"logs/{START_TIME_STRING}_agent_and_environment.log", level=logging.INFO)
        # self.logger = logging.getLogger(__name__)
        # shutil.copyfile(CONFIGURATION_PATH, f"logs/{START_TIME_STRING}_used_configuration.tnp")


    def step(self, action, prev_action, total_reward):
        """
        Apply the given action, update the environment state, calculate reward, and check if episode is over.

        Args:
        action (int): The action to take.

        Returns:
        tuple: Returns the new state, the reward, whether the episode is over, and additional info.
        """
        temp_valve_states = [valve.desired_flow for valve in self.valves.values()]
        action_states = self.use_agent_action(action)

        # Facilitate no action taken logic
        if action == prev_action:
            # valve_states = temp_valve_states
            no_action_taken = True
        else:
            no_action_taken = False

        done = False
        reward = 0

        # Simulate one tick
        self.tanks = self.simulate_tick()

        # TODO: read values from config and use them instead of hard-coded values
        self.add_remove_alarms((20, 70), (10, 80))

        # # Check for breakages
        # self.simulate_breakages()
        # Check if any tank is overfilled
        done = any(
            (tank.contains >= (tank.capacity - 25) or tank.contains <= 25)
            for tank in self.tanks.values()
            if tank.tank_id != "SN" and tank.tank_id != "SR"
        )

        ## currently, fail reward is set to zero, to prevent false negative highlight of the last game step
        # if done:
        #     reward = -self.reward_fail
        # else:
        #     reward = self.reward_step
        
        reward = self.reward_step

        # Calculate balance reward
        reward_for_balance, is_balanced = self.calculate_balance_reward(self.tanks)
        reward += reward_for_balance

        # if no_action_taken and is_balanced:
        #     reward += 3

        self.state = self.update_state_after_step(action_states)

        self.current_step += 1
        if self.current_step > self.max_steps:
            done = True
            reward = self.reward_finish
            return self.state, reward, done, self.tank_levels()

        # if self.current_step % 100 == 0 and self.current_step != 0:
        #     reward += 100

        # if not no_action_taken:
        #     self.logger.info(
        #         f"Step {self.current_step}, Action: {action_states}, Total Reward: {total_reward + reward}, Reward: {reward}"
        #     )
        # else:
        #     self.logger.info(
        #         f"Step {self.current_step}, Action: {action_states}, No Action Taken, Total Reward: {total_reward + reward}, Reward: {reward}"
        #     )

        # self.log_state()
        return self.state, reward, done, self.tank_levels()

    def tank_levels(self):
        return [tank.contains for tank in self.tanks.values() if "S" not in tank.tank_id]

    def simulate_breakages(self):
        """
        Simulate random breakages of valves or taps.
        """
        for name, connector in self.connectors.items():
            if random.random() < connector.p_break / 10 and name not in list(
                self.broken_valves.keys()
            ):
                connector.is_broken = True
                self.broken_valves[name] = (
                    self.current_step
                )  # Track the step number when the valve broke
                self.logger.info(f"Breakage occurred in connector: {name}")

    def heal_valves(self):
        """
        Heal broken valves after a certain number of steps.
        """
        to_heal = [
            (name, step)
            for name, step in self.broken_valves.items()
            if self.current_step - step >= np.random.randint(15, 40)
        ]
        for name, step in to_heal:
            valve = self.connectors[name]
            valve.desired_flow = 0
            self.logger.info(f"Valve healed: {name} after {self.current_step - step} steps")
            del self.broken_valves[name]

    def add_remove_alarms(self, yellow_alarm_ranges, red_alarm_ranges):
        """
        Add or remove alarms based on the tank levels.

        Args:
        yellow_alarm_ranges (tuple): Thresholds for yellow alarms.
        red_alarm_ranges (tuple): Thresholds for red alarms.
        """
        yellow_alarm_ranges = [(0, 300)]    #low alarm
        red_alarm_ranges = [(700, 1000)]    #high alarm

        for tank in self.tanks.values():
            if "S" not in tank.tank_id:
                current_level = tank.contains

                if any(lower <= current_level <= upper for lower, upper in red_alarm_ranges):
                    tank.triggered_alarm = HighAlarm(self, current_level)
                elif any(lower <= current_level <= upper for lower, upper in yellow_alarm_ranges):
                    tank.triggered_alarm = LowAlarm(self, current_level)
                else:
                    tank.triggered_alarm = None

        return None

    def alarms_to_state(self):
        """
        Convert alarms to state representation.

        Returns:
        list: A list of alarm states.
        """
        alarm_states = []
        for tank in self.tanks.values():
            if "S" not in tank.tank_id:
                if tank.triggered_alarm is None:
                    alarm_states.append(0)
                elif isinstance(tank.triggered_alarm, LowAlarm):
                    alarm_states.append(1)
                elif isinstance(tank.triggered_alarm, HighAlarm):
                    alarm_states.append(2)
        return alarm_states

    def calculate_balance_reward(self, tanks):
        """
        Calculate reward based on the balance of tank levels.

        Args:
        tanks (dict): Dictionary of tank objects.

        Returns:
        tuple: Reward for balance and a boolean indicating if all tanks are balanced.
        """
        reward = 0
        for tank in tanks.values():
            if (tank.tank_id != "SR" and tank.tank_id != "SN"):
                if self.balance_tanks[0] <= tank.contains <= self.balance_tanks[1]:
                    reward += 1
                elif tank.contains > self.balance_tanks[1] or tank.contains < self.balance_tanks[0]:     # penalty for high and low alarm
                    reward -= 0 
        # If all tanks are balanced...
        if reward >= 4:
            return reward, True
        return reward, False

    def update_state_after_step(self, valve_states):
        """
        Update the state of the environment after each step.

        Args:
        valve_states (list): List of valve states.

        Returns:
        list: Updated state of the environment.
        """
        if any(valve == 5 for valve in valve_states):
            valve_states = [valve // 5 for valve in valve_states]
        return valve_states + self.alarms_to_state()

    def use_agent_action(self, action):
        """
        Apply the action taken by the agent to the environment.

        Args:
        action (int): The action taken by the agent.

        Returns:
        list: List of valve states after the action is applied.
        """
        action_states = [int(x) for x in format(action, "04b")]
        for (name, valve), state in zip(self.valves.items(), action_states):
            if name not in self.broken_valves.keys():
                if state == 0:
                    valve.desired_flow = valve.min_flow
                else:
                    valve.desired_flow = valve.max_flow
        return action_states

    def simulate_tick(self):
        """
        Simulate one time step in the environment.

        Returns:
        dict: Updated tank levels after the tick.
        """
        changes = {tank_id: 0 for tank_id in self.tanks}
        for name, connector in self.connectors.items():
            if not connector.is_broken:
                if "PP" in name or "RP" in name:
                    if str(connector.source) != "Sink":
                        flow = min(
                            connector.desired_flow,
                            (connector.source.contains if str(connector.source) != "Source" else connector.desired_flow)
                        )
                        if str(connector.source) != "Source":
                            changes[connector.source.tank_id] -= flow
                        if str(connector.sink) != "Sink":
                            changes[connector.sink.tank_id] += flow
                if ("VL" in name or "TP" in name) and connector.desired_flow > 0:
                    if str(connector.source) != "Sink":
                        flow = min(
                            connector.desired_flow,
                            (connector.source.contains if str(connector.source) != "Source" else connector.desired_flow)
                        )
                        if str(connector.source) != "Source":
                            changes[connector.source.tank_id] -= flow
                        if str(connector.sink) != "Sink":
                            changes[connector.sink.tank_id] += flow

        for tank_id, flow in changes.items():
            if self.tanks[tank_id].tank_id not in ["SR", "SN"]:
                new_level = self.tanks[tank_id].contains + flow
                self.tanks[tank_id].contains = max(0, min(new_level, self.tanks[tank_id].capacity))
        return self.tanks

    def reset(self):
        """
        Reset the environment to its initial state.

        Returns:
        list: Initial state of the environment.
        """
        self.setup_initial_state()
        self.current_step = 0
        self.broken_valves.clear()  # Clear broken valves
        return [0] * (len(self.valves) + len(self.tanks) - 2)

    def setup_initial_state(self):
        """
        Setup the initial state of the environment.
        """
        for connector in self.system_config.get_all_connectors().values():
            connector.is_broken = False

        self.tanks = self.system_config.get_all_tanks()
        for tank in self.tanks.values():
            tank.contains = 310
        self.valves = {
            k: v
            for k, v in self.system_config.get_all_connectors().items()
            if k.startswith("VL") or k.startswith("TP")
        }
        self.pipes = {
            k: v
            for k, v in self.system_config.get_all_connectors().items()
            if k.startswith("PP") or k.startswith("RP")
        }
        self.connectors = self.system_config.get_all_connectors()

    def parse_tnp_file(self, path):
        """
        Parse the TNP configuration file.

        Args:
        path (str): Path to the TNP configuration file.

        Returns:
        System: Parsed system configuration.
        """
        return parse(path)[0]

    def render(self, mode="human", close=False):
        """
        Render the current state of the environment.

        Args:
        mode (str): Mode for rendering.
        close (bool): Flag to close the rendering.
        """
        print(f"Current state: {self.state}")

    def log_state(self):
        """
        Log the current state of the environment.
        """
        state_info = {
            "Step": self.current_step,
            "Alarms": self.alarms_to_state(),
            "Valve States": [valve.desired_flow for valve in self.valves.values()],
            "Tank Fillings": [
                tank.contains for tank in self.tanks.values() if "S" not in tank.tank_id
            ],
        }
        self.logger.info(f"System State: {state_info}")


### Helper functions for agent to be able to use the state info ###

def state_to_index(state):
    """
    Convert the system's state into a unique index for accessing the Q-table.

    This function combines both valve states and discretized tank levels into a single index,
    which can be used to reference a specific row in the Q-table where corresponding Q-values are stored.

    Args:
        state (list): The current state of the environment consisting of valve states and tank levels.

    Returns:
        int: A unique index representing the given state in the Q-table.
    """
    valve_states = state[:4]  # Assuming first four entries are valve states
    alarm_states = state[4:]  # Remaining entries are tank levels

    # Combine valve states and discretized tank levels
    full_state = valve_states + alarm_states

    # Calculate the cumulative index for the Q-table
    base_sizes = [2] * len(valve_states) + [3] * len(alarm_states)
    index = 0
    for state_value, base in zip(reversed(full_state), reversed(base_sizes)):
        index = index * base + state_value

    return index


import tensorflow as tf

class LSTMAgent:
    def __init__(self, state_size, action_size, sequence_length=10):
        self.state_size = state_size
        self.action_size = action_size
        self.sequence_length = sequence_length
        self.model = make_lstm_model(state_size, action_size, sequence_length=sequence_length)
        self.target_model = make_lstm_model(state_size, action_size, sequence_length=sequence_length)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.loss_function = tf.keras.losses.MeanSquaredError()
        self.gamma = 0.95
        self.update_target_network()

    def choose_action(self, state_sequence, epsilon):
        if np.random.rand() < epsilon:
            return np.random.randint(self.action_size)
        else:
            if (state_sequence.shape != (1,10,8)):
                print("Invalid shape", state_sequence.shape)
            q_values = self.model(state_sequence)
            return np.argmax(q_values.numpy()[0])

    # @tf.function
    def train(self, states, next_states, actions, rewards, dones):
        # states, actions, rewards, next_states, dones = batch
        
        # states = tf.convert_to_tensor(states, dtype=tf.float32)
        # next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        # actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        # rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        # dones = tf.convert_to_tensor(dones, dtype=tf.float32)


        with tf.GradientTape() as tape:
            # Q-values for current states
            q_values = self.model(states)
            print(states.shape, q_values.shape)
            current_q = tf.reduce_sum(tf.one_hot(actions, self.action_size, dtype=tf.float32) * q_values, axis=1)

            # Target Q-values
            next_q_values = self.target_model(next_states)
            max_next_q = tf.reduce_max(next_q_values, axis=1)
            target_q = rewards + (1 - dones) * self.gamma * max_next_q

            # Compute loss
            loss = self.loss_function(target_q, current_q)

        # Compute gradients and update the model
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return loss, 0

    def update_target_network(self):
        self.target_model.set_weights(self.model.get_weights())


def make_dqn_model(num_inputs, num_actions):
    print(num_inputs)
    inputs = tf.keras.layers.Input((8,))
    x = tf.keras.layers.Dense(128, activation='relu')(inputs)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    outputs = tf.keras.layers.Dense(num_actions, activation='relu')(x)

    return tf.keras.Model(inputs, outputs)


class AdvantageLayer(tf.keras.layers.Layer):
    # @tf.function
    def call(self, x):
        return (x - tf.reduce_mean(x, axis=1, keepdims=True))

def make_lstm_model(num_inputs, num_actions, lstm_units=64, sequence_length=10):
    inputs = tf.keras.layers.Input((sequence_length, num_inputs,))

    lstm = tf.keras.layers.LSTM(lstm_units, return_sequences=False)(inputs)
    x = tf.keras.layers.Dense(256, activation='softmax')(lstm)


    value = tf.keras.layers.Dense(128, activation='relu')(x)
    value  = tf.keras.layers.Dense(1)(value)

    advantage = tf.keras.layers.Dense(128, activation='softmax')(x)
    advantage = tf.keras.layers.Dense(num_actions)(advantage)

    avg_advantage = AdvantageLayer()(advantage)

    q_values = value - avg_advantage

    return tf.keras.Model(inputs, q_values)

class ReplayBuffer:
    def __init__(self, capacity=100_000):
        """
        Initialize the replay buffer.

        Args:
        capacity (int): Capacity of the replay buffer.
        """
        self.buffer = np.zeros((capacity,), dtype=object)
        self.capacity = capacity
        self.position = 0
        self.is_full = False

    def add(self, experience):
        """
        Add an experience to the replay buffer.

        Args:
        experience (tuple): Experience to add.
        """
        self.buffer[self.position] = experience

        if not self.is_full and self.position + 1 == self.capacity:
            self.is_full = True
        self.position = (self.position + 1) % self.capacity


    def sample(self, batch_size):
        """
        Sample a batch of experiences from the replay buffer.

        Args:
        batch_size (int): Size of the batch to sample.

        Returns:
        list: A batch of experiences.
        """
        mask = np.ones((self.capacity,), dtype=np.float64)
        if not self.is_full:
            mask[self.position:self.capacity] = 0
            mask = mask/(self.position)
        else:
            mask = mask/self.capactity

        return np.random.choice(self.buffer, size=batch_size, p=mask)

    def choose_action(self, state, epsilon):
        self.update_state_memory(state)
        if np.random.rand() < epsilon:
            return np.random.randint(self.action_size)
        else:
            state_sequence = self.state_memory.reshape(1, self.sequence_length, self.state_size)
            q_values = self.model(state_sequence)
            return np.argmax(q_values.numpy()[0])

    def train(self, batch):
        states, actions, rewards, next_states, dones = batch
        
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)

        with tf.GradientTape() as tape:
            # Q-values for current states
            q_values = self.model(states)
            current_q = tf.reduce_sum(tf.one_hot(actions, self.action_size) * q_values, axis=1)

            # Target Q-values
            next_q_values = self.target_model(next_states)
            max_next_q = tf.reduce_max(next_q_values, axis=1)
            target_q = rewards + (1 - dones) * self.gamma * max_next_q

            # Compute loss
            loss = self.loss_function(target_q, current_q)

gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        # Restrict TensorFlow to only use the first GPU
        tf.config.experimental.set_visible_devices(gpus[0], "GPU")
        logical_gpus = tf.config.experimental.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        print(e)
        
    def update_target_network(self):
        self.target_model.set_weights(self.model.get_weights())

from functools import reduce
import traceback
import asyncio
from concurrent.futures import ThreadPoolExecutor

# TODO: pass in the configuration file as an object and create environment based on it
def run_episode(agent: LSTMAgent, epsilon: float, tnp_config) -> list[tuple[list[int], int, int, list[int], bool]]:
    # Initialize the custom environment with a configuration file and reward settings
    env = SystemEnvironment(tnp_config, 1, 0, 1000)
    # Reset the environment and get the initial state
    state = env.reset()
    total_reward = 0  # Initialize total reward for the episode
    done = False  # Flag to indicate if the episode is finished
    steps = 0  # Step counter for the current episode
    prev_action = -1    # Initialize int for previous action taken by agent
    game_log = []
    tank_levels = []
    states = np.zeros((1,10, 8), dtype="float32")

    # Episode loop until the environment signals done
    while not done:

        # Only choose an action based on the current state and exploration rate (epsilon), if there is at least one
        # alarm (last four elements in state array)
        # if 1 in state[-4:] or 2 in state[-4:] or prev_action == -1:
        #     action = agent.choose_action(state, epsilon)
        # else:
        #     action = prev_action
        states = np.roll(states, -1, axis=1) #states[:, 1:, :]
        states[0][-1] = np.array(state)
        # state_np = np.array(state)
        # states = np.insert(states, states.shape[1], state_np, axis=1) 
        if (states.shape != (1, 10, 8)):
            print(state)
            print(states.shape)
            print(states)
        action = agent.choose_action(states, epsilon)
        # Perform the action in the environment and get the next state, reward, and done flag
        next_state, reward, done, levels = env.step(action, prev_action, total_reward)
        total_reward += reward  # Accumulate the reward
        prev_action = action
        tank_levels.append(levels)
        # Add the experience to the replay buffer
        game_log.append((state, action, reward, next_state, done))
        # Update the current state to the next state
        state = next_state
        steps += 1  # Increment the step counter
    
    return game_log, tank_levels


def flatten(xss):
    return [x for xs in xss for x in xs]

# TODO: 
def no_post_processing(episode_log):
    return episode_log

reward_index = 2


def sliding_window_processing(look_back, look_ahead):
    def slding_window_internal(episode_log):
        new_log = []
        for i in range(len(episode_log)):
            start = min(i, i - look_back)
            end = min(len(episode_log), i + look_ahead)

            step_slice = episode_log[start:end]

            average_reward = reduce(lambda x, y: x + y, map(lambda x: x[reward_index], step_slice), 0)

            new_log.append((episode_log[i][0], episode_log[i][1], average_reward, episode_log[i][3], episode_log[i][4]))
        
        return new_log

    return slding_window_internal

def episode_logging(episode_log, tank_log):
    # TODO: add logging of a game episode
    assert len(episode_log) == len(tank_log)
    try:
        with open(f"logs/{START_TIME_STRING}_agent_and_env_log.txt", "a") as values_file:
            total_reward = 0
            for count, log in enumerate(episode_log):
                env_state = log[3]
                action_taken = log[1]
                reward = log[2]
                total_reward = total_reward + reward

                state_info = {
                    "Step": count + 1,
                    "Alarms": env_state[4:],
                    "Valve States": env_state[0:4],
                    "Tank Fillings": tank_log[count],
                }
                values_file.write(f"Step {count + 1}, Action: {[int(x) for x in format(action_taken, '04b')]}, Total Reward: {total_reward}, Reward: {reward}\n")
                values_file.write(f"System State: {state_info}\n")
    except Exception as e:
        print("An error occurred:", e)
    pass 


async def training_step(agent, epsilon, num_instances, episode_post_processing, replay_buffer, tnp_config):
    tasks = []
    executor = ThreadPoolExecutor()
    for i in range(num_instances):
        tasks.append(asyncio.get_running_loop().run_in_executor(executor, run_episode, agent, epsilon, tnp_config))

    results = await asyncio.gather(*tasks) 

    episode_results = []
    
    for episode_result, tank_levels in results:
        episode_results.append(episode_result)
        episode_logging(episode_result, tank_levels)

    post_processed_results = map(episode_post_processing, episode_results)
    flattened_result = flatten(post_processed_results)

    for experience in flattened_result:
        replay_buffer.add(experience)

    sequenced = []
    for i in range(agent.sequence_length, len(flattened_result)):
        sequenced.append(flattened_result[i-agent.sequence_length:i])
    states = tf.convert_to_tensor(list(map(lambda x: list(map(lambda y: y[0], x)), sequenced)), dtype=tf.float32)
    next_states = tf.convert_to_tensor(list(map(lambda x: list(map(lambda y: y[3], x)), sequenced)), dtype=tf.float32)

    actions = tf.convert_to_tensor(list(map(lambda x: x[1], flattened_result[agent.sequence_length:])), dtype=tf.int32)
    rewards = tf.convert_to_tensor(list(map(lambda x: x[2], flattened_result[agent.sequence_length:])), dtype=tf.float32)
    dones = tf.convert_to_tensor(list(map(lambda x: x[4], flattened_result[agent.sequence_length:])), dtype=tf.float32)
    loss, accuracy = agent.train(states, next_states, actions, rewards, dones)

    # buffer_sample = replay_buffer.sample(2048)

    # random_loss, random_accuracy = agent.train(buffer_sample)

    steps_per_episode = list(map(len, episode_results))

    return flattened_result, steps_per_episode, loss, accuracy, -1, -1


async def training_loop():
    # TODO: the agent doesn't need the full environment instance, but instead just the space sizes

    # Initial exploration rate for epsilon-greedy policy
    epsilon = 1.0
    # Decay rate for epsilon to decrease exploration over time
    epsilon_decay = 0.995
    # Minimum value for epsilon to ensure some exploration
    min_epsilon = 0.01
    # Number of features in the state representation
    num_features = 10
    # Sequence length for LSTM input (if used)
    sequence_length = 10
    # Number of possible actions the agent can take
    num_actions = 4
    # pass configuration file as a parsed object that will be deeply cloned for each game (faster) or as string of path to file (slower)
    tnp_config = parse("configurations/sample.tnp")[0]
    # tnp_config = "configurations/sample.tnp"
    # Initialize the custom environment with a configuration file and reward settings
    env = SystemEnvironment(tnp_config, 1, 0, 1000)
    # Initialize the DQN agent with the state and action space sizes
    agent = LSTMAgent(env.observation_space.shape[0], env.action_space.n, sequence_length)
    agent.model.summary()

    

    # Number of concurrent episodes to run
    num_instances = 64
    #
    num_episodes = 10_000
    # Lists to track metrics during training
    episode_rewards = []
    steps_per_episode = []
    losses = []
    accuracies = []
    average_steps_per_loop = []
    epsilons = []

    replay_buffer = ReplayBuffer(1_000_000)

    try:
        # Training loop for a specified number of epochs
        for epoch in tqdm(range(int(num_episodes / num_instances)), desc="Training Progress"):
            episode_results, steps_per_episode_run, loss, accuracy, random_loss, random_accuracy = await training_step(agent, epsilon, num_instances, no_post_processing, replay_buffer, tnp_config)

            steps_per_episode = steps_per_episode + steps_per_episode_run
            average_steps_per_loop = average_steps_per_loop + [sum(steps_per_episode_run) / len(steps_per_episode_run)]

            episode_rewards = episode_rewards + list(map(lambda x: x[reward_index], episode_results))

            losses = losses + [loss]
            accuracies = accuracies + [accuracy]
            epsilons = epsilons + [epsilon]

            # Decay epsilon to reduce exploration over time, but keep it above min_epsilon
            epsilon = max(min_epsilon, epsilon * epsilon_decay)

            agent.update_target_network()

            # try:
            #     fig, ax1 = plt.subplots(figsize=(12, 6))
            #     ax1.plot(average_steps_per_loop, label="Avg Steps per Batch")
            #     ax1.set_title("Average Steps per Loop Over Time")
            #     ax1.set_xlabel("Loop")
            #     ax1.set_ylabel("Steps", color='tab:blue')
            #     ax1.tick_params('y', labelcolor='tab:blue')
            #     ax1.grid(True)
            #     ax2 = ax1.twinx()
                
            #     ax2.plot(epsilons, label="Epsilon", color='tab:red')
            #     ax2.set_ylabel("Epsilon", color='tab:red')
            #     ax2.tick_params('y', labelcolor='tab:red')
            #     fig.legend()
            #     fig.tight_layout()
            #     fig.savefig(f"logs/{START_TIME_STRING}_Steps_Per_Episode")
            #     print(f'Saved to logs/{START_TIME_STRING}_Steps_Per_Episode')
            # except Exception as e:
            #     print("An error occurred:")
            #     traceback.print_exc()

            # try:
            #     plt.figure(figsize=(12, 6))
            #     plt.plot(losses, label="Loss per Episode", color="red")
            #     plt.title("Training Loss Over Episodes")
            #     plt.xlabel("Episode")
            #     plt.ylabel("Loss")
            #     plt.legend()
            #     plt.grid(True)
            #     plt.savefig(f"logs/{START_TIME_STRING}_Loss_Per_Episode")
            # except Exception as e:
            #     print("An error occurred:")
            #     traceback.print_exc()

    except KeyboardInterrupt:
        print("Training cancelled, writing results")
    except asyncio.exceptions.CancelledError:
        print("Training cancelled, writing results")

    try:
        with open(f"logs/{START_TIME_STRING}_Game_Values.txt", "w") as values_file:
            values_file.write(f"List of total rewards: {episode_rewards}\n")
            values_file.write(f"List of steps per game: {steps_per_episode}\n")
            values_file.write(f"List of losses: {losses}\n")
            values_file.write(f"List of accuracies: {accuracies}\n")
    except Exception as e:
        print("An error occurred:", e)
    
    try:
        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax1.plot(average_steps_per_loop, label="Avg Steps per Batch")
        ax1.set_title("Average Steps per Loop Over Time")
        ax1.set_xlabel("Loop")
        ax1.set_ylabel("Steps", color='tab:blue')
        ax1.tick_params('y', labelcolor='tab:blue')
        ax1.grid(True)
        ax2 = ax1.twinx()
        
        ax2.plot(epsilons, label="Epsilon", color='tab:red')
        ax2.set_ylabel("Epsilon", color='tab:red')
        ax2.tick_params('y', labelcolor='tab:red')
        fig.legend()
        fig.tight_layout()
        fig.savefig(f"logs/{START_TIME_STRING}_Steps_Per_Episode")
        print(f'Saved to logs/{START_TIME_STRING}_Steps_Per_Episode')
    except Exception as e:
        print("An error occurred:")
        traceback.print_exc()

    try:
        plt.figure(figsize=(12, 6))
        plt.plot(losses, label="Loss per Episode", color="red")
        plt.title("Training Loss Over Episodes")
        plt.xlabel("Episode")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"logs/{START_TIME_STRING}_Loss_Per_Episode")
    except Exception as e:
        print("An error occurred:")
        traceback.print_exc()
            
    try:
        plt.figure(figsize=(12, 6))
        plt.plot(accuracies, label="Accuracy per Episode", color="green")
        plt.title("Training Accuracy Over Episodes")
        plt.xlabel("Episode")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"logs/{START_TIME_STRING}_Accuracy_Per_Episode")
    except Exception as e:
        print("An error occurred:")
        traceback.print_exc()

    try:
        agent.model.save(f"logs/{START_TIME_STRING}_dqn_model.keras")
    except tf.errors.OpError as op_err:
        print("TensorFlow operation error occurred:")
        traceback.print_exc()
    except IOError as io_err:
        print("File I/O error occurred when trying to save the model:")
        traceback.print_exc()
    except Exception as e:
        print("An unspecified error occurred:")
        traceback.print_exc()

    try:
        agent.model.save_weights(f"logs/{START_TIME_STRING}_dqn_model.weights.h5")
    except Exception as e:
        print("An error occurred:")
        traceback.print_exc()

    

if __name__ == '__main__':
    asyncio.run(training_loop())

# asyncio.run(training_loop())

