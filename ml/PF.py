import copy
import time
import json
import matplotlib.pyplot as plt
import numpy as np
from core.system import *
from core.tp_parser import parse
import redis

class ParticleFilter:
    """
    A Particle Filter implementation for estimating system state.
    """

    def __init__(self, num_particles, system):
        """
        Initialize the Particle Filter.

        Args:
            num_particles (int): Number of particles to use in the filter.
            system (System): The system being modeled.
        """
        self.num_particles = num_particles
        self.system = system
        self.particles = np.array([self.initial_state_distribution() for _ in range(num_particles)])
        self.distances = []
        self.pf_norms = []
        self.true_norms = []
        self.particle_states = []
        self.breakage_counts = []

    def initial_state_distribution(self):
        """
        Generate an initial particle state.

        Returns:
            dict: A dictionary representing a particle's initial state and weight.
        """
        particle = {
            "state": {
                "system": copy.deepcopy(self.system),
                "tank_alarms": np.zeros(len(self.system.get_all_tanks()) - 2, dtype=int),
            },
            "weight": 1 / self.num_particles,
        }

        # Initialize tank levels with some randomness
        for id in particle["state"]["system"].get_all_tanks():
            if "S" not in id:
                true_value = self.system.get_tank(id).contains
                particle["state"]["system"].get_tank(id).contains = max(
                    0, min(true_value + np.random.randint(-10, 11), 100)
                )

        # Initialize connector flows with some randomness
        for id in particle["state"]["system"].get_all_connectors():
            if "RP" not in id and "PP" not in id:
                true_flow = self.system.get_connector(id).desired_flow
                particle["state"]["system"].get_connector(id).desired_flow = true_flow + np.random.choice(
                    [0, 1, 2, 3, 4, 5]
                )

        return particle
    
    def check_and_send_recommendations(self, redis_client, latest_observation):
        """
        Check for critical conditions and send recommendations via Redis.

        This method checks for two conditions:
        1. Connectors with 50% or higher probability of being broken.
        2. Tanks with 80% or higher probability of high alarm.

        If either condition is met, a JSON message is sent to the 'recommendations' 
        channel on Redis.

        Args:
            redis_client (redis.StrictRedis): An instance of Redis client for publishing messages.
            latest_observation (np.array): The latest observed state of the system.
        """
        # Check for broken connectors
        breakage_probs = self.count_particles_with_breakages()
        for connector, prob in breakage_probs.items():
            if float(prob) >= 0.1:  # 50% or higher probability
                message = {
                    "type": "connector_warning",
                    "connector": connector,
                    "probability": float(prob),
                    "latest_state": latest_observation.tolist()
                }
                redis_client.publish('recommendations', json.dumps(message))
        
        # Check for high alarms
        high_alarm_probs = self.count_high_alarms()
        for tank, prob in high_alarm_probs.items():
            if prob >= 0.3:  # 80% or higher probability
                message = {
                    "type": "high_alarm_alert",
                    "tank": tank,
                    "probability": prob,
                    "latest_state": latest_observation.tolist()
                }
                redis_client.publish('recommendations', json.dumps(message))

    def count_high_alarms(self):
        """
        Count the number of particles with high alarms for each tank.

        Returns:
            dict: A dictionary with tank IDs as keys and the fraction of particles
                  where that tank has a high alarm as values.
        """
        high_alarm_counts = {tank: 0 for tank in self.system.get_all_usable_tanks()}
        for particle in self.particles:
            for tank_id, tank in particle['state']['system'].get_all_usable_tanks().items():
                if isinstance(tank.triggered_alarm, HighAlarm):
                    high_alarm_counts[tank_id] += 1
        
        return {k: v / self.num_particles for k, v in high_alarm_counts.items()}


    def update_weights(self, observation):
        """
        Update particle weights based on the latest observation.

        Args:
            observation (np.array): The observed state of the system.
        """
        for particle in self.particles:
            likelihood = self.observation_model(observation, particle)
            particle["weight"] *= likelihood
        self.normalize_weights()

    def normalize_weights(self):
        """Normalize the weights of all particles."""
        total_weight = sum(p["weight"] for p in self.particles)
        if total_weight > 0:
            for p in self.particles:
                p["weight"] /= total_weight
        else:
            # If all weights are zero, reinitialize with equal weights
            for p in self.particles:
                p["weight"] = 1.0 / len(self.particles)

    def observation_model(self, observation, particle):
        """
        Calculate the likelihood of an observation given a particle's state.

        Args:
            observation (np.array): The observed state of the system.
            particle (dict): A particle representing a possible system state.

        Returns:
            float: The likelihood of the observation given the particle's state.
        """
        particle_state = self.state_to_1d_array(particle["state"]["system"])
        observation_state = np.array(observation)

        # Ensure both states have the same shape
        if particle_state.shape != observation_state.shape:
            print(f"Shape mismatch: particle_state {particle_state.shape}, observation_state {observation_state.shape}")
            max_length = max(len(particle_state), len(observation_state))
            particle_state = np.pad(particle_state, (0, max_length - len(particle_state)))
            observation_state = np.pad(observation_state, (0, max_length - len(observation_state)))

        # Calculate likelihood using Gaussian noise model
        diff = particle_state - observation_state
        sigma = 1.0  # Noise standard deviation (adjust as needed)
        likelihood = np.prod(np.exp(-0.5 * (diff / sigma)**2) / (sigma * np.sqrt(2 * np.pi)))

        # Store state information for later analysis
        self.particle_states.append(particle_state)
        self.pf_norms.append(np.exp(-np.linalg.norm(particle_state)))
        self.true_norms.append(np.exp(-np.linalg.norm(observation_state)))
        self.distances.append(np.linalg.norm(diff))

        return likelihood

    def state_to_1d_array(self, system):
        """
        Convert the system state to a 1D numpy array.

        Args:
            system (System): The system to convert.

        Returns:
            np.array: 1D array representing the system state.
        """
        valve_states = [
            int(valve[1].desired_flow / 5) for valve in system.get_all_valves().items()
        ]
        alarm_states = [
            (2 if isinstance(alarm[1].triggered_alarm, HighAlarm)
             else 1 if isinstance(alarm[1].triggered_alarm, LowAlarm) else 0)
            for alarm in system.get_all_usable_tanks().items()
        ]
        return np.array(valve_states + alarm_states)

    def systematic_resample(self):
        """Perform systematic resampling of particles."""
        weights = np.array([p["weight"] for p in self.particles])
        N = len(weights)
        positions = (np.arange(N) + np.random.random()) / N

        indices = np.zeros(N, 'i')
        cumulative_sum = np.cumsum(weights)
        i, j = 0, 0
        while i < N:
            if positions[i] < cumulative_sum[j]:
                indices[i] = j
                i += 1
            else:
                j += 1

        self.particles = [copy.deepcopy(self.particles[i]) for i in indices]
        for p in self.particles:
            p["weight"] = 1.0 / N

    def resample_particles(self):
        """Resample particles if the effective sample size is too low."""
        if self.effective_sample_size() < self.num_particles / 2:
            self.systematic_resample()
            for particle in self.particles:
                self.add_noise_to_particle(particle)

    def add_noise_to_particle(self, particle):
        """
        Add noise to a particle's state to introduce variety.

        Args:
            particle (dict): The particle to add noise to.
        """
        system = particle["state"]["system"]
        for tank in system.get_all_tanks().values():
            if "S" not in tank.tank_id:
                tank.contains += np.random.normal(0, 2)  # Adjust standard deviation as needed
                tank.contains = max(0, min(tank.contains, tank.capacity))

        for name, connector in system.get_all_connectors().items():
            if "RP" not in name and "PP" not in name:
                connector.desired_flow += np.random.normal(0, 0.5)  # Adjust as needed
                connector.desired_flow = max(0, min(connector.desired_flow, connector.max_flow))

    def effective_sample_size(self):
        """
        Calculate the effective sample size to determine when to resample.

        Returns:
            float: The effective sample size.
        """
        weights = np.array([particle['weight'] for particle in self.particles])
        return 1.0 / np.sum(weights ** 2)

    def count_particles_with_breakages(self):
        """
        Count the number of particles with broken connectors.

        Returns:
            dict: A dictionary with connector IDs as keys and the fraction of particles
                  where that connector is broken as values.
        """
        counts = {connector: 0 for connector in self.system.get_all_connectors()}
        for particle in self.particles:
            for valve_id, valve in particle['state']['system'].get_all_connectors().items():
                if valve.is_broken():
                    counts[valve_id] += 1
        counts = {k: format(v / self.num_particles, '.4f') for k, v in counts.items()}
        self.breakage_counts.append(counts)
        return counts

class Env:
    """
    Environment class to simulate the system and manage the particle filter.
    """

    def __init__(self, tnp_file):
        """
        Initialize the environment.

        Args:
            tnp_file (str): Path to the TNP configuration file.
        """
        self.system = self.parse_file(tnp_file)
        self.particle_filter = ParticleFilter(num_particles=1000, system=self.system)
        self.breakage_prob = 0.01

    def parse_file(self, tnp_file):
        """
        Parse the TNP configuration file.

        Args:
            tnp_file (str): Path to the TNP configuration file.

        Returns:
            System: The parsed system configuration.
        """
        return parse(tnp_file)[0]

    def simulate_step(self):
        """Simulate one step of the system for all particles."""
        for particle in self.particle_filter.particles:
            system = particle["state"]["system"]
            self.simulate_flow(system)
            self.add_remove_alarms((0.3, 0.7), (0.2, 0.8))
            self.simulate_breakages(particle=particle)

    def simulate_flow(self, system):
        """
        Simulate flow through the system for one time step.

        Args:
            system (System): The system to simulate.
        """
        changes = {tank: 0 for tank in system.get_all_tanks().keys()}
        for name, connector in system.get_all_connectors().items():
            if not connector._broken:
                if "PP" in name or "RP" in name:
                    # Simulate pipe flow
                    flow = min(connector.desired_flow, connector.source.contains)
                    changes[connector.source.tank_id] -= flow
                    changes[connector.sink.tank_id] += flow
                elif ("VL" in name or "TP" in name) and connector.desired_flow > 0:
                    # Simulate valve flow
                    flow = min(connector.desired_flow, connector.source.contains)
                    changes[connector.source.tank_id] -= flow
                    changes[connector.sink.tank_id] += flow

        # Update tank levels
        for tank_id, flow in changes.items():
            tank = system.get_tank(tank_id)
            if tank.tank_id not in ["SR", "SN"]:
                tank.contains = max(0, min(tank.contains + flow, tank.capacity))

    def add_remove_alarms(self, yellow_alarm_ranges, red_alarm_ranges):
        """
        Add or remove alarms based on the tank levels.

        Args:
            yellow_alarm_ranges (tuple): Thresholds for yellow alarms (lower, upper).
            red_alarm_ranges (tuple): Thresholds for red alarms (lower, upper).
        """
        for tank in self.system.get_all_tanks().values():
            if "S" not in tank.tank_id:
                current_level = tank.contains
                tank_capacity = tank.capacity

                # Check for red alarm
                red_lower, red_upper = red_alarm_ranges
                if int(red_lower * tank_capacity) <= current_level <= int(red_upper * tank_capacity):
                    tank.triggered_alarm = HighAlarm(self, current_level)
                # Check for yellow alarm
                else:
                    yellow_lower, yellow_upper = yellow_alarm_ranges
                    if int(yellow_lower * tank_capacity) <= current_level <= int(yellow_upper * tank_capacity):
                        tank.triggered_alarm = LowAlarm(self, current_level)
                    else:
                        tank.triggered_alarm = None

    def simulate_breakages(self, particle):
        """
        Simulate breakages in the system's connectors.

        Args:
            particle (dict): The particle to simulate breakages for.
        """
        system = particle["state"]["system"]

        for valve_id, valve in system.get_all_connectors().items():
            system.get_connector(valve_id).breakage_timer -= 1
            if system.get_connector(valve_id).breakage_timer <= 0:
                system.get_connector(valve_id)._broken = False

            if np.random.randint(100) <= valve.p_break * 10:
                system.get_connector(valve_id)._broken = True
                system.get_connector(valve_id).breakage_timer = np.random.randint(15, 41)

def read_env_states(file_path): 
    """
    Read environment states from a JSON file.

    Args:
        file_path (str): Path to the JSON file containing environment states.

    Returns:
        dict: A dictionary of environment states, or an empty dict if file not found or invalid.
    """
    try:
        with open(file_path, "r") as f:
            env_states = json.load(f)
        return env_states
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return {}
    except json.JSONDecodeError:
        print(f"Error: The file {file_path} is not valid JSON.")
        return {}

def receive_real_time_update(redis_client):
    """
    Get the latest environment state from Redis.

    Args:
        redis_client (redis.StrictRedis): Redis client instance.

    Returns:
        np.array: The latest state from Redis, or None if not available.
    """
    state_json = redis_client.get("system_state")
    if state_json:
        state = json.loads(state_json)
        return np.array(state)
    return None

def plot_particles_distance(observation_list):
    """
    Plot the mean particle distance over time.

    Args:
        observation_list (list): List of particle distances.
    """
    observation_mean_distance = [
        np.mean(np.array(observation_list[i : i + 100]))
        for i in range(0, len(observation_list), 100)
    ]

    plt.figure()
    plt.plot(observation_mean_distance, label="Mean Particle Distance", color="green")
    plt.title("Mean Particle Distance Graph")
    plt.xlabel("Epoch")
    plt.ylabel("Distance")
    plt.legend()
    plt.grid(True)
    plt.savefig("particle_distance.png")

def plot_state_comparison(true_states, pf_states):
    """
    Plot a comparison of true states vs. particle filter estimated states.

    Args:
        true_states (list): List of true system states.
        pf_states (list): List of particle filter estimated states.
    """
    plt.figure(figsize=(10, 6))
    steps = range(len(true_states) // 100)

    true_mean_state = [
        np.mean(np.array(true_states[i : i + 100]))
        for i in range(0, len(true_states), 100)
    ]
    pf_mean_state = [
        np.mean(np.array(pf_states[i : i + 100])) for i in range(0, len(pf_states), 100)
    ]

    plt.plot(steps, true_mean_state, label="Observed States", color="blue", marker="o")
    plt.plot(
        steps, pf_mean_state, label="PF State", color="red", linestyle="--", marker="x"
    )

    plt.xlabel("Step")
    plt.ylabel("State")
    plt.title("True vs PF States")
    plt.legend()
    plt.grid(True)
    plt.savefig("true_vs_pf.png")

def monitor():
    """
    Main monitoring function to run the particle filter simulation.
    """
    r = redis.StrictRedis(host="localhost", port=6379)
    env = Env("configurations/sample.tnp")
    
    step = 0
    last_update_time = time.time()

    while True:
        env.simulate_step()
        
        # Check if 10 seconds have passed since the last update
        current_time = time.time()
        if current_time - last_update_time >= 10:
            observation = receive_real_time_update(r)
            last_update_time = current_time
            
            if observation is not None:
                env.particle_filter.update_weights(observation)
                env.particle_filter.resample_particles()
                env.particle_filter.check_and_send_recommendations(r, observation)

                # Count particles with breakages
                num_particles_with_breakages = env.particle_filter.count_particles_with_breakages()

                print(f"Step {step}, Observation: {observation}")
                print(f"Breakage probabilities: {num_particles_with_breakages}")
                
                # Publish the formatted dictionary to the Redis channel
                r.publish('breakage_probs', json.dumps(num_particles_with_breakages))
                
                step += 1
            else:
                print("Waiting for system state update from UI...")

        time.sleep(1)  # Sleep for 1 second to prevent busy-waiting

if __name__ == "__main__":
    monitor()