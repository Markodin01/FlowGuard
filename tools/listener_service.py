import redis
from core.system import *
import numpy as np
import json
import time
import jsonpickle

r = redis.StrictRedis(host='localhost', port=6379, db=0)

class QTable():
    def __init__(self, qtable_file):
        self.q_table = np.load(qtable_file)
        
    def get_action(self, state_index):
        # Retrieve the best action for this state
        # The best action is the one with the highest Q-value in the row corresponding to the state
        best_action = np.argmax(self.q_table[state_index])
        return best_action

    def state_to_index(self, state):
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
        tank_levels = state[4:]   # Remaining entries are tank levels

        # Convert tank levels from continuous values to discrete indices
        discretized_tank_levels = [self.level_to_discrete(level) for level in tank_levels]

        # Combine valve states and discretized tank levels
        full_state = valve_states + discretized_tank_levels

        # Calculate the cumulative index for the Q-table
        base_sizes = [2] * len(valve_states) + [21] * len(discretized_tank_levels)
        index = 0
        for state_value, base in zip(reversed(full_state), reversed(base_sizes)):
            index = index * base + state_value

        return index
    
    def level_to_discrete(self, level):
        """
        Converts a tank level from a continuous scale (0 to 100) into a discrete index (0 to 20).
        Each index represents a range of 5 units on the original scale.
        """
        return int(level // 5)  # Integer division to convert the level to a discrete step


q_table = QTable('rl_agent_q_table.npy')

def listen_for_system_updates():
    pubsub = r.pubsub()
    pubsub.subscribe('system_updates')
    
    print("Listening for messages on 'system_updates' channel to make decision...")

    while True:
        message = pubsub.get_message()
        if message and message['type'] == 'message':
            current_state = jsonpickle.decode(message['data'])

            # Compare current state with the last state
            changes = find_changes(current_state)
            decision = query_q_table(changes)
            emit_decision(decision)
        else:
            print("Something went wrong...")
        
        time.sleep(5)  # Sleep to avoid constant polling
        
def find_changes(current_state):
    """Recursively find differences between two nested dictionaries and flatten the output."""
    system = ''
    
    tank_contains = []
    for tank_id, tank in current_state.get_all_tanks().items():
        if 'S' not in tank_id :
            tank_contains.append(tank.contains)
    print(f'received tank state: {tank_contains}')
            
    valves_status = []
    for valve_id, valve in current_state.get_all_connectors().items():
        if 'VL' in valve_id or 'TP' in valve_id:
            valves_status.append(valve.desired_flow)
    print(f'received valves state: {valves_status}')

    
    return valves_status + tank_contains

def query_q_table(system_state):
    current_state_index = q_table.state_to_index(system_state)
    action = q_table.get_action(current_state_index)
    
    valve_states = [int(x) for x in format(action, '04b')]
    
    return valve_states


def emit_decision(decision):
    r.publish('valve_action', json.dumps(decision))
    print(f'decision sent: {decision}')

if __name__ == "__main__":
    listen_for_system_updates()