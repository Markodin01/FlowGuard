# state_manager.py
import redis
import jsonpickle
import networkx as nx

class StateManager:
    def __init__(self, redis_host='localhost', redis_port=6379, redis_db=0):
        self.redis = redis.StrictRedis(host=redis_host, port=redis_port, db=redis_db)
    
    def get_node_positions(self, system_json, fixed_node_pos_json):
        pass

    def process_system(self, tp_string):
        pass

from state_manager import StateManager
state_manager = StateManager()