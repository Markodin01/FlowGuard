from core.model import *
class System:
    
    """
    Represents the entire simulation environment, organizing and managing 
    various components like tanks, connectors, alarms, and repair teams.
    Utilizes a registry to efficiently manage these components.
    """
    
    def __init__(self):
        
        # Compose Registry into System for
        # interfacing with the model
        self.registry = Registry()
        
        self.visible_tanks=set()
        self.score_conditions=[]
        self.manual_breaks={}
        self.score=0
        self.time=0
        
            
    def reset(self):
        self.time=0
        self.score=0
        for t in self.get_all_tanks().values():
            t.reset()
        for p in self.get_all_connectors().values():
            p.reset()
        for a in self.get_all_alarms().values():
            a.reset()
        for r in self.get_all_repair_teams().values():
            r.reset()
            

    ### TANKS ###

    def add_tank(self, tank_id, tank):
        """
        Adds a tank to the system.

        Parameters:
        - tank_id: Unique identifier for the tank.
        - tank: The tank object to be added.
        """
        self.registry.add('tanks', tank_id, tank)

    def remove_tank(self, tank_id):
        """
        Removes a tank from the system by its identifier.

        Parameters:
        - tank_id: Unique identifier of the tank to be removed.
        """
        self.registry.remove('tanks', tank_id)

    def get_tank(self, tank_id):
        """
        Retrieves a tank by its identifier.

        Parameters:
        - tank_id: Unique identifier of the tank.

        Returns:
        - Tank object associated with the given identifier.
        """
        return self.registry.get('tanks', tank_id)

    def get_all_tanks(self, visible_only=False):
        """
        Returns all tanks, optionally filtered to only include visible tanks.

        Parameters:
        - visible_only: If True, filters to only return visible tanks.

        Returns:
        - Dictionary of tank identifiers to tank objects.
        """
        all_tanks = self.registry.get_all('tanks')
        return {k: v for k, v in all_tanks.items() if v.visible} if visible_only else all_tanks
    
    
    def get_all_usable_tanks(self):
    # no sinks or sources
        return {k: v for k, v in self.get_all_tanks().items() if not k.startswith('S')}

    ### CONNECTORS ###

    def add_connector(self, connector_id, connector):
        """
        Adds a connector to the system.

        Parameters:
        - connector_id: Unique identifier for the connector.
        - connector: The connector object to be added.
        """
        self.registry.add('connectors', connector_id, connector)

    def remove_connector(self, connector_id):
        """
        Removes a connector from the system by its identifier.

        Parameters:
        - connector_id: Unique identifier of the connector to be removed.
        """
        self.registry.remove('connectors', connector_id)

    def get_connector(self, connector_id):
        """
        Retrieves a connector by its identifier.

        Parameters:
        - connector_id: Unique identifier of the connector.

        Returns:
        - Connector object associated with the given identifier.
        """
        return self.registry.get('connectors', connector_id)

    def get_all_connectors(self):
        """
        Returns all connectors in the system.

        Returns:
        - Dictionary of connector identifiers to connector objects.
        """
        return self.registry.get_all('connectors')
    
    def get_all_valves(self):
        
        return {k: v for k, v in self.get_all_connectors().items() if k.startswith('VL') or k.startswith('TP')}

    ### ALARMS ###

    def add_alarm(self, alarm_id, alarm):
        """
        Adds an alarm to the system.

        Parameters:
        - alarm_id: Unique identifier for the alarm.
        - alarm: The alarm object to be added.
        """
        self.registry.add('alarms', alarm_id, alarm)

    def remove_alarm(self, alarm_id):
        """
        Removes an alarm from the system by its identifier.

        Parameters:
        - alarm_id: Unique identifier of the alarm to be removed.
        """
        self.registry.remove('alarms', alarm_id)

    def get_alarm(self, alarm_id):
        """
        Retrieves an alarm by its identifier.

        Parameters:
        - alarm_id: Unique identifier of the alarm.

        Returns:
        - Alarm object associated with the given identifier.
        """
        return self.registry.get('alarms', alarm_id)

    def get_all_alarms(self):
        """
        Returns all alarms in the system.

        Returns:
        - Dictionary of alarm identifiers to alarm objects.
        """
        return self.registry.get_all('alarms')

    ### REPAIR TEAMS ###

    def add_repair_team(self, repair_team_id, repair_team):
        """
        Adds a repair team to the system.

        Parameters:
        - repair_team_id: Unique identifier for the repair team.
        - repair_team: The repair team object to be added.
        """
        self.registry.add('repair_teams', repair_team_id, repair_team)

    def remove_repair_team(self, repair_team_id):
        """
        Removes a repair team from the system by its identifier.

        Parameters:
        - repair_team_id: Unique identifier of the repair team to be removed.
        """
        self.registry.remove('repair_teams', repair_team_id)

    def get_repair_team(self, repair_team_id):
        """
        Retrieves a repair team by its identifier.

        Parameters:
        - repair_team_id: Unique identifier of the repair team.

        Returns:
        - Repair team object associated with the given identifier.
        """
        return self.registry.get('repair_teams', repair_team_id)

    def get_all_repair_teams(self):
        """
        Returns all repair teams in the system.

        Returns:
        - Dictionary of repair team identifiers to repair team objects.
        """
        return self.registry.get_all('repair_teams')
    
    
    
    def add_manual_break(self,connector:TapPipe,time,repair_time,flow):
        b=self.manual_breaks.get(time,[])
        b.append((connector,repair_time,flow))
        self.manual_breaks[time]=b
    
    def add_score_condition(self,sc):
        self.score_conditions.append(sc)

    def try_breaks(self):
        "run the various breakages"
        #handle manual break instructions
        for (connector,repair_time,flow) in self.manual_breaks.get(str(self.time),[]): #str because we are parsing json which coverts to string I think
            connector.broken(repair_time,flow)
        
        #handle probabilistic break instructions
        for c in self.get_all_connectors().values():
            c.do_probabilistic_breaks()

    def tick(self,time,breaking=True):
        """returns True if overflow happened, False otherwise. breaking is a parameter as to whether to try break things, used for different evolutions of the system."""
        self.time=time
        if breaking:
            self.try_breaks()
        
        #move liquids    
        for t in self.get_all_tanks().values():
            t.tick1()
        for c in self.get_all_connectors().values():
            c.tick2()
        for t in self.get_all_tanks().values():
            t.tick3()
            if t.overflowed:
                return True 
        
        #handle alarms
        for a in self.get_all_alarms().items():
            a[1].test_trigger(time)

        #update score        
        ds=1
        for s in self.score_conditions:
            if not s.test():
                ds=0
                break
        self.score+=ds
        
        for r in self.get_all_repair_teams().items(): #every time a repair team moves, lose 1 point if the location is not broken
            if r[1].moving and r[1].location!=None and not r[1].location.is_broken():
                self.score-=1
                r[1].moving=False
        
        #handle repair instructions
        for c in self.get_all_connectors().values():
            c.do_repairs()

        return False #returns false if sim should be disabled/stopped

    def consistent(self,s):
        """This checks whether this system is consistent with s. A system is consistent iff all observable tanks have the same level of liquid and all alarms are the same. WE IGNORE BROKEN WHEN CHECKING FOR CONSISTENCY. WE ALSO IGNORE CONNECTORS AS THEIR VALUES SHOULD BE SET BEFORE CALLING CONSISTENT"""
        print(f"time: {self.time} {s.time}")
        for t in self.get_all_tanks().keys():
            if not self.get_tank(t).observable:
                continue
            if self.get_tank(t).contains!=s.get_tank(t).contains:
                print(f"tank {t} {self.get_tank(t).contains} vs {s.get_tank(t).contains}")
                return False
        
        for c in self.get_all_connectors().keys():
            if self.get_connector(c).desired_flow!=s.get_connector(c).desired_flow:
                print(f"desired flow {c} {self.get_connector(c).desired_flow} vs {s.get_connector(c).desired_flow}")
                return False
        
        for a in self.get_all_alarms().values():
            #find which tank the alarm refers to
            my_k=None
            for k,v in self.get_all_tanks().items():
                if a.tank==v:
                    my_k=k
                    break

            for b in s.alarms:
                if b.tank==s.get_tank(my_k) and a.level==b.level and a.__class__==b.__class__ and (a.triggered!=b.triggered or a.trigger_time!=b.trigger_time):
                    print(f"alarm for tank {my_k} {a.triggered} {b.triggered} {a.trigger_time} {b.trigger_time} a level: {a.tank.contains} b level: {b.tank.contains}")
                    return False
        return True
                

    def system_to_vec(self):
        """returns a vector representation of the system. The vector is a list of the levels of the tanks as a percentage of their capacity, followed by the desired flow of the connectors and a one-hot encoding of the alarms"""
        
        
class Registry:
    """
    A central registry for managing entities within the system, including tanks,
    connectors, alarms, and repair teams. It provides a unified interface to add,
    retrieve, and remove entities based on their category and ID.

    This approach simplifies the management of entities and their visibility across
    different parts of the system, ensuring that all components have consistent and
    up-to-date access to the entities they need.

    Attributes:
        _items (dict): A dictionary that stores entities by category. Each category
                       itself is a dictionary mapping from entity IDs to entity objects.
    """

    def __init__(self):
        # Initialize the registry with empty dictionaries for each category of items.
        self._items = {'tanks': {}, 'connectors': {}, 'alarms': {}, 'repair_teams': {}}

    def add(self, category, item_id, item):
        """
        Adds an item to the registry under the specified category with the given ID.

        Parameters:
            category (str): The category of the item (e.g., 'tanks', 'connectors').
            item_id (str): The unique identifier for the item within its category.
            item (object): The item to be added to the registry.
        """
        self._items[category][item_id] = item

    def get(self, category, item_id):
        """
        Retrieves an item from the registry based on its category and ID.

        Parameters:
            category (str): The category of the item.
            item_id (str): The unique identifier for the item.

        Returns:
            object: The requested item, or None if it does not exist.
        """
        return self._items[category].get(item_id, None)

    def remove(self, category, item_id):
        """
        Removes an item from the registry based on its category and ID.

        Parameters:
            category (str): The category of the item.
            item_id (str): The unique identifier for the item.
        """
        if item_id in self._items[category]:
            del self._items[category][item_id]

    def get_all(self, category):
        """
        Retrieves all items of a specific category from the registry.

        Parameters:
            category (str): The category of items to retrieve.

        Returns:
            dict: A dictionary of all items in the specified category, mapped by their IDs.
        """
        return self._items[category]

