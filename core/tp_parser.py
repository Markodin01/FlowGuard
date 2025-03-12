from lark import Lark, Transformer
from core.model import *
from core.system import System
import redis
import jsonpickle

# Define the grammar for the TP language
TP_GRAMMAR = r"""
STRING: /([a-zA-Z_][a-zA-Z0-9_]*)/
NUMBER: /([0-9]+)/
DECIMAL: /([0-9]+(\.[0-9]*)?)/
COMMENT: /#[^\n]*/

!boolean: "True"i | "False"i

_system: _tanks _connections _alarms repairteam _scoreconditions _breaklist? mspertick?

_tanks: "TANKS"i src snk tank*
_connections: "CONNECTIONS" connection*
_alarms: "ALARMS" alarm*
_scoreconditions: "SCORING" scorecondition*

mspertick: "MillisPerTick"i NUMBER
            
src: "Source:"i STRING location?
snk: "Sink:"i STRING location?
tank: "Tank:"i STRING "Capacity:"i? NUMBER "Overflow:"i? boolean "Visible:"i? boolean location?
                 
location: "X:"i? DECIMAL? " Y:"i? DECIMAL?

_fromto: "From:"i? STRING "To:"i? STRING
connection: (pipe | valve | tap | randompipe) break_information?

pipe: "Pipe:"i STRING _fromto _flow
valve: "Valve:"i STRING _fromto _min_flow _max_flow 
tap: "Tap:"i STRING _fromto _min_flow _max_flow 
randompipe: "RandomPipe:"i STRING _fromto _min_flow _max_flow
              
_min_flow: "Min_flow:"i? NUMBER
_max_flow: "Max_flow:"i? NUMBER
_flow: "Flow:"i? NUMBER

break_information: "p_break:"i? DECIMAL  "min_repair_time:"i? NUMBER "max_repair_time:"i? NUMBER "min_break_flow:"i? NUMBER "max_break_flow:"i? NUMBER 
              
repairteam: "RepairTeams"i NUMBER

alarm: "Alarm"i "Tank:"i? STRING "Level:"i? NUMBER "Type:"i? _lowhigh
scorecondition: "ScoreCondition"i "Tank:"i? STRING "Level:"i? NUMBER "Type:"i? _lowhigh

!_lowhigh: ("Low"i| "High"i)

_breaklist: "BREAKS" manual_break*

manual_break: "Break"i STRING "Time"i? NUMBER "RepairTime"i? NUMBER "Flow"i? NUMBER
              
%import common.WS
%ignore WS
%ignore COMMENT
"""

class LanguageTransformer(Transformer):
    """
    Transformer class to convert parsed TP language into a System object.
    """

    def __init__(self):
        self.system = System()
        self.pos = {}
        self.millispertick = 1000
        super().__init__()
    
    def boolean(self, args):
        """Convert string boolean to Python boolean."""
        return args[0].type == "TRUE"

    def src(self, args):
        """Process Source definition."""
        s = Source()
        self.system.registry.add('tanks', args[0], s)
        if len(args) == 2:
            self.pos[args[0]] = args[1]
        
    def snk(self, args):
        """Process Sink definition."""
        s = Sink()
        self.system.registry.add('tanks', args[0], s)
        if len(args) == 2:
            self.pos[args[0]] = args[1]
    
    def tank(self, args):
        """Process Tank definition."""
        t = Tank(capacity=args[1], overflow=args[2], observable=True, visible=args[3], tank_id=args[0])
        self.system.add_tank(tank_id=args[0], tank=t)
        if len(args) == 5:
            self.pos[args[0]] = args[4]

    def NUMBER(self, args):
        """Convert string number to integer."""
        return int(args)
    
    def DECIMAL(self, args):
        """Convert string decimal to float."""
        return float(args)
    
    def STRING(self, args):
        """Convert token to string."""
        return str(args)
    
    def pipe(self, args):
        """Process Pipe definition."""
        return {"type": "pipe", "id": args[0], "source": args[1], "sink": args[2], "flow": args[3]}
    
    def valve(self, args):
        """Process Valve definition."""
        return {"type": "valve", "id": args[0], "source": args[1], "sink": args[2], "min_flow": args[3], "max_flow": args[4]}
    
    def tap(self, args):
        """Process Tap definition."""
        return {"type": "tap", "id": args[0], "source": args[1], "sink": args[2], "min_flow": args[3], "max_flow": args[4]}
    
    def randompipe(self, args):
        """Process RandomPipe definition."""
        return {"type": "random", "id": args[0], "source": args[1], "sink": args[2], "min_flow": args[3], "max_flow": args[4]}
    
    def location(self, args):
        """Process location coordinates."""
        return (args[0], args[1])

    def break_information(self, args):
        """Process break information for connections."""
        return {
            "p_break": args[0],
            "min_repair_time": args[1],
            "max_repair_time": args[2],
            "min_break_flow": args[3],
            "max_break_flow": args[4]
        }
    
    def connection(self, args):
        """Process connection definitions and add to the system."""
        ci = args[0]
        break_information = args[1] if len(args) > 1 else {}
        
        connector_types = {
            "pipe": Pipe,
            "valve": ValvePipe,
            "tap": TapPipe,
            "random": RandomPipe
        }
        
        ConnectorClass = connector_types.get(ci["type"])
        if ConnectorClass:
            connector = ConnectorClass(
                self.system.get_tank(ci["source"]),
                self.system.get_tank(ci["sink"]),
                ci.get("min_flow", ci.get("flow")),
                ci.get("max_flow", ci.get("flow")),
                **break_information
            )
            self.system.add_connector(ci["id"], connector)

    def scorecondition(self, args):
        """Process score condition and add to the system."""
        condition_class = LowScoreCondition if args[2].type == "LOW" else HighScoreCondition
        self.system.add_score_condition(condition_class(self.system.get_tank(args[0]), args[1]))
   
    def alarm(self, args):
        """Process alarm definition and add to the system."""
        alarm_types = {
            "LOW": LowAlarm,
            "HIGH": HighAlarm
        }
        alarm_type = args[2].upper()
        AlarmClass = alarm_types.get(alarm_type)
        if AlarmClass:
            new_alarm = AlarmClass(self.system.get_tank(args[0]), args[1])
            self.system.add_alarm(f"{alarm_type}{args[0]}", new_alarm)
        else:
            print(f"Unknown alarm type: {alarm_type}")

    def repairteam(self, args):
        """Process repair team definition and add to the system."""
        for i in range(args[0]):
            self.system.add_repair_team(i+1, RepairTeam())
    
    def manual_break(self, args):
        """Process manual break definition and add to the system."""
        self.system.add_manual_break(self.system.get_connector(args[0]), args[1], args[2], args[3])
    
    def mspertick(self, args):
        """Set milliseconds per tick for the simulation."""
        self.millispertick = args[0]

def publish_system_state(system, pos, mspertick):
    """
    Publish the system state to Redis.

    Args:
        system (System): The system object to publish.
        pos (dict): Dictionary of positions.
        mspertick (int): Milliseconds per tick.
    """
    r = redis.StrictRedis(host='localhost', port=6379)
    
    # Serialize the data
    message = jsonpickle.encode({
        "system": jsonpickle.encode(system),
        "positions": jsonpickle.encode(pos),
        "mspertick": jsonpickle.encode(mspertick)
    })

    # Publish the message to Redis
    r.publish('system_updates', message)

def parse(filename):
    """
    Parse a TP language file and return the system, positions, and milliseconds per tick.

    Args:
        filename (str): Path to the TP language file.

    Returns:
        tuple: (System object, positions dictionary, milliseconds per tick)
    """
    with open(filename, "r") as f:
        parser = Lark(TP_GRAMMAR, start='_system')
        tree = parser.parse(f.read())
        transformer = LanguageTransformer()
        transformer.transform(tree)
        
        # Publish the system state to Redis
        publish_system_state(transformer.system, transformer.pos, transformer.millispertick)
        
        return transformer.system, transformer.pos, transformer.millispertick

# Example usage
if __name__ == "__main__":
    system, positions, ms_per_tick = parse("path/to/your/tp_file.tnp")
    print(f"Parsed system with {len(system.get_all_tanks())} tanks and {ms_per_tick} ms/tick")