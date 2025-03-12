import random

class ScoreCondition:
    def __init__(self,tank,level):
        self.tank=tank
        self.level=level
    
    def test(self):
        pass

class LowScoreCondition(ScoreCondition):
    def test(self):
        if self.tank.contains<self.level:
            return True
        return False

class HighScoreCondition(ScoreCondition):
    def test(self):
        if self.tank.contains>self.level:
            return True
        return False

class Alarm:
    def __init__(self,tank,level, color):
        self.tank=tank
        self.level=level
        self.triggered=False
        self.trigger_time=0
        self.color=color
    
    def reset(self):
        self.untrigger()

    def trigger(self,time):
        self.triggered=True
        if self.trigger_time==0:  #done to store first trigger time of alarm
            self.trigger_time=time
    
    def untrigger(self):
        self.triggered=False
        self.trigger_time=0
    
    def test_trigger(self,time):
        pass

class LowAlarm(Alarm):
    
    def __init__(self, tank, level):
        super().__init__(tank, level, "#fff888")
        
    def test_trigger(self,time):
        if self.tank.contains<self.level:
            self.trigger(time)
        else:
            self.untrigger()

class HighAlarm(Alarm):
    
    def __init__(self, tank, level):
        super().__init__(tank, level, "#ff8888")
    
    def test_trigger(self,time):
        if self.tank.contains>self.level:
            self.trigger(time)
        else:
            self.untrigger()

################################################################

class TapPipe:
    def __init__(self,source,sink,min_flow,max_flow,**kwargs):
        self.source=source
        self.sink=sink
        self.source.outputs.add(self)
        self.sink.inputs.add(self)
        self.min_flow=min_flow
        self.max_flow=max_flow
        self._current_flow=min_flow
        self._desired_flow=min_flow
        self.out_content=0
        self.in_content=0

        self._broken=False
        self.p_break=kwargs.get("p_break",0)
        self.min_repair_time=kwargs.get("min_repair_time",0)
        self.max_repair_time=kwargs.get("max_repair_time",0)
        self.min_break_flow=kwargs.get("min_break_flow",0)
        self.max_break_flow=kwargs.get("max_break_flow",0)
        self.breakage_timer = 0

        self.repairTeam=None
        self.repair_time=0
        
    def __str__(self) -> str:
        return self.__class__.__name__
    
    def reset(self):
        self._current_flow=self.min_flow
        self._desired_flow=self.min_flow
        self.in_content=0
        self.out_content=0
        self._broken=False
        self.repairTeam=None
        self.repair_time=0
    
    
    def is_broken(self):
        return self._broken
    
    def do_probabilistic_breaks(self):
        if not self._broken and random.random()<=self.p_break:
            self.broken(random.randint(self.min_repair_time,
                                       self.max_repair_time),
                        random.randint(self.min_break_flow,
                                       self.max_break_flow))
    
    def broken(self,repair_time,current_flow):
        self._broken=True
        self._current_flow=current_flow
        self.repair_time=repair_time
    

    @property
    def desired_flow(self):
        return self._desired_flow
    
    @desired_flow.setter
    def desired_flow(self,value):
        self._desired_flow=value
        #if value>self.max_flow or  value<self.min_flow or self._broken:
        #    return
        #else:
        #    self._current_flow=self._desired_flow

    def do_repairs(self):
        if self.repairTeam!=None and self.repair_time>0:
            self.repair_time-=1
        if self.repairTeam!=None and self.repair_time==0 and self._broken:
            self._broken=False
            self.repairTeam.location=None
            self.repairTeam=None
        if self.repairTeam!=None and not self._broken:
            self.repairTeam.location=None
            self.repairTeam=None

    def tick2(self):
        if not self._broken:
            self._current_flow=self._desired_flow
        #step 2 of tick, move from in_content to out_content
        out_capacity=self.max_flow-self.out_content #the amount of space in the out_content bin
        
        amount_transferable=self._current_flow if self._current_flow<=self.in_content else self.in_content #the amount we can move from in to out

        if amount_transferable>=out_capacity:
            self.out_content+=out_capacity
            self.in_content-=out_capacity
        else:
            self.out_content+=amount_transferable
            self.in_content-=amount_transferable
            
    def __str__(self) -> str:
        return str(self.source)

class ValvePipe(TapPipe):
    pass
    

class Pipe(TapPipe):

    @property
    def desired_flow(self):
        return super().desired_flow
    
    @desired_flow.setter
    def desired_flow(self,_):
        return

class RandomPipe(TapPipe):
    """This type of pipe adds a random amount of liquid every tick"""
    def tick2(self):
        if not self._broken:
            self._current_flow=random.randint(self.min_flow,self.max_flow)
        super.tick2()

################################################################

class RepairTeam:
    def __init__(self):
        self.location=None
        self.moving=False #used to penalise for moving repair team
    
    def reset(self):
        self.location=None

    def dispatch(self,location):
        self.location=location
        if location!=None:
            self.location.repairTeam=self
            self.moving=True

################################################################
class Tank:
    def __init__(self,capacity,overflow=False,observable=True, visible=True, tank_id='S'):
        self.capacity=capacity
        self.inputs=set()
        self.outputs=set()
        self.contains=0
        self.allow_overflow=overflow
        self.overflowed=False
        self.observable=observable
        self.visible=visible
        self.tank_id=tank_id
        self.available_alarms = {}
        self.triggered_alarm = None
        
    def __str__(self) -> str:
        return self.__class__.__name__
    
    def reset(self):
        self.contains=0
        self.overflowed=False

    def tick1(self):
        #step 1 of tick is pipe filling. Note, if we don't have enough liquid than a random element of the set is filled. TODO: do actual randomisation
        for p in self.outputs:
            amount_drained = p.max_flow-p.in_content if p.max_flow-p.in_content<=self.contains else self.contains

            self.contains-=amount_drained
            p.in_content+=amount_drained
    
    def tick3(self):
        for p in self.inputs:
            if p.out_content<=self.capacity-self.contains:
                self.contains+=p.out_content
                p.out_content=0
            elif self.allow_overflow==False:
                p.out_content-=self.capacity-self.contains
                self.contains=self.capacity
            else:
                p.out_content=0
                self.contains=self.capacity
                self.overflowed=True

class Source(Tank):
    def __init__(self):
        super().__init__(0,0,False,True,"SR")

    def tick1(self):
        for p in self.outputs:
            p.in_content=p.max_flow

class Sink(Tank):
    def __init__(self):
        super().__init__(0,0,False,True,"SN")

    def tick3(self):
        for p in self.inputs:
            p.out_content=0

################################################################