import random

class LevelTrainer():
    def __init__(self,popSize,noOfLvls):

        self.pSize = popSize
        self.lQty = noOfLvls

        self.levels = {}
       
        individualsPerLevel = round((popSize/noOfLvls),0)
        for level in range(0,noOfLvls):
            for individual in range(0,int(individualsPerLevel)):
                failFlag = True
                while failFlag == True:
                    geneome = Level()
                    geneome.generateLevel()
                    failFlag = geneome.testLevel()
                
                geneome.layoutObjects()
                levelfnm = str(level) + "-" + str(individual) + ".tnp"
                #print(levelfnm)
                geneome.exportFile(levelfnm)
                self.levels[level] = {"individual":individual,"genes":geneome}


class Level():
    def __init__(self):
        self.tanks = {}
        self.valves = {}
        self.pipes = {}
        self.taps = {}
        self.breaks = {}
        self.alarms = []
        self.scoreConds = []

    def testLevel(self):
        #print("newLevelTest")
        failFlg = False
        while failFlg == False:
            for rt in self.routeMap:
                #print("newRoute")
                qTanks = len(rt)
                cnt = 0
                #print("qTanks",rt)

                for tnk in rt:
                    pntr = int(tnk[1])
                    #print("newTank",tnk)
                    dptr = pntr +1

                    dmax = self.getMaxFlow(self.tanks[tnk].outlet)
                    dmin = self.getMinFlow(self.tanks[tnk].outlet)
                    
                    if ((qTanks >1) and ((cnt+1) <qTanks)) :
                        for dstnk in range(cnt,qTanks-1):
                            dtnkTag = "T" + str((dptr))
                            #print("dtnktag",dtnkTag)
                            if self.getMinFlow(self.tanks[dtnkTag].outlet) < dmin:
                                dmin = self.getMinFlow(self.tanks[dtnkTag].outlet)
                            if self.getMaxFlow(self.tanks[dtnkTag].outlet) > dmax:
                                dmax = self.getMaxFlow(self.tanks[dtnkTag].outlet)
                            dptr = dptr + 1

                    umax = self.getMaxFlow(self.tanks[tnk].inlet)
                    umin = self.getMinFlow(self.tanks[tnk].inlet)
                    
                    uptr = pntr - 1
                    if  ((qTanks >1) and ((cnt) >0)) :
                        for ustnk in range(1,(cnt+1)):
                            utnkTag = "T" + str((uptr))
                            #print("utnTnk",utnkTag)
                            if self.getMinFlow(self.tanks[utnkTag].inlet) < umin:
                                umin = self.getMinFlow(self.tanks[utnkTag].inlet)
                            if self.getMaxFlow(self.tanks[utnkTag].outlet) > umax:
                                umax = self.getMaxFlow(self.tanks[utnkTag].inlet)
                            uptr = uptr - 1
                        
                    #print("Umin",umin)
                    #print("Umax",umax)
                    #print("Dmin",dmin)
                    #print("Dmax",dmax)

                    if not (umax > dmin):
                        failFlg = True
                        #print("----Tank Unfillable-------------------------------------------------------------------------------")
                    if not (dmax > umin):
                        failFlg = True
                        #print("----Tank Unemptyable-------------------------------------------------------------------------------------------")
                    cnt = cnt + 1
            break
        #print("Fail=",failFlg)
        return failFlg
                
        
    def generateLevel(self):
        self.qtyTanks = random.randint(1,5)
        for tank in range(0,self.qtyTanks):
            tag = "T" + str((tank+1))
            newTank = Tank()
            self.tanks[tag] = newTank
            if newTank.alarmL:
                string = "Alarm "+ tag + " "+ str(int(newTank.alarmL)) + " Low"
                self.alarms.append(string)
                
            if newTank.alarmH:
                string = "Alarm " +tag + " "+ str(int(newTank.alarmH)) + " High"
                self.alarms.append(string)

            if newTank.lScoreCond:
                string = "ScoreCondition " + tag + " "+ str(int(newTank.lScoreCond)) + " " + "Low"
                self.scoreConds.append(string)

            if newTank.hScoreCond:
                string = "ScoreCondition " + tag + " "+ str(int(newTank.hScoreCond)) + " " + "High"
                self.scoreConds.append(string)

        tanksRemaining = self.qtyTanks
        #print(self.qtyTanks)

        openRoutes = 1
        pipeCnt = 1
        valveCnt = 1
        tapCnt = 1
        srcTag = "SR"
        rows=[]
        qtyRows = 0
        row = []
        qtyRow = 0
        psrcTag = "SR"
        
        while openRoutes > 0:
            if ((srcTag == "SR") or((srcTag[0] =="T") and (srcTag[1] !="P")) ):
                nxtType = 1
            else:
                nxtType = 2
                
            if nxtType == 1:
                pick = random.choice(["pipe","valve","tap"])
                if pick == "pipe":
                    tag = "P" + str(pipeCnt)
                    tmpPipe = Pipe(srcTag,tag)
                    self.pipes[tmpPipe.tag] = tmpPipe
                    tag = tmpPipe.tag
                    self.terminate(srcTag,tag)
                    pipeCnt = pipeCnt + 1
                elif pick == "valve":
                    tag = "VL" + str(valveCnt)
                    self.terminate(srcTag,tag)
                    self.valves[tag] = Valve(srcTag,tag)
                    valveCnt = valveCnt + 1
                else:
                    tag = "TP"+str(tapCnt)
                    self.terminate(srcTag,tag)
                    self.taps[tag] = Tap(srcTag,tag)
                    tapCnt = tapCnt + 1
                if srcTag != "SR":
                    self.tanks[srcTag].oulet = srcTag
                psrcTag = srcTag
                srcTag = tag
                
            elif nxtType  == 2:
                if tanksRemaining == 0:
                    pick = "sink"
                elif psrcTag == "SR":
                    pick = "tank"                    
                else:
                    pick = random.choice(["tank","sink"])
                    
                if pick == "sink":
                    self.terminate(srcTag,"SN")
                    rows.append(row)
                    row = []
                    if tanksRemaining == 0:
                        openRoutes = 0
                    else:
                        psrcTag = srcTag
                        srcTag = "SR"  
                else:
                    tag = "T" + str((self.qtyTanks-tanksRemaining+ 1))
                    #print(tag)
                    self.terminate(srcTag,tag)
                    self.tanks[tag].inlet = srcTag
                    tanksRemaining = tanksRemaining - 1
                    psrcTag = srcTag
                    srcTag = tag
                    row.append(tag)
        self.routeMap = rows
#        print(self.routeMap)
        
        self.repairTeams = random.randint(1,3)
        
        self.noOfBreaks = random.randint(1,3)

        for breaks in range(0,self.noOfBreaks):
            choice = random.choice(["PP","RP","TP","VL"])
            if choice == "PP" or choice == "RP":
                if len(self.pipes.keys()) > 0:
                    breakTag = random.choice(list(self.pipes.keys()))
                    self.breaks[breakTag] = Break(self.pipes[breakTag].flow)
                else:
                    choice = "TP"
            
            if choice == "TP":
                if len(self.taps.keys()) > 0:
                    keyList = self.taps.keys()
                    breakTag = random.choice(list(keyList))
                    self.breaks[breakTag] = Break(self.taps[breakTag].maxF)
                else:
                    choice = "VL"                                             

            if choice == "VL":
                if len(self.valves.keys()) > 0:
                    keyList = self.valves.keys()
                    breakTag = random.choice(list(keyList))
                    self.breaks[breakTag] = Break(self.valves[breakTag].maxF)
                else:
                    choice = "VL"  

        self.millisPerTick = 1000
            

    def terminate(self,tag1,snkTag):
        if ((tag1[0] == "R") or (tag1[0] == "P")):
            self.pipes[tag1].outlet = snkTag
        elif tag1[0] == "T":
            if tag1[1] =="P":
                self.taps[tag1].outlet = snkTag
            else:
                self.tanks[tag1].outlet = snkTag
        elif tag1[0] == "V": 
            self.valves[tag1].outlet = snkTag


    def getMaxFlow(self,tag1):
        if ((tag1[0] == "R") or (tag1[0] == "P")):
            return self.pipes[tag1].flow 
        elif tag1[0] == "T":
            return self.taps[tag1].maxF
        elif tag1[0] == "V": 
            return self.valves[tag1].maxF

    def getMinFlow(self,tag1):
        if ((tag1[0] == "R") or (tag1[0] == "P")):
            #print(self.pipes.keys())
            return self.pipes[tag1].flow 
        elif tag1[0] == "T":
            return self.taps[tag1].minF
        elif tag1[0] == "V": 
            return self.valves[tag1].minF        
            
    def layoutObjects(self):
        xP = 0
        yP = 0
        #print(len(self.routeMap))
        #print(len(self.routeMap))
        yspace = 1 / (len(self.routeMap) + 1)
        for route in self.routeMap:
            yP = yspace + yP
            #print(len(route))
            xP = 0
            xspace = 1 / (len(route) + 1)
            for tank in route:
                xP = xP + xspace
                self.tanks[tank].xloc = xP
                self.tanks[tank].yloc = yP
                
    def exportFile(self,fnm):
        with open (fnm,'w') as fle:
            fle.write("TANKS")
            fle.write("\n")
            fle.write("Source: SR 0 0.5")
            fle.write("\n")
            fle.write("Sink: SN 1 0.5")
            fle.write("\n")
            
            for key in self.tanks.keys():
                fle.write("Tank: ")
                fle.write(key)
                fle.write(" ")
                fle.write(str(self.tanks[key].size))
                fle.write(" ")
                fle.write(str(self.tanks[key].oFlow))                
                fle.write(" ")
                fle.write(str(self.tanks[key].visible))
                fle.write(" ")
                fle.write(str(self.tanks[key].xloc))
                fle.write(" ")
                fle.write(str(self.tanks[key].yloc))
                fle.write("\n")
                
            fle.write("\n")
            fle.write('CONNECTIONS')
            fle.write("\n")
            
            for key in self.pipes.keys():
                fle.write("Pipe: ")
                fle.write(key)
                fle.write(" ")
                fle.write(self.pipes[key].inlet)
                fle.write(" ")
                fle.write(self.pipes[key].outlet)                
                fle.write(" ")
                fle.write(str(self.pipes[key].flow))
                fle.write("\n")
                
            for key in self.valves.keys():
                fle.write("Valve: ")
                fle.write(key)
                fle.write(" ")
                fle.write(self.valves[key].inlet)
                fle.write(" ")
                fle.write(self.valves[key].outlet)                
                fle.write(" ")
                fle.write(str(self.valves[key].minF))
                fle.write(" ")
                fle.write(str(self.valves[key].maxF))
                fle.write("\n")
                
            for key in self.taps.keys():
                fle.write("Tap: ")
                fle.write(key)
                fle.write(" ")
                fle.write(self.taps[key].inlet)
                fle.write(" ")
                fle.write(self.taps[key].outlet)                
                fle.write(" ")
                fle.write(str(self.taps[key].minF))
                fle.write(" ")
                fle.write(str(self.taps[key].maxF))
                fle.write("\n")
                
            fle.write("\n")
            fle.write('ALARMS')
            fle.write("\n")
            #print(self.alarms)
            for item in self.alarms:
                fle.write(item)
                fle.write("\n")
                
                
            fle.write("\n")
            fle.write("RepairTeams ")
            fle.write(str(self.repairTeams))
            fle.write("\n\n")

            fle.write("SCORING\n")
            for item in self.scoreConds:
                fle.write(item)
                fle.write("\n")
#################
                                   
            fle.write("\n")
                                   
            fle.write("\nBREAKS\n")

            for key in self.breaks.keys():
                fle.write("Break ")
                fle.write(key)
                fle.write(" ")
                fle.write(str(self.breaks[key].probability))
                fle.write(" ")
                fle.write(str(self.breaks[key].maxF))                
                fle.write(" ")
                fle.write(str(self.breaks[key].faultFlow) )               
                fle.write("\n")

            fle.write("\n\n")

            fle.write("MillisPerTick ")
            fle.write(str(self.millisPerTick))
                                   
            fle.close()
            
    
            
class Tank():
    def __init__(self):
        self.size = random.randint(1,100)
        self.oFlow = bool(random.getrandbits(1))
        self.visible = bool(random.getrandbits(1))
        self.alarmH = False
        self.alarmL = False
        self.lScoreCond = False
        self.hScoreCond = False
        self.inlet = False
        self.outlet = False
        if bool(random.getrandbits(1)):
            self.alarmH = random.uniform((0.6*self.size),self.size)
        if bool(random.getrandbits(1)):
            self.alarmL = random.uniform(0,(0.4*self.size))
        if bool(random.getrandbits(1)):
            if (self.alarmL and self.alarmH):
                print("A")
                self.lScoreCond = random.uniform(self.alarmL,self.alarmH)
            elif self.alarmL:
                print("B")
                self.lScoreCond = random.uniform(self.alarmL,self.size)
            elif self.alarmL:
                print("C")
                self.lScoreCond = random.uniform(0,self.alarmH)
            else:
                print("D")
                self.lScoreCond = random.uniform(0,self.size)
        if bool(random.getrandbits(1)):
            if self.lScoreCond:
                if self.alarmH:
                    print("E")
                    self.hScoreCond = random.uniform(self.lScoreCond,self.alarmH)
                else:
                    print("F")
                    self.hScoreCond = random.uniform(self.lScoreCond,self.size)
            else:
                if ((self.alarmL) and (self.alarmH)):
                    print("G")
                    self.hScoreCond = random.uniform(self.alarmL,self.alarmH)
                elif self.alarmL:
                    print("H")
                    self.hScoreCond = random.uniform(self.alarmL,self.size)
                elif self.alarmL:
                    print("I")
                    self.hScoreCond = random.uniform(0,self.alarmH)
                else:
                    print("J")
                    self.hScoreCond = random.uniform(0,self.size)
                
class Valve():
    def __init__(self,src,tg):
        self.maxF = random.randint(1,5)
        self.minF = random.randint(0,(self.maxF-1))

        self.inlet = src
  
class Pipe():
    def __init__(self,src,tg):
        self.flow = random.randint(1,5)
        self.inlet = src
        prefix = random.choice(["R","P"])
        self.tag = prefix + tg

class Tap():
    def __init__(self,src,tg):
        self.maxF = random.randint(1,5)
        self.minF = random.randint(0,(self.maxF-1))
        self.inlet = src
        
class Break():
    def __init__(self,maxFlow):
        self.probability = random.randint(1,5)
        self.maxF = maxFlow
        self.faultFlow = random.randint(0,5)

trainer = LevelTrainer(100,10)
