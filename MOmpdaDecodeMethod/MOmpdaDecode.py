import readcfg as r_d
from MOmpdaInstance import MO_MPDAInstance
import os, sys
from MOmpdaDecodeMethod.mpdaRobot import RobotState,Robot
from MOmpdaDecodeMethod.mpdaTask import  Task
from MOmpdaDecodeMethod.mpdaDecoderActSeq import ActionSeq,ActionTuple,EventType
import numpy as np
from enum import Enum
from collections import  namedtuple

RobTaskPair = namedtuple('RobTaskPair',['robID','taskID'])
import math

class CalType(Enum):
    arriveCond = 1
    leaveCond = 2
    endCond = 3
    backCond = 4
    stateInvalidCond = 5

def generateRandEncode(robNum,taskNum):
    encode = np.zeros((robNum, taskNum),dtype =int)
    for i in range(robNum):
        permLst = [x for x in range(taskNum)]
        np.random.shuffle(permLst)
        encode[i][:] = permLst
    return encode




AbsolutePath = os.path.abspath(__file__)
# 将相对路径转换成绝对路径
SuperiorCatalogue = os.path.dirname(AbsolutePath)
# 相对路径的上级路径
BaseDir = os.path.dirname(SuperiorCatalogue)

degBoolean = False




class MO_MPDADecoder(object):
    def __init__(self, mo_ins :MO_MPDAInstance):
        self._insName = mo_ins._insName
        # self._robNum = mo_ins._robNum
        self._taskNum = mo_ins._taskNum
        self._threhold = mo_ins._threhold
        self._robAbi = mo_ins._robAbi
        self._robVel = mo_ins._robVel
        self._taskStateLst = mo_ins._taskStateLst
        self._taskRateLst = mo_ins._taskRateLst
        self._taskDisMat = mo_ins._taskDisMat

        # self._rob2taskDisMat = ins._rob2taskDisMat
        # self._taskDisMat = ins._taskDisMat
        if degBoolean:
            self._degFile = open(BaseDir+ '/debugData/deg.dat', 'w')

    def decode(self, robNum, x):
        self.encode = x
        self._robNum = robNum
        self._actSeq = ActionSeq()
        self.initStates()
        # if self.decodeProcessor():
        validStateBoolean = self.decodeProcessor()
        if degBoolean:
            self._degFile.write(str(self.cmpltLst))
        # print(self._actSeq.convert2MultiPerm(self._robNum))
        return validStateBoolean, self._actSeq
    def initStates(self):
        '''
        initialize states of decode method
        '''
        self.taskLst = []
        self.robotLst = []
        self.cmpltLst = [False] * self._taskNum
        for i in range(self._robNum):
            rob = Robot()
            rob._ability = self._robAbi
            rob._vel = self._robVel
            rob.encodeIndex = 0
            rob.taskID = self.encode[i][0]
            stopBool = False
            # represent the robot at the depot
            dis = self._taskDisMat[rob.taskID][self._taskNum]
            dis_time = dis / rob._vel
            rob.arriveTime = dis_time
            rob.stopBool = stopBool
            rob.stateType = RobotState['onRoad']
            rob.leaveTime = 0
            self.robotLst.append(rob)

        for i in range(self._taskNum):
            task = Task()
            task.cState = self._taskStateLst[i]
            task._initState = self._taskStateLst[i]
            task.cRate = self._taskRateLst[i]
            task._initRate = self._taskRateLst[i]
            task._threhod = self._threhold
            task.cmpltTime = sys.float_info.max
            self.taskLst.append(task)

        # self.decodeTime = 0
        # self.validStateBool = True
    def decodeProcessor(self):
        while not self.allTaskCmplt():
            cal_type, actionID = self.findActionID()
            if cal_type  == CalType['arriveCond']:
                rob = self.robotLst[actionID]
                arriveTime = rob.arriveTime
                encodeInd = rob.encodeIndex
                taskID = self.encode[actionID][encodeInd]
                # if degBoolean:
                self._actSeq.append(ActionTuple(robID =actionID,taskID= taskID, eventType = EventType.arrive,eventTime = arriveTime))

                if self.cmpltLst[taskID]:
                    # =============================================================================
                    #  the task has been cmplt
                    # =============================================================================
                    rob = self.robotLst[actionID]
                    rob.leaveTime = rob.arriveTime
                    rob.taskID = taskID
                    rob.stateType = RobotState['onTask']
                    # if degBoolean:
                    self._actSeq._arrCmpltTaskLst.append((actionID, taskID))
                else:
# =============================================================================
# the task has not been cmplt
# =============================================================================
                    task = self.taskLst[taskID]
                    rob.taskID = taskID
                    validStateBool = task.calCurrentState(arriveTime)
                    if not validStateBool:
                        break
                    task.cRate = task.cRate - rob._ability
# can not be cmplted
                    if task.cRate >= 0:
                        leaveTime = sys.float_info.max
# can be completed
                    else:
                        rob.executeDur = task.calExecuteDur()
                        rob.executeBool = False
                        leaveTime = rob.arriveTime + rob.executeDur
                        coordLst = self.findCoordRobot(actionID)
                        for coordID in coordLst:
                            coordRob = self.robotLst[coordID]
                            coordRob.leaveTime = leaveTime
                            coordRob.executeDur = coordRob.leaveTime - coordRob.arriveTime
                    rob.leaveTime = leaveTime
                    rob.stateType = RobotState['onTask']


            if cal_type == CalType['leaveCond']:
                rob = self.robotLst[actionID]
                taskID = rob.taskID
                task = self.taskLst[taskID]
                preTaskID = taskID
                self.cmpltLst[taskID] = True
                # if degBoolean:
                self._actSeq.append(ActionTuple(robID =actionID,taskID= taskID, eventType = EventType.leave,eventTime = rob.leaveTime))
                task.cmpltTime = rob.leaveTime

                coordLst = self.findCoordRobot(actionID)
                for coordID in coordLst:
                    self.updateRobLeaveCond(robID = coordID)
                    # if degBoolean:
                    self._actSeq.append(ActionTuple(robID = coordID, taskID= taskID,eventType = EventType.leave,eventTime = task.cmpltTime))
                self.updateRobLeaveCond(robID = actionID)

                '''
                debug is here
                '''

                if degBoolean:
                    self._degFile.write(str(taskID) + ' have been completed\n')

            if cal_type == CalType.endCond:
                # print('stuck')
                # invalidFitness = True
                validStateBool = False
                break

        if not validStateBool:
            pass
            # print('the state is explosion')
        return  validStateBool
    '''
    some fucntions
    '''

    def allTaskCmplt(self):
        if False in self.cmpltLst:
            return False
        else:
            return True

    def findActionID(self):
        cal_type = CalType['endCond']
        actionID = sys.float_info.max
        minTime = sys.float_info.max
        for i in range(self._robNum):
            rob = self.robotLst[i]
            if rob.stopBool != True:
                if rob.stateType == RobotState['onRoad']:
                    if rob.arriveTime < minTime:
                        minTime = rob.arriveTime
                        cal_type = CalType['arriveCond']
                        actionID = i
                if rob.stateType == RobotState['onTask']:
                    if rob.leaveTime < minTime:
                        minTime = rob.leaveTime
                        cal_type = CalType['leaveCond']
                        actionID = i
        if degBoolean:
            self.saveRobotInfo(degFile= self._degFile)
            self._degFile.write(str(actionID) + ' time = '+ str(minTime)
                              + ' type = ' + str(cal_type) + '\n')
        # self.saveEventInMemory()
        # if minTime < self.decodeTime:
        #     cal_type = CalType['backCond']
        #            print(minTime)
        #            print(self.decodeTime)
        #            taskID = self.robotLst[actionI].taskID
        #        self.saveRobotInfo()

        return cal_type, actionID

    def findCoordRobot(self, robID):
        '''
        find robots which are corrdinated with the robot A
        '''
        coordLst = []
        rob = self.robotLst[robID]
        taskID = rob.taskID
        for i in range(self._robNum):
            if i == robID:
                continue
            #            crob = self.robotLst[i]
            if self.robotLst[i].stateType == RobotState['onRoad']:
                continue
            if self.robotLst[i].stopBool == True:
                continue
            if self.robotLst[i].taskID == taskID:
                coordLst.append(i)
        return coordLst

    def updateRobLeaveCond(self,robID):
        rob = self.robotLst[robID]
        preTaskID = rob.taskID
        while True:
            if rob.encodeIndex == (len(self.encode[robID]) - 1):
                rob.stopBool = True
                break
            rob.encodeIndex += 1
            taskID = self.encode[robID][rob.encodeIndex]
            # print(taskID)
            if self.cmpltLst[taskID]:
                continue
            else:
                roadDur = self.calRoadDur(preTaskID, taskID, robID)
                arriveTime = rob.leaveTime + roadDur
                if arriveTime > self.taskLst[taskID].cmpltTime:
                    continue
                rob.roadDur = roadDur
                rob.taskID = taskID
                rob.arriveTime = rob.leaveTime + rob.roadDur
                rob.stateType = RobotState['onRoad']
                break

    def getRobTask(self, robID=0, encodeIndex=0):
            '''
            get the robot next task ID
            '''
            stopBool = False
            while True:
                if encodeIndex == self._taskNum:
                    stopBool = True
                    break
                taskID = self.encode[robID][encodeIndex]
                if taskID < 0:
                    encodeIndex += 1
                    continue
                else:
                    break
            return taskID, encodeIndex, stopBool

    def calRoadDur(self,taskID1,taskID2,robID):
        '''
        calculate the time fragment from the time when robID leaves the taskID1 to
        the time when rob arrives the taskID2
        '''
        dis = self._taskDisMat[taskID1][taskID2]
        rob = self.robotLst[robID]
        roadDur = dis/rob._vel
        return roadDur
    def calMakespan(self):
        if False in self.cmpltLst:
            return None
        cmpltTime = []
        for task in self.taskLst:
            cmpltTime.append(task.cmpltTime)
        return max(cmpltTime)
    def saveRobotInfo(self,degFile):
        '''
        save robot information into the deg files
        '''
        deg = degFile
        deg.write('\n')
        for i in range(self._robNum):
            lst = []
            lst.append(i)
            lst.append('arriveTime')
            lst.append(self.robotLst[i].arriveTime)
            lst.append('leaveTime')
            lst.append(self.robotLst[i].leaveTime)
            lst.append('state')
            lst.append(self.robotLst[i].stateType)
            lst.append('taskID')
            lst.append(self.robotLst[i].taskID)
            str_lst  = [str(x) for x in lst]
            robInfo = '  '
            robInfo = robInfo.join(str_lst)
            deg.write(robInfo+'\n')
        deg.write('\n')
        deg.flush()



if __name__ == '__main__':
    print('test_mo_mpdaDecoder')
    print(BaseDir)
    ins = MO_MPDAInstance()
    insFileName = BaseDir + '//MOmpdaStaticBenchmarkSet//10_C_RCL_(2-5)_(5-8).txt'
    ins.loadCfg(fileName =  insFileName)

    degBoolean = True
    mo_decoder = MO_MPDADecoder(ins)
    import random
    random.seed(2)
    np.random.seed(2)
    for i in range(20):
        robNum = random.randint(1,20)
        x = generateRandEncode(robNum = robNum, taskNum = ins._taskNum)
        import time
        start_time = time.time()
        print('robNum = ', robNum)
        validStateBoolean,actSeq = mo_decoder.decode(x = x,robNum = robNum)
        print('simulator_time  =', mo_decoder.calMakespan())
        end_time = time.time()
        print('time = ',end_time - start_time)

    print('first decoder is over')
    exit()
    # np.random.seed(1)

