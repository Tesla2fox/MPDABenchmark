import readcfg as r_d
from mpdaInstance import MPDAInstance
import os, sys
from mpdaDecodeMethod.mpdaRobot import RobotState,Robot
from mpdaDecodeMethod.mpdaTask import  Task
from mpdaDecodeMethod.mpdaDecoderActSeq import ActionSeq,ActionTuple,EventType,MPDADecoderActionSeq
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
    updateCond = 6

AbsolutePath = os.path.abspath(__file__)
# 将相对路径转换成绝对路径
SuperiorCatalogue = os.path.dirname(AbsolutePath)
# 相对路径的上级路径
BaseDir = os.path.dirname(SuperiorCatalogue)

degBoolean = False

class ACO_Updater(object):
    def __init__(self,ins :MPDAInstance):
        self._insName = ins._insName
        # readCfg = rd.Read_Cfg(fileName)
        self._robNum = ins._robNum
        self._taskNum = ins._taskNum
            # int(readCfg.getSingleVal('taskNum'))
        self._threhold = ins._threhold
        self._robAbiLst  = ins._robAbiLst
        self._robVelLst = ins._robVelLst
        self._taskStateLst = ins._taskStateLst
        self._taskRateLst = ins._taskRateLst
        self._rob2taskDisMat = ins._rob2taskDisMat
        self._taskDisMat = ins._taskDisMat
        if degBoolean:
            self._degFile = open(BaseDir+ '/debugData/deg.dat', 'w')
        self.encode = [[] for x in range(self._robNum)]
    def addInitVisitLst(self,robInitVisitLst : list):
        for robID in range(self._robNum):
            self.encode[robID].append(robInitVisitLst[robID])
        # print(self.encode)
        self.initStates()
    def updateState(self):

        cmpltBool,updateRobLst = self.decodeProcessor()
        # stateBool,updateRobLst = self.decodeProcessor()
        return cmpltBool,updateRobLst
    def updateEncode(self,robTaskPairLst : list):
        for rob_task_pair in robTaskPairLst:
            self.encode[rob_task_pair.robID].append(rob_task_pair.taskID)
            self.updateRobLeaveCond(robID= rob_task_pair.robID)

    def initStates(self):
        '''
        initialize states of decode method
        '''
        self.taskLst = []
        self.robotLst = []
        self.cmpltLst = [False] * self._taskNum
        self._arrCmpltTaskLst = []
        for i in range(self._robNum):
            rob = Robot()
            rob._ability = self._robAbiLst[i]
            rob._vel = self._robVelLst[i]
            rob.encodeIndex = 0
            rob.taskID, rob.encodeIndex, stopBool = self.getRobTask(robID=i, encodeIndex=0)
            if not stopBool:
                dis = self._rob2taskDisMat[i][rob.taskID]
                dis_time = dis / rob._vel
                rob.arriveTime = dis_time
            rob.stopBool = stopBool
            rob.stateType = RobotState['onRoad']
            rob.leaveTime = 0
            self.robotLst.append(rob)
            # print('robID task ID = ', rob.taskID)
            # print('rob arriveTime = ',rob.arriveTime)
        for i in range(self._taskNum):
            task = Task()
            task.cState = self._taskStateLst[i]
            task._initState = self._taskStateLst[i]
            task.cRate = self._taskRateLst[i]
            task._initRate = self._taskRateLst[i]
            task._threhod = self._threhold
            task.cmpltTime = sys.float_info.max
            self.taskLst.append(task)
        if degBoolean:
            self._actSeq = ActionSeq()
    def decodeProcessor(self):
        while not self.allTaskCmplt():
            cal_type, actionID = self.findActionID()
            if cal_type == CalType['arriveCond']:
                rob = self.robotLst[actionID]
                arriveTime = rob.arriveTime
                encodeInd = rob.encodeIndex
                taskID = self.encode[actionID][encodeInd]
                if degBoolean:
                    self._actSeq.append(
                        ActionTuple(robID=actionID, taskID=taskID, eventType=EventType.arrive, eventTime=arriveTime))

                if self.cmpltLst[taskID]:
                    # =============================================================================
                    #  the task has been cmplt
                    # =============================================================================
                    rob = self.robotLst[actionID]
                    rob.leaveTime = rob.arriveTime
                    rob.taskID = taskID
                    rob.stateType = RobotState['onTask']
                    # raise Exception('xxx')
                    self._arrCmpltTaskLst.append((actionID,taskID))
                    if degBoolean:
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
                        task.cmpltTime = leaveTime
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
                task.cmpltTime = rob.leaveTime
                coordLst = self.findCoordRobot(actionID)

                # print('task ',taskID,' has been completed')
                if self.allTaskCmplt():
                    return True,[]
                else:
                    updateRobLst = [actionID]
                    updateRobLst.extend(coordLst)
                    # print(updateRobLst)
                    # print('task cmpltTime = ', task.cmpltTime)
                    # print('leaveTime = ')

                    return False,updateRobLst

                '''
                debug is here
                '''

                if degBoolean:
                    self._degFile.write(str(taskID) + ' have been completed\n')
            if cal_type == CalType.updateCond:
                coordLst = self.findCoordRobot(actionID)
                coordLst.append(actionID)
                return False,coordLst
                # raise Exception('sssss')
            if cal_type == CalType.endCond:
                # invalidFitness = True
                validStateBool = False
                return True,[]
                # break

        if not validStateBool:
            pass
        # if self.allTaskCmplt()

            # print('the state is explosion')
        return validStateBool

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
                        # if rob.encodeIndex == (len(self.encode[i]) -1):
                        #     cal_type = CalType['updateCond']
                        #     actionID = i
                        # else:
                        cal_type = CalType['leaveCond']
                        actionID = i

        if degBoolean:
            self.saveRobotInfo(degFile=self._degFile)
            self._degFile.write(str(actionID) + ' time = ' + str(minTime)
                                + ' type = ' + str(cal_type) + '\n')
        # self.saveEventInMemory()
        # if minTime < self.decodeTime:
        #     cal_type = CalType['backCond']
        #            print(minTime)
        #            print(self.decodeTime)
        #            taskID = self.robotLst[actionI].taskID
        #        self.saveRobotInfo()
        # if cal_type == CalType.endCond:
        #     for i in range(self._robNum):
        #         rob = self.robotLst[i]
        #         print(rob.leaveTime)
            # raise Exception('cal_type endCond')
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

    def updateRobLeaveCond(self, robID):
        rob = self.robotLst[robID]
        preTaskID = rob.taskID
        while True:
            if rob.encodeIndex == (self._taskNum - 1):
                rob.stopBool = True
                break
            rob.encodeIndex += 1
            # print(self.encode[robID])
            taskID = self.encode[robID][rob.encodeIndex]
            roadDur = self.calRoadDur(preTaskID, taskID, robID)
            arriveTime = rob.leaveTime + roadDur
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

    def calRoadDur(self, taskID1, taskID2, robID):
        '''
        calculate the time fragment from the time when robID leaves the taskID1 to
        the time when rob arrives the taskID2
        '''
        dis = self._taskDisMat[taskID1][taskID2]
        rob = self.robotLst[robID]
        roadDur = dis / rob._vel
        return roadDur

    def saveRobotInfo(self, degFile):
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
            str_lst = [str(x) for x in lst]
            robInfo = '  '
            robInfo = robInfo.join(str_lst)
            deg.write(robInfo + '\n')
        deg.write('\n')
        deg.flush()