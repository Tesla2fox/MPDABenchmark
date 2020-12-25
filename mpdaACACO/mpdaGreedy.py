import os
import sys
import readcfg as r_d
from mpdaInstance import MPDAInstance
import os, sys
from mpdaDecodeMethod.mpdaRobot import RobotState,Robot
from mpdaDecodeMethod.mpdaTask import  Task
from mpdaDecodeMethod.mpdaDecoderActSeq import ActionSeq,ActionTuple,EventType,MPDADecoderActionSeq
import numpy as np
from enum import Enum
from collections import  namedtuple
import math

degBoolean = False

AbsolutePath = os.path.abspath(__file__)
SuperiorCatalogue = os.path.dirname(AbsolutePath)
BaseDir = os.path.dirname(SuperiorCatalogue)

from mpdaDecodeMethod.mpdaDecodeCon import RobTaskPair
from mpdaDecodeMethod.mpdaDecode import MPDADecoder,generateRandEncode

import readcfg as rd
import copy

import  random
class CalType(Enum):
    arriveCond = 1
    leaveCond = 2
    endCond = 3
    backCond = 4
    stateInvalidCond = 5
    updateCond = 6

class MPDA_Greedy(object):
    def __init__(self,ins :MPDAInstance, h_rule  = '_Dis'):
        self._ins =  ins
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
        # self._taskConLst = ins._taskConLst
        if degBoolean:
            self._degFile = open(BaseDir+ '/debugData/deg.dat', 'w')
        self.encode = [[] for x in range(self._robNum)]

        if h_rule == '_Dis':
            self.calHeuristic =  self.calHeuristicDis
        elif h_rule == '_Edur':
            self.calHeuristic = self.calHeuristicEDur
        elif h_rule == '_Cdur':
            self.calHeuristic = self.calHeuristicCDur
        elif h_rule == '_Syn':
            self.calHeuristic = self.calHeuristicSynthesis
        elif h_rule == '_Drvis':
            self.calHeuristic = self.calHeuristicDrvis
        elif h_rule == '_SPP':
            self.calHeuristic = self.calHeuristicSpp
        elif h_rule == '_Dvis':
            self.calHeuristic = self.calHeuristicDvis
        else:
            raise  Exception('not in the h_rule of greedy method')
    def construct(self):
        self.genRobFirstAct()
        self.initStates()
        while True:
            cmpltBool,updateRobLst = self.updateState()
            if cmpltBool:
                break
            # print(nextUpdateRobTaskPairLst)
            nextUpdateRobTaskPairLst = self.generateNextUpdateRobTaskPairLst(updateRobLst)
            if nextUpdateRobTaskPairLst == []:
                pass
            self.updateEncode(nextUpdateRobTaskPairLst)
            # print(self.encode)
        fitness = self.calFitness()
        encode = self.encode
        # print(encode)
        return encode,fitness

    def updateState(self):
        cmpltBool,updateRobLst = self.decodeProcessor()
        # stateBool,updateRobLst = self.decodeProcessor()
        return cmpltBool,updateRobLst
    def updateEncode(self,robTaskPairLst : list):
        for rob_task_pair in robTaskPairLst:
            self.encode[rob_task_pair.robID].append(rob_task_pair.taskID)
            self.updateRobLeaveCond(robID= rob_task_pair.robID)
    def genRobFirstAct(self):
        self.robInitVisitLst = []
        robSeq = [x for x in range(self._robNum)]
        random.shuffle(robSeq)

        for robID in robSeq:
            eta_Lst = []
            rob2taskLst = list(self._rob2taskDisMat[robID])
            for taskID in range(self._taskNum):
                eta =  self.calHeuristic(robID,taskID,dummy = True)
                eta_Lst.append((taskID,eta))
            # print('CdurLst = ',e_durLst)
            minUnit = min(eta_Lst, key=lambda x: x[1])
            taskID = minUnit[0]
            # taskID = rob2taskLst.index(min(rob2taskLst))
            self.robInitVisitLst.append(taskID)
            self.encode[robID].append(taskID)
        return self.robInitVisitLst

    def generateNextUpdateRobTaskPairLst(self, updateRobLst):
        nextUpdateRobTaskPairLst = []
        for robID in updateRobLst:
            candinateLst = []
            rob = self.robotLst[robID]
            eta_Lst = []
            for taskID in range(self._taskNum):
                if taskID in self.encode[robID]:
                    continue
                if self.cmpltLst[taskID]:
                    continue
                roadDur = self._taskDisMat[rob.taskID][taskID] / self._robVelLst[robID]
                predictArrTime = self.robotLst[robID].leaveTime + roadDur
                if predictArrTime > self.taskLst[taskID].cmpltTime:
                    continue
                eta = self.calHeuristic(robID,taskID,dummy = False)
                eta_Lst.append((taskID,eta))
            if len(eta_Lst) == 0:
                rob.stopBool = True
                continue
            minUnit = min(eta_Lst, key=lambda x: x[1])
            # taskID = minUnit[0]
            nextUpdateRobTaskPairLst.append(RobTaskPair(robID=robID, taskID=minUnit[0]))
        return nextUpdateRobTaskPairLst

    def calHeuristicDis(self,robID,taskID,dummy):
        if dummy == True:
            roadDur = self._rob2taskDisMat[robID][taskID] / self._robVelLst[robID]
            h_eta = roadDur
        else:
            rob = self.robotLst[robID]
            roadDur = self._taskDisMat[rob.taskID][taskID] / self._robVelLst[robID]
            h_eta = roadDur
        return h_eta

    def calHeuristicSpp(self,robID,taskID,dummy):
        if dummy == True:
            # rateLst = copy.copy(self._taskRateLst)
            cRate = self._taskRateLst[taskID]
            for i_robID,i_taskID in enumerate(self.robInitVisitLst):
                if i_taskID == taskID:
                    cRate =  cRate - self._robAbiLst[i_robID]
                # rateLst[taskID] = rateLst[taskID] - self._robAbiLst[robID]
            h_eta = 1/math.exp(cRate)
        else:
            rob = self.robotLst[robID]
            roadDur = self._taskDisMat[rob.taskID][taskID] / self._robVelLst[robID]
            h_eta = roadDur
        return h_eta
    def calHeuristicDvis(self,robID,taskID,dummy):
        if dummy == True:
            roadDur = self._rob2taskDisMat[robID][taskID] / self._robVelLst[robID]
            h_eta = roadDur / self._taskRateLst[taskID]
        else:
            rob = self.robotLst[robID]
            roadDur = self._taskDisMat[rob.taskID][taskID] / self._robVelLst[robID]
            h_eta = roadDur / self._taskRateLst[taskID]
        return h_eta

    def calHeuristicDrvis(self,robID,taskID,dummy):
        if dummy == True:
            roadDur = self._rob2taskDisMat[robID][taskID] / self._robVelLst[robID]
            h_eta = roadDur * self._taskRateLst[taskID]
        else:
            rob = self.robotLst[robID]
            roadDur = self._taskDisMat[rob.taskID][taskID] / self._robVelLst[robID]
            h_eta = roadDur * self._taskRateLst[taskID]
        return h_eta

    def calHeuristicEDur(self,robID,taskID,dummy):
        if dummy == True:
            roadDur = self._rob2taskDisMat[robID][taskID] / self._robVelLst[robID]
            changeDur = roadDur
            incre = changeDur * self._taskRateLst[taskID]
            cDemand = self._taskStateLst[taskID] + incre
            cRate = self._taskRateLst[taskID] - self._robAbiLst[robID]

            futureLst = []

            if cRate >= 0:
                e_dur = 99999999
            else:
                e_dur = (cDemand - self._threhold) / (-cRate)
            h_eta = e_dur
        else:
            rob = self.robotLst[robID]
            roadDur = self._taskDisMat[rob.taskID][taskID] / self._robVelLst[robID]
            predictArrTime = self.robotLst[robID].leaveTime + roadDur
            task = self.taskLst[taskID]
            # roadDur = rob2taskLst[taskID] / self._robVelLst[robID]
            changeDur = predictArrTime - task.changeRateTime
            incre = changeDur * task.cRate
            cDemand = task.cState + incre
            cRate = task.cRate - self._robAbiLst[robID]
            if cRate >= 0:
                e_dur = 99999999
            else:
                e_dur = (cDemand - self._threhold) / (-cRate)
            h_eta = e_dur
        return h_eta
    def calHeuristicCDur(self,robID,taskID,dummy):
        if dummy == True:
            roadDur = self._rob2taskDisMat[robID][taskID] / self._robVelLst[robID]
            changeDur = roadDur
            incre = changeDur * self._taskRateLst[taskID]
            cDemand = self._taskStateLst[taskID] + incre
            cRate = self._taskRateLst[taskID] - self._robAbiLst[robID]
            if cRate >= 0:
                e_dur = 99999999
            else:
                e_dur = (cDemand - self._threhold) / (-cRate)
            h_eta = e_dur + roadDur
        else:
            rob = self.robotLst[robID]
            roadDur = self._taskDisMat[rob.taskID][taskID] / self._robVelLst[robID]
            predictArrTime = self.robotLst[robID].leaveTime + roadDur
            task = self.taskLst[taskID]
            # roadDur = rob2taskLst[taskID] / self._robVelLst[robID]
            changeDur = predictArrTime - task.changeRateTime
            incre = changeDur * task.cRate
            cDemand = task.cState + incre
            cRate = task.cRate - self._robAbiLst[robID]
            if cRate >= 0:
                e_dur = 99999999
            else:
                e_dur = (cDemand - self._threhold) / (-cRate)
            h_eta = e_dur + predictArrTime
        return h_eta
    def calHeuristicSynthesis(self,robID,taskID,dummy):
        if dummy == True:
            roadDur = self._rob2taskDisMat[robID][taskID] / self._robVelLst[robID]
            dis_h_eta = 1/roadDur
            changeDur = roadDur
            # incre = changeDur * self._taskRateLst[taskID]
            cRate = self._taskRateLst[taskID] - self._robAbiLst[robID]
            if cRate >= 0:
                # print(cRate)
                # raise Exception('ssss')
                incre = changeDur * self._taskRateLst[taskID]
                cDemand = self._taskStateLst[taskID] + incre
                e_dur = (cDemand - self._threhold)/ (0.1 * self._robAbiLst[robID])
                cmpltTime = e_dur + roadDur
                contribution = self._taskRateLst[taskID] /(e_dur + roadDur)
                # robConsumeDemand = e_dur * self._robAbiLst[robID]
                contribution_h_eta = (self._robAbiLst[robID]/self._taskRateLst[taskID]) * contribution
                # self._robAbiLst[]
                pass
            else:
                incre = changeDur * self._taskRateLst[taskID]
                cDemand = self._taskStateLst[taskID] + incre
                e_dur = (cDemand - self._threhold) / (-cRate)
                cmpltTime = e_dur + roadDur
                contribution = self._taskRateLst[taskID] /(cmpltTime)
                totalDemand = cmpltTime * self._taskRateLst[taskID] + self._taskStateLst[taskID]
                robConsumeDemand = e_dur * self._robAbiLst[robID]
                contribution_h_eta = (robConsumeDemand/totalDemand) * contribution
            h_eta = contribution_h_eta * dis_h_eta
        else:
            rob = self.robotLst[robID]
            roadDur = self._taskDisMat[rob.taskID][taskID] / self._robVelLst[robID]
            dis_h_eta = 1/roadDur

            predictArrTime = self.robotLst[robID].leaveTime + roadDur
            task = self.taskLst[taskID]
            changeDur = predictArrTime - task.changeRateTime
            incre = changeDur * task.cRate
            cDemand = task.cState + incre
            cRate = task.cRate - self._robAbiLst[robID]
            if cRate >= 0:
                e_dur = (cDemand - self._threhold)/ (0.1 * cRate)
                cmpltTime = e_dur + predictArrTime
                contribution = self._taskRateLst[taskID] /(cmpltTime)
                # robConsumeDemand = e_dur * self._robAbiLst[robID]
                contribution_h_eta = (self._robAbiLst[robID]/self._taskRateLst[taskID]) * contribution
                # raise Exception('sssss')
            else:
                e_dur = (cDemand - self._threhold) / (-cRate)
                cmpltTime = e_dur + predictArrTime
                contribution = self._taskRateLst[taskID] /(e_dur + roadDur)
                totalDemand = cmpltTime * self._taskRateLst[taskID] + self._taskStateLst[taskID]
                robConsumeDemand = e_dur * self._robAbiLst[robID]
                contribution_h_eta = (robConsumeDemand/totalDemand) * contribution
            h_eta = contribution_h_eta * dis_h_eta
            # rob = self.robotLst[robID]
            # roadDur = self._taskDisMat[rob.taskID][taskID] / self._robVelLst[robID]
        return h_eta

    def calFitness(self):
        cmpltTimeLst = []
        for task in self.taskLst:
            cmpltTimeLst.append(task.cmpltTime)
        return max(cmpltTimeLst)

    def initStates(self):
        '''
        initialize states of decode method
        '''
        self.taskLst = []
        self.robotLst = []
        self.cmpltLst = [False] * self._taskNum
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

        for i in range(self._taskNum):
            task = Task()
            # print(self._taskStateLst[i])
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
                        for coordID in coordLst:
                            coordRob = self.robotLst[coordID]
                            coordRob.leaveTime = leaveTime
                            coordRob.executeDur = coordRob.leaveTime - coordRob.arriveTime
                    task.cmpltTime = leaveTime
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

                if self.allTaskCmplt():
                    return True,[]
                else:
                    updateRobLst = [actionID]
                    updateRobLst.extend(coordLst)
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
                break


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
            taskID = self.encode[robID][rob.encodeIndex]
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


    def estimate4SPPScope(self):
        rob_sum_abi = sum(self._robAbiLst)
        rob_meanTsk_abi = rob_sum_abi/ self._taskNum

        task_mean_abi = np.mean(self._taskRateLst)
        if rob_meanTsk_abi <= task_mean_abi:
            return  1
        mean_dis = np.mean(self._taskDisMat)
        SPPcost = mean_dis + (task_mean_abi * mean_dis)/ (rob_meanTsk_abi - task_mean_abi)
        print(SPPcost)

        _2cost = 0
        base =  mean_dis + (task_mean_abi * mean_dis) /(rob_sum_abi/2 - task_mean_abi)
        _2cost += base
        for x in range(math.ceil(self._taskNum/2)):
            base = mean_dis + (task_mean_abi * (_2cost + mean_dis) )/ (rob_sum_abi/2 -task_mean_abi)
            # step_cost = (self._taskNum -x) * ((task_mean_abi/(rob_sum_abi/2 - task_mean_abi)) ** x)
            # print(x , step_cost)
            _2cost += base

        print(_2cost)
        # print(' robNum = ',self._robNum)
        # print(' taskNum = ', self._taskNum)
        # exit()
        if _2cost /SPPcost > 100:
            return 0
        else:
            return  1
        # print()
        # exit()
    def estimate4TSPScope(self):
        rob_sum_abi = sum(self._robAbiLst)
        task_mean_abi = np.mean(self._taskRateLst)

        mean_dis = np.mean(self._taskDisMat)

        cost = 0
        base =  mean_dis + (task_mean_abi * mean_dis) /(rob_sum_abi - task_mean_abi)
        cost += base
        for x in range(self._taskNum):
            base = mean_dis + (task_mean_abi * (cost + mean_dis) )/ (rob_sum_abi -task_mean_abi)
            # step_cost = (self._taskNum -x) * ((task_mean_abi/(rob_sum_abi/2 - task_mean_abi)) ** x)
            # print(x , step_cost)
            cost += base
        print(cost)

        if rob_sum_abi/2 - task_mean_abi <0:
            # print(rob_sum_abi/2)
            # print(task_mean_abi)
            # raise Exception('sss')
            return 0
        _2cost = 0
        base =  mean_dis + (task_mean_abi * mean_dis) /(rob_sum_abi/2 - task_mean_abi)
        _2cost += base
        for x in range(math.ceil(self._taskNum/2)):
            base = mean_dis + (task_mean_abi * (_2cost +mean_dis) )/ (rob_sum_abi/2 -task_mean_abi)
            # step_cost = (self._taskNum -x) * ((task_mean_abi/(rob_sum_abi/2 - task_mean_abi)) ** x)
            # print(x , step_cost)
            _2cost += base
        print(_2cost)

        _initCost = 0
        # exit()
        if _2cost/cost > 100:
            return 0
        else:
            return 1
        # exit()


class MPDA_Mean_Greedy(object):
    def __init__(self, ins: MPDAInstance,benchmarkName):
        self._ins = ins
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
        self.benchmarkName = benchmarkName

    def construct(self):
        readCfg = rd.Read_Cfg(self._ins._insName)
        rob_x_lst = []
        rob_y_lst = []
        readCfg.get('rob_x',rob_x_lst)
        readCfg.get('rob_y',rob_y_lst)
        mean_rob_x = np.mean(rob_x_lst)
        mean_rob_y = np.mean(rob_y_lst)

        tsk_x_lst = []
        tsk_y_lst = []
        readCfg.get('task_x',tsk_x_lst)
        readCfg.get('task_y',tsk_y_lst)
        _taskDisLst = []
        robVisitedLst = []

        for taskID in range(self._taskNum):
            _taskDisLst.append(
                np.linalg.norm([tsk_x_lst[taskID] - mean_rob_x, tsk_y_lst[taskID] - mean_rob_y], ord=2)
            )
        while len(robVisitedLst) != self._taskNum:
            taskID = _taskDisLst.index(min(_taskDisLst))
            # taskID = minUnit[0]
            robVisitedLst.append(taskID)
            _taskDisLst = []
            preTaskID = taskID
            for taskID in range(self._taskNum):
                if taskID in robVisitedLst:
                    _taskDisLst.append(np.inf)
                else:
                    _taskDisLst.append(self._taskDisMat[preTaskID][taskID])
        # _taskDisLst = []
        # robVisitedLst = []
        # for taskID in range(self._taskNum):
        #     _taskDisLst.append(self._taskRateLst[taskID])
        # while len(robVisitedLst) != self._taskNum:
        #     taskID = _taskDisLst.index(max(_taskDisLst))
        #     robVisitedLst.append(taskID)
        #     _taskDisLst = []
        #     preTaskID = taskID
        #     for taskID in range(self._taskNum):
        #         if taskID in robVisitedLst:
        #             _taskDisLst.append(-np.inf)
        #         else:
        #             _taskDisLst.append(self._taskRateLst[taskID])
        mid_encode = [copy.copy(robVisitedLst) for x in range(self._robNum)]
        decoder = MPDADecoder(self._ins)
        decoder.decode(mid_encode)
        fitness = decoder.calMakespan()
        print(fitness)
        # exit()
        return mid_encode, fitness

