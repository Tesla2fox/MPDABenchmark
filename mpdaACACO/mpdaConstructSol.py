from mpdaACACO.mpdaTaskUpdater import ACO_Updater
from numpy.random import choice as np_choice
import numpy as np
import sys
import random
import math
from mpdaDecodeMethod.mpdaDecodeCon import RobTaskPair
import copy
from mpdaDecodeMethod.mpdaDecode import MPDADecoder

debugBool = False
if debugBool:
    f_con_deg = open('xxxxxx.dat', 'w')

class ACO_Constructor(object):
    def __init__(self,ins, alpha, beta, taskPheromoneLst, robTaskPheromoneLst, rob_sum_abi, task_sum_Abi):
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
        self.taskPheromoneLst = taskPheromoneLst
        self.robTaskPheromoneLst = robTaskPheromoneLst

        self.alpha = alpha
        self.beta = beta

        self.rob_sum_abi = rob_sum_abi
        self.task_sum_abi = task_sum_Abi

        self.calHeuristic = self.calHeursitic_disDemandPdis
        self.rob_sum_abi = rob_sum_abi
        self.task_sum_abi = task_sum_Abi
        self.sum_ratio = self.task_sum_abi / self.rob_sum_abi
        self.pInd = 1
        self.genRobFirstAct = self.genRobFirstActTrad


    def construct(self):
        robInitVisitLst = self.genRobFirstAct()
        self._mpdaUpdater = ACO_Updater(self._ins)
        self._mpdaUpdater.addInitVisitLst(robInitVisitLst)
        self.robotLst = self._mpdaUpdater.robotLst
        self.taskLst = self._mpdaUpdater.taskLst
        self.encode = self._mpdaUpdater.encode
        encode, fitness, arrCmpltTaskLst = self.genRobSol()
        return [encode, fitness, arrCmpltTaskLst]

    def genRobSol(self):
        while True:
            cmpltBool,updateRobLst = self._mpdaUpdater.updateState()
            # print(cmpltBool)
            if cmpltBool:
                break
            # print(updateRobLst)
            nextUpdateRobTaskPairLst = []
            random.shuffle(updateRobLst)
            self.updateRobDic = dict()
            for robID in updateRobLst:
                self.updateRobDic[robID] = np.inf

            for robID in updateRobLst:
                rob = self.robotLst[robID]
                pheromoneLst = np.copy(self.taskPheromoneLst[robID][rob.taskID])
                pheromoneLst[self.encode[robID]] = 0
                row = np.zeros([self._taskNum])
                for taskID, pheromone in enumerate(pheromoneLst):
                    # print(pheromone)
                    if pheromone == 0:
                        continue
                    else:
                        eta = self.calHeuristic(robID, taskID, dummy=False)
                        row[taskID] = pheromone ** self.alpha * (eta ** self.beta)

                row_sum = row.sum()
                if row_sum == 0:
                    rob.stopBool = True
                    break
                norm_row = row / row_sum
                next_taskID = np_choice(range(self._taskNum), 1, p=norm_row)[0]
                nextUpdateRobTaskPairLst.append(RobTaskPair(robID = robID, taskID = next_taskID))
                self.updateRobDic[robID] = next_taskID

            # print(nextUpdateRobTaskPairLst)
            if len(nextUpdateRobTaskPairLst) == 0:
                continue
            if False not in self._mpdaUpdater.cmpltLst:
               break
            encode, fitness, nextUpdateRobTaskPairLst = self.fixSol(nextUpdateRobTaskPairLst,dummy= False)
            if fitness != 0:
                return encode, fitness,[]
            self._mpdaUpdater.updateEncode(nextUpdateRobTaskPairLst)
            # print(self.encode)

        fitness = self.calFitness()
        encode = self.encode
        arrCmpltTaskLst = self._mpdaUpdater._arrCmpltTaskLst
        if fitness == sys.float_info.max:
            pass
        return encode,fitness,arrCmpltTaskLst

    def fixSol(self,nextUpdateRobTaskPairLst,dummy):

        updateRobtLst = [unit.robID for unit in nextUpdateRobTaskPairLst]
        nextUpdateRobTaskPairDic = dict()
        for robID,taskID in nextUpdateRobTaskPairLst:
            nextUpdateRobTaskPairDic[robID] = taskID
        curAccRobAbiLst = [0 for x in self._taskRateLst]

        for robID in range(self._robNum):
            if robID in updateRobtLst:
                nextVisitTaskID = nextUpdateRobTaskPairLst[updateRobtLst.index(robID)].taskID
                curAccRobAbiLst[nextVisitTaskID] = curAccRobAbiLst[nextVisitTaskID] + self._robAbiLst[robID]
            else:
                nextVisitTaskID = self._mpdaUpdater.encode[robID][-1]
                curAccRobAbiLst[nextVisitTaskID] = curAccRobAbiLst[nextVisitTaskID] + self._robAbiLst[robID]
        curTaskRateLst = []

        for taskID in range(self._taskNum):
            if curAccRobAbiLst[taskID] == 0:
                continue
            if dummy == False:
                if self._mpdaUpdater.cmpltLst[taskID]:
                    continue
            curTaskRateLst.append((taskID,self._taskRateLst[taskID] -curAccRobAbiLst[taskID] ))

        if len(curTaskRateLst) == 0:
            return [],0,nextUpdateRobTaskPairLst

        minTaskID,minCurTaskRate = min(curTaskRateLst,key = lambda x: x[1])

        if minCurTaskRate >=0 or math.isclose(minCurTaskRate, 0, abs_tol = 0.000001 ):
            # print('minCurTaskRate = asdsaddsad ')
            row = np.zeros([self._robNum])
            preMinCurTaskRate = minCurTaskRate
            for robID in updateRobtLst:
                if nextUpdateRobTaskPairDic[robID] != minTaskID:
                    # print('b p = ',preMinCurTaskRate  )
                    # print('robID = ',robID)
                    # print('rob_abi = ',self._robAbiLst[robID])
                    preMinCurTaskRate = preMinCurTaskRate - self._robAbiLst[robID]
                    # print('a p = ',preMinCurTaskRate  )
                    # print(self.taskPheromoneLst[robID][self.encode[robID][-1]][minTaskID])
                    if dummy:
                        row[robID] = self.robTaskPheromoneLst[robID][minTaskID]
                    else:
                        row[robID] = self.taskPheromoneLst[robID][self.encode[robID][-1]][minTaskID]
            row_sum = row.sum()
            if row_sum != 0:
                norm_row = row / row_sum
            # norm_row = list(norm_row)
            #     print(norm_row)
            # print('minminTaskID = ',minTaskID)
            # print('minCurTaskRate = ',minCurTaskRate)
            # print('min = ',preMinCurTaskRate)
            # print(minCurTaskRate + curAccRobAbiLst[minTaskID])
            if preMinCurTaskRate >=0 or math.isclose(preMinCurTaskRate,0,abs_tol = 0.01):
                # print(preMinCurTaskRate)
                # print(self.encode)
                encode = copy.deepcopy(self.encode)
                # encode = copy.copy(self.encode)
                limitedSet = set()
                for robID in range(self._robNum):
                    if robID in updateRobtLst:
                        encode[robID].append(minTaskID)
                    else:
                        limitedSet.add(encode[robID][-1])
                        encode[robID][-1] = minTaskID

                cmpltTaskSet = set()
                for robID in range(self._robNum):
                    for i in range(len(encode[robID])):
                        cmpltTaskSet.add(encode[robID][i])
                # print(cmpltTaskSet)

                # _taskDisLst = [ for taskID in range(self._taskNum)]
                pretaskID = minTaskID
                robVisitedLst = []
                while len(cmpltTaskSet) != self._taskNum:
                    _taskDisLst = []
                    for taskID in range(self._taskNum):
                        if taskID in cmpltTaskSet:
                            _taskDisLst.append(np.inf)
                        else:
                            _taskDisLst.append(self._taskDisMat[pretaskID][taskID])
                    taskID = _taskDisLst.index(min(_taskDisLst))
                    pretaskID = taskID
                    robVisitedLst.append(taskID)
                    cmpltTaskSet.add(taskID)
                mid_encode = [copy.copy(robVisitedLst) for x in range(self._robNum)]
                for robID in range(self._robNum):
                    encode[robID].extend(mid_encode[robID])
                decoder = MPDADecoder(self._ins)
                decoder.decode(encode)
                fitness = decoder.calMakespan()
                # print(fitness)
                # for robID in range(self._robNum):
                # raise Exception('exist some problems')
                return encode,fitness,nextUpdateRobTaskPairLst
        else:
            nextUpdateRobTaskPairLst = sorted(nextUpdateRobTaskPairLst, key=lambda x: x[0])
            return [],0,nextUpdateRobTaskPairLst
        nextChangeRobTaskPairLst = []
        nextChangeRobIDLst = []
        # print('minCurTaskRate = ',minCurTaskRate)
        while minCurTaskRate >= 0 or  math.isclose(minCurTaskRate,0,abs_tol = 0.000001  ):
            for robID in updateRobtLst:
                if norm_row[robID] != 0:
                    q = random.random()
                    # print('norm_row = ',norm_row[robID])
                    # print('rand = ',q)
                    if  q < norm_row[robID]:
                        minCurTaskRate = minCurTaskRate - self._robAbiLst[robID]
                        nextChangeRobTaskPairLst.append(RobTaskPair(robID,minTaskID))
                        nextChangeRobIDLst.append(robID)
                        norm_row[robID] = 0
            row_sum = norm_row.sum()
            norm_row = norm_row/row_sum

        for key in nextUpdateRobTaskPairDic:
            if key not in nextChangeRobIDLst:
                nextChangeRobTaskPairLst.append(RobTaskPair(key, nextUpdateRobTaskPairDic[key]))
        nextChangeRobTaskPairLst  = sorted(nextChangeRobTaskPairLst, key = lambda x : x[0])

        return [],0,nextChangeRobTaskPairLst
    def genRobFirstActTrad(self):
        '''
        is used to generate first event.
        :return the first event:
        '''
        robInitVisitLst = []
        robSeq = [x for x in range(self._robNum)]
        random.shuffle(robSeq)

        robInitVisitLst = [np.inf for x in range(self._robNum)]
        self.robInitVisitLst = [np.inf for x in range(self._robNum)]

        for robID in robSeq:
            row = np.zeros([self._taskNum])
            for taskID,pheromone in enumerate(self.robTaskPheromoneLst[robID]):
                eta = self.calHeuristic(robID,taskID,dummy=True)
                row[taskID] = pheromone ** self.alpha * (eta ** self.beta)
                # roadDur = self._rob2taskDisMat[robID][taskID] / self._robVelLst[robID]
            row_sum = row.sum()
            norm_row = row / row_sum

            firstAct = np_choice(range(self._taskNum), 1, p=norm_row)[0]

            robInitVisitLst[robID] = firstAct
            self.robInitVisitLst[robID] = firstAct

        '''
        此处需要和后续的生成方法一致
        '''
        # print('robInitVisitLst = ',robInitVisitLst)
        nextUpdateRobTaskPairLst = []
        for robID,taskID in enumerate(robInitVisitLst):
            nextUpdateRobTaskPairLst.append(RobTaskPair(robID,taskID))
        encode, fitness, nextUpdateRobTaskPairLst = self.fixSol(nextUpdateRobTaskPairLst, dummy=True)
        robInitVisitLst = [x.taskID for x in nextUpdateRobTaskPairLst]
        # print('robInitVisitLst = ',robInitVisitLst)
        return robInitVisitLst

    def updatingLimited(self,nextUpdateRobTaskPairLst):
        updateRobLst = []
        for robID,taskID in nextUpdateRobTaskPairLst:
            updateRobLst.append(robID)

        curAccRobAbiLst = [0 for x in self._taskRateLst]
        for robID in range(self._robNum):
            if robID in updateRobLst:
                nextVisitTaskID = nextUpdateRobTaskPairLst[updateRobLst.index(robID)].taskID
                curAccRobAbiLst[nextVisitTaskID] = curAccRobAbiLst[nextVisitTaskID] + self._robAbiLst[robID]
            else:
                nextVisitTaskID = self._mpdaUpdater.encode[robID][-1]
                curAccRobAbiLst[nextVisitTaskID] = curAccRobAbiLst[nextVisitTaskID] + self._robAbiLst[robID]

        curTaskRateLst = []
        for taskID in range(self._taskNum):
            if curAccRobAbiLst[taskID] == 0:
                continue
            if self._mpdaUpdater.cmpltLst[taskID]:
                continue
            curTaskRateLst.append((taskID,self._taskRateLst[taskID] -curAccRobAbiLst[taskID] ))

        # print('length curTaskRateLst = ',len(curTaskRateLst))
        if len(curTaskRateLst) <= self.limitedNum:
            return nextUpdateRobTaskPairLst

        minTaskID,minCurTaskRate = min(curTaskRateLst,key = lambda x: x[1])

        limitedSet = set()
        for robID in range(self._robNum):
            if robID in updateRobLst:
                pass
            else:
                taskID = self.encode[robID][-1]
                if self._mpdaUpdater.cmpltLst[taskID] == False:
                    limitedSet.add(taskID)

        newVisitedTaskLst = []
        for robID,taskID in nextUpdateRobTaskPairLst:
            if taskID not in limitedSet:
                newVisitedTaskLst.append(taskID)
        # print(limitedSet)
        # print(newVisitedTaskLst)


        '''
        此处可以修正 如何将 newVisitedTaskLst 的元素加入 limitedSet 中
        '''
        random.shuffle(newVisitedTaskLst)
        # print(newVisitedTaskLst)
        while len(limitedSet) < self.limitedNum:
            limitedSet.add(newVisitedTaskLst[0])
            newVisitedTaskLst.remove(newVisitedTaskLst[0])
        # print(limitedSet)

        nextUpdateRobTaskPairLst = []
        for robID in updateRobLst:
            rob = self.robotLst[robID]
            pheromoneLst = np.copy(self.taskPheromoneLst[robID][rob.taskID])
            pheromoneLst[self.encode[robID]] = 0
            row = np.zeros([self._taskNum])
            for taskID, pheromone in enumerate(pheromoneLst):
                # print(pheromone)
                if taskID not in limitedSet:
                    continue
                if pheromone == 0:
                    continue
                else:
                    eta = self.calHeuristic(robID, taskID, dummy=False)
                    roadDur = self._taskDisMat[rob.taskID][taskID] / self._robVelLst[robID]
                    predictArrTime = self.robotLst[robID].leaveTime + roadDur
                    if predictArrTime > self.taskLst[taskID].cmpltTime:
                        continue
                    row[taskID] = pheromone ** self.alpha * (eta ** self.beta)
            row_sum = row.sum()
            if row_sum == 0:
                rob.stopBool = True
                break
            norm_row = row / row_sum
            # print(norm_row)
            next_taskID = np_choice(range(self._taskNum), 1, p=norm_row)[0]
            nextUpdateRobTaskPairLst.append(RobTaskPair(robID=robID, taskID=next_taskID))
        return nextUpdateRobTaskPairLst

    def calFitness(self):
        cmpltTimeLst = []
        for task in self.taskLst:
            cmpltTimeLst.append(task.cmpltTime)
        return max(cmpltTimeLst)


    def calHeursitic_disDemandPdis(self,e_robID,e_taskID,dummy):
        if dummy == True:
            roadDur = self._rob2taskDisMat[e_robID][e_taskID] / self._robVelLst[e_robID]
            dis_h_eta = 1 / roadDur
            changeDur = roadDur
            futureEventLst = []
            for f_robID in range(self._robNum):
                if self.robInitVisitLst[f_robID] == e_taskID:
                    futureEventLst.append((f_robID,e_taskID))
            if len(futureEventLst) != 0:
                # print(futureEventLst)
                # if self.sum_ratio < 1:
                cRate = self._taskRateLst[e_taskID]
                        # - self._robAbiLst[e_robID]
                for f_robID,f_taskID in futureEventLst:
                    cRate =  cRate - self._robAbiLst[f_robID]
                if cRate >= 0 and len(futureEventLst) > 0:
                    h_eta = dis_h_eta * (len(futureEventLst) + 1) ** self.pInd
                    # h_eta = dis_h_eta
                elif cRate >= 0 and  len(futureEventLst) == 0:
                    h_eta = dis_h_eta
                else:
                    if self.sum_ratio > 2:
                        h_eta = dis_h_eta * (len(futureEventLst) + 1)
                    else:
                        h_eta = dis_h_eta * ((1/(len(futureEventLst) + 1) ) ** (self.pInd))
                    # h_eta = dis_h_eta * (len(futureEventLst) +1)
                    return h_eta
            else:
                h_eta =  dis_h_eta
            return h_eta
        else:
            rob = self.robotLst[e_robID]
            roadDur = self._taskDisMat[rob.taskID][e_taskID] / self._robVelLst[e_robID]
            predictArrTime = self.robotLst[e_robID].leaveTime + roadDur
            if predictArrTime > self.taskLst[e_taskID].cmpltTime:
                return 0
            roadDur = self._taskDisMat[rob.taskID][e_taskID] / self._robVelLst[e_robID]
            cRate = self._taskRateLst[e_taskID]
            e_taskRobNum = 0
            for f_robID in range(self._robNum):
                if f_robID in self.updateRobDic:
                    if self.updateRobDic[f_robID] == e_taskID:
                        cRate  = cRate - self._robAbiLst[f_robID]
                        e_taskRobNum += 1
                else:
                    f_rob = self.robotLst[f_robID]
                    if f_rob.taskID == e_taskID:
                        cRate = cRate - self._robAbiLst[f_robID]
                        e_taskRobNum += 1
            if roadDur == 0:
                roadDur = 0.1

            dis_h_eta = 1/ (roadDur)
            if cRate >= 0 and e_taskRobNum > 0:
                h_eta = dis_h_eta * (e_taskRobNum + 1) ** self.pInd
                # h_eta = dis_h_eta
            elif cRate >= 0 and e_taskRobNum == 0:
                h_eta = dis_h_eta
            else:
                # if self.sum_ratio > 2:
                if self.sum_ratio > 2:
                    h_eta = dis_h_eta * (e_taskRobNum + 1)
                else:
                    h_eta = dis_h_eta *((1 / (e_taskRobNum + 1)) ** (self.pInd))
                # h_eta = dis_h_eta * (e_taskRobNum + 1)
                return h_eta
            return h_eta


