import random
import math
import numpy as np
from mpdaInstance import  MPDAInstance
from mpdaDecodeMethod.mpdaDecode import MPDADecoder
# from mpdaTaskACO.mpdaTaskUpdater import ACO_Updater



from deap import algorithms
from deap import base
from deap import benchmarks
from deap import creator
from deap import tools
import numpy
import random
import time
from numpy.random import choice as np_choice
import copy

import os
import sys


AbsolutePath = os.path.abspath(__file__)
SuperiorCatalogue = os.path.dirname(AbsolutePath)
BaseDir = os.path.dirname(SuperiorCatalogue)

import mpdaACACO.mpdaACOInit as _init
from mpdaACACO.mpdaConstructSol import ACO_Constructor

debugBool = False

import mpdaACACO.mpdaConstructSol
mpdaACACO.mpdaConstructSol.debugBool = debugBool


def permutationSinglePointSwapACO(perm,acoSize):
    index1 = random.randint(0,acoSize - 1)
    unit1 = perm[index1]
    index2 = random.randint(0,len(perm)-1)
    unit2 = perm[index2]
    perm[index1] = unit2
    perm[index2] = unit1
    return perm


class MPDA_Task_ACO(object):
    def __init__(self, ins: MPDAInstance,benchmarkName,
                 rdSeed = 1):
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

        '''
        xxx
        '''
        self.rdSeed = rdSeed
        random.seed(rdSeed)
        np.random.seed(rdSeed)
        self.benchmarkName = benchmarkName
        self._algName = 'ac_aco'

        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual",list,
                       fitness = creator.FitnessMin, actionSeq = object)

        self.toolbox = base.Toolbox()
        self.toolbox.register("mpda_attr", _init.mpda_init_aco, self._robNum)

        self.toolbox.register("individual", tools.initIterate, creator.Individual,
                         self.toolbox.mpda_attr)

        # define the population to be a list of individuals
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        self.n_ants = self._taskNum * self._robNum
        self.n_iterations = 100
        self.decay = 0.95
        self.n_best = math.floor(self.n_ants * 1)
        self.alpha = 1
        self.beta = 1
        calTime = max(self._robNum,self._taskNum) * self._robNum * self._taskNum * 0.1
        calTime = min(29990,calTime)
        maxCalTime = calTime
        self._maxRunTime = maxCalTime
        self._robAbiSum = sum(self._robAbiLst)
        self.localSearchIndNum = 40
        self.localSearchRowNum = 3

    def run(self):
        randomSeed = self.rdSeed
        floderName = BaseDir + '//debugData//' + str(self.benchmarkName) + '//' + str(self._algName)
        folder = os.path.exists(BaseDir + '//debugData//' + str(self.benchmarkName) + '//' + str(self._algName))
        if not folder:
            os.makedirs(BaseDir + '//debugData//' + str(self.benchmarkName) + '//' + str(self._algName))
        f_con = open(
            BaseDir + '//debugData//' + str(self.benchmarkName) + '//' + str(self._algName) + '//' + 'r_' + str(
                randomSeed) + '.dat', 'w')
        save_data = BaseDir + '//debugData//' + str(self.benchmarkName) + '//' + str(self._algName) + '//' + 'r_' + str(
            randomSeed) + '.dat'
        print(save_data)
        if debugBool:
            f_con_deg = open(BaseDir + '//debugData//' + str(self.benchmarkName) + '//' + str(self._algName) + '//' + 'r_' + str(
                randomSeed) + 'debug.dat', 'w')
        self.rob_sum_Abi = np.sum(self._robAbiLst)
        print('rob_sum_abi = ', self.rob_sum_Abi)
        self.task_sum_Abi = np.sum(self._taskRateLst)
        print('task_sum_rate = ', self.task_sum_Abi)
        print('phi = ', self.task_sum_Abi/ self.rob_sum_Abi)
        min_h_fit = _init.init_pheromone(self._ins,  self.benchmarkName)

        NFE = 0
        # exit()
        self.taskPheromoneLst = [np.ones([self._taskNum, self._taskNum]) / (min_h_fit) for x in range(self._robNum)]
        self.robTaskPheromoneLst = [np.ones([self._taskNum]) / (min_h_fit) for x in range(self._robNum)]
        ngen = 7000
        NP = self.n_ants

        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", numpy.mean)
        stats.register("std", numpy.std)
        stats.register("min", numpy.min)
        stats.register("max", numpy.max)
        logbook = tools.Logbook()
        logbook.header = "gen", "min", "std", "avg", "max"
        self.hof = tools.HallOfFame(1)
        # self.eliteHof = tools.HallOfFame(self.n_ants)
        MAXNFE = self._taskNum * self._robNum * 700000
        start = time.time()

        self.pop = self.toolbox.population(n = NP)

        constructor = ACO_Constructor(self._ins, self.alpha, self.beta,
                                      self.taskPheromoneLst, self.robTaskPheromoneLst, self.rob_sum_Abi, self.task_sum_Abi )
        for gen in range(ngen):
            allAntSol = []
            for i in range(self.n_ants):
                unit = constructor.construct()
                allAntSol.append(unit)
                mpdaACACO.mpdaConstructSol.debugBool = False

            '''Statistics results'''
            for i,unit in enumerate(allAntSol):
                self.pop[i].clear()
                self.pop[i].extend(unit[0])
                self.pop[i].fitness.values = (unit[1],)
                self.pop[i].actionSeq = unit[2]

            self.hof.update(self.pop)
            # self.eliteHof.update(self.pop)

            # for ind in self.eliteHof:
            #     print(ind.fitness.values[0])
            record = stats.compile(self.pop)
            logbook.record(gen=gen, **record)
            print(logbook.stream)
            runTime = time.time() - start
            NFE = NFE + self.n_ants
            self.writeDir(f_con, record, gen,
                          NFE = NFE,
                          runTime = runTime, hofFitness= self.hof[0].fitness.values[0])
            if runTime > self._maxRunTime:
                break
            if NFE > MAXNFE:
                break

            '''
            the elite local search method 
            '''
            NFE += self._taskNum * self.localSearchIndNum
            localBase = copy.deepcopy(self.hof[0])
            localBase = []
            localBaseSize = []
            # for robID in range
            for robID in range(self._robNum):
                localBase.append(copy.deepcopy(self.hof[0][robID]))
                localBaseSize.append(len(self.hof[0][robID]))
                seq = [x for x in range(self._taskNum)]
                for unit in localBase[robID]:
                    seq.remove(unit)
                random.shuffle(seq)
                localBase[robID].extend(seq)
            decoder = MPDADecoder(self._ins)
            indLst = []
            for x in range(self._taskNum * self.localSearchIndNum):
                ind = copy.deepcopy(localBase)
                # decoder.decode(ind)
                r_step = random.randint(1, self.localSearchRowNum)
                if r_step > self._robNum:
                    r_step = self._robNum - 1
                r_stepLst = random.sample(list(range(self._robNum)), r_step)
                for rdRobID in r_stepLst:
                    # print(ind[rdRobID])
                    perm = permutationSinglePointSwapACO(ind[rdRobID], localBaseSize[rdRobID])
                    ind[rdRobID] = perm
                    # print(ind[rdRobID])
                # indLst.append(ind)
                pb, _actSeq = decoder.decode(ind)
                fitness = decoder.calMakespan()
                indLst.append((_actSeq, fitness))
            # minlInd = min(lIndLst, key=lambda x: x.fitness.values[0])
            minInd = min(indLst, key=lambda x: x[1])
            if minInd[1] < self.hof[0].fitness.values[0]:
                perm = minInd[0].convert2MultiPerm(self._ins._robNum)
                for robID in range(self._robNum):
                    self.hof[0][robID] = perm[robID]
                self.hof[0].fitness.values = (minInd[1],)

            for robID in range(self._robNum):
                self.taskPheromoneLst[robID] = self.taskPheromoneLst[robID] * self.decay
                self.robTaskPheromoneLst[robID] = self.robTaskPheromoneLst[robID] * self.decay
            self.spread_pheronome(allAntSol,self.n_best)

            if debugBool:
                f_con_deg.write(str(gen) + '  ' +str(self.robTaskPheromoneLst[0]) + '\n')
                f_con_deg.flush()
        end = time.time()
        runTime = end - start
        print('runTime', runTime)
        print('hof = ', self.hof[0])
        print('hofFitness = ', self.hof[0].fitness.values[0])
        f_con.write(str(self.hof[0]) + '\n')
        f_con.write('min  ' + str(self.hof[0].fitness.values[0]) + '\n')
        encode = np.ones([self._robNum, self._taskNum], dtype=int) * -1
        robEncodeInd = [0 for x in range(self._robNum)]
        for robID, seq in enumerate(self.hof[0]):
            for pos, taskID in enumerate(seq):
                encode[robID][pos] = taskID
        gaHof = []
        for unit in encode:
            gaHof.extend(unit)
        f_con.write('gaMin ' + str(gaHof) + '\n')
        print('gaMin = ', gaHof)
        f_con.write('runTime ' + str(runTime) + '\n')
        # if self.selectedMechanism == '_Limit':
        #     print('limited = ', self.limitedNum)
        f_con.close()

    def spread_pheronome(self, all_ant_sols, n_best):
        sorted_ant_sols = sorted(all_ant_sols, key=lambda x: x[1])
        for ant_sol, fitness,x in sorted_ant_sols[:n_best]:
            for robID in range(self._robNum):
                seq = ant_sol [robID]
                self.robTaskPheromoneLst[robID][seq[0]] +=  1 / (fitness * self._robNum *self._taskNum)
                for i in range(len(seq) - 1):
                    self.taskPheromoneLst[robID][seq[i]][seq[i + 1]] += 1 / (fitness * self._robNum *self._taskNum)
        if True:
            encode = self.hof[0]
            for robID in range(self._robNum):
                seq = encode[robID]
                if len(seq) != 0:
                    self.robTaskPheromoneLst[robID][seq[0]] +=  1 / (self.hof[0].fitness.values[0] * self._robNum *self._taskNum)
                for i in range(len(seq) - 1):
                        self.taskPheromoneLst[robID][seq[i]][seq[i + 1]] += 1 / (self.hof[0].fitness.values[0] * self._robNum * self._taskNum)


    def localSearchBaseInd(self,baseInd:[],baseFitness):
        localBase = copy.deepcopy(baseInd)
        localBaseSize = []
        # for robID in range
        for robID in range(self._robNum):
            localBase.append(copy.deepcopy(baseInd[robID]))
            localBaseSize.append(len(baseInd[robID]))
            seq = [x for x in range(self._taskNum)]
            for unit in localBase[robID]:
                seq.remove(unit)
            random.shuffle(seq)
            localBase[robID].extend(seq)
        decoder = MPDADecoder(self._ins)
        indLst = []
        for x in range(self._taskNum * self.localSearchIndNum):
            ind = copy.deepcopy(localBase)
            # decoder.decode(ind)
            r_step = random.randint(1, self.localSearchRowNum)
            if r_step > self._robNum:
                r_step = self._robNum - 1
            r_stepLst = random.sample(list(range(self._robNum)), r_step)
            for rdRobID in r_stepLst:
                # print(ind[rdRobID])
                perm = permutationSinglePointSwapACO(ind[rdRobID], localBaseSize[rdRobID])
                ind[rdRobID] = perm
                # print(ind[rdRobID])
            # indLst.append(ind)
            pb, _actSeq = decoder.decode(ind)
            fitness = decoder.calMakespan()
            indLst.append((_actSeq, fitness))
        # minlInd = min(lIndLst, key=lambda x: x.fitness.values[0])
        resInd = [[] for x in range(self._robNum)]
        resFitness = baseFitness
        minInd = min(indLst, key=lambda x: x[1])
        if minInd[1] < baseFitness:
            perm = minInd[0].convert2MultiPerm(self._ins._robNum)
            for robID in range(self._robNum):
                resInd[robID] = perm[robID]
            resFitness = minInd[1]
        if resFitness < baseFitness:
            return  True,resInd, resFitness
        else:
            return  False,None,None

    def writeDir(self, f_con, RecordDic, gen, NFE,runTime,hofFitness):
        f_con.write('gen '+ str(gen) + ' ')
        f_con.write(str(NFE) + ' ')
        f_con.write(str(runTime) + ' ')
        f_con.write(str(RecordDic['avg']) + ' ')
        f_con.write(str(RecordDic['std']) + ' ')
        f_con.write(str(RecordDic['min']) + ' ')
        f_con.write(str(RecordDic['max']) + ' ')
        f_con.write(str(hofFitness) + '\n')
        f_con.flush()
