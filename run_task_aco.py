import os
import sys

'''
'''

from mpdaACACO import MPDA_Task_ACO
from mpdaInstance import MPDAInstance


if __name__ == '__main__':
    print('begin to run task ACO\n')
    ins = MPDAInstance()
    insConfDir = './/staticMpdaBenchmarkSet//'
    print(sys.argv)
    print(len(sys.argv))
    if len(sys.argv) == 14:
        pass
    elif len(sys.argv) == 1:
        benchmarkName = 'M_20_20_0.97'
        randomSeed = 11
    else:
        raise Exception('something wrong on the sys.argv')
        pass
    print(sys.argv)
    ins.loadCfg(fileName=insConfDir + benchmarkName + '.txt')
    print(benchmarkName)
    mpda_as = MPDA_Task_ACO(ins, benchmarkName=benchmarkName, rdSeed = randomSeed)
    mpda_as.run()
