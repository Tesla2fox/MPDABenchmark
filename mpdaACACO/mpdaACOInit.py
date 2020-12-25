_INS = object
IND_ROBNUM = 0
IND_TASKNUM = 0
import mpdaACACO.mpdaGreedy
from mpdaACACO.mpdaGreedy import MPDA_Greedy,MPDA_Mean_Greedy

def mpda_init_aco(robNum):
    res = []
    for robID in range(robNum):
        res.append([])
    return  res


def init_pheromone(_ins,benchmarkName):
    h_Lst = []
    mean_greedy = MPDA_Mean_Greedy(_ins, benchmarkName)
    encode, fitness = mean_greedy.construct()
    print(encode)
    h_Lst.append(fitness)
    # exit()


    constructor = MPDA_Greedy(_ins, '_Dis')
    encode, fitness = constructor.construct()
    h_Lst.append(fitness)

    constructor = MPDA_Greedy(_ins, '_Edur')
    encode, fitness = constructor.construct()
    h_Lst.append(fitness)

    constructor = MPDA_Greedy(_ins, '_Cdur')
    encode, fitness = constructor.construct()
    h_Lst.append(fitness)

    constructor = MPDA_Greedy(_ins,'_SPP')
    encode, fitness = constructor.construct()
    h_Lst.append(fitness)

    constructor = MPDA_Greedy(_ins,'_Dvis')
    encode, fitness = constructor.construct()
    h_Lst.append(fitness)


    print('h_lst = ', h_Lst)
    min_h_fit = min(h_Lst)
    return min_h_fit

'''
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
'''
