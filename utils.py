import pandas as pd
import numpy as np
import os

def flattenData(fich,size):
    df = pd.read_csv(fich, header=None)
    dfa = df.iloc[:, 1].transpose()
    dfb = df.iloc[:, 2].transpose()
    dfc = df.iloc[:, 3].transpose()
    s = size-len(dfa)
    #z = [0](s)
    z = np.zeros(shape=[s])
    #m = np.ravel([dfa, dfb, dfc])
    #m.transpose()
    m = np.concatenate((dfa,z,dfb,z,dfc,z))
    #print(type(dfa))
    #print(m)
    #print(np.shape(m))
    return m


def createMatrix(user, size):
    print(user)
    dat = np.array([])
    numList = os.listdir(user)
    for num in numList:
        print(num)
        for file in os.listdir(os.path.join(user, str(num))):
            data = os.path.join(user, str(num), str(file))
            m = flattenData(data, size)
            dat = np.append(dat,m)
            dat = np.append(dat,np.array(num))
    sh = int(np.shape(dat)[0]/(size*3+1))
    #print(sh)
    dat = np.reshape(dat,(sh,size*3+1))
    print(np.shape(dat))
    #print(dat)
    #print(labels)
    df = pd.DataFrame(dat)
    df.to_csv(user+".csv", index=False, header=False)

def getDataPred(data, size):
    return flattenData(data,size)

def getSizeLimits():
    f = open('size.txt', 'r')
    fdata = f.readlines()
    fdata = [int(s.replace("\n", "")) for s in fdata]
    fmin = min(fdata)
    fmax = max(fdata)
    f.close
    return fmin, fmax
