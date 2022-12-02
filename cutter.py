import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

def fillZeros(fich, sizeMax):
    df = pd.read_csv(fich, header=None)
    size = sizeMax - len(df)
    df1 = df.values.tolist()
    df1.extend([0]*size)
    print(type(df1))
    df = pd.DataFrame(df1)
    df.to_csv(fich, index=False, header=False)
    print(len(df), size, sizeMax)
    print(df)


def cut(data, cutpoint, time, dataDir, sepDir, f, type):
    fich = os.path.join(dataDir,data)
    user = data[:-5]
    file = data[-5]
    #print(fich)
    df = pd.read_csv(fich, header=None)
    df1 = df.copy()
    timer3 = 50

    # Corte V2

    dfa = df1.iloc[:, 1]
    np.abs(dfa, dfa)
    dfb = df1.iloc[:, 2]
    np.abs(dfb, dfb)
    dfc = df1.iloc[:, 3]
    np.abs(dfc, dfc)
    dfsum = np.add(dfa, np.add(dfb, dfc))

    indmin = np.argwhere(dfsum.to_numpy() < cutpoint)
    inicio = indmin[0]
    final = []

    for i in range(1, len(indmin)):
        if (indmin[i] - indmin[i - 1]) > time:
            inicio = np.append(inicio, indmin[i])
            final = np.append(final, indmin[i - 1])
    final = np.append(final, indmin[-1])

    rem = []
    for k in range(len(inicio) - 1):
        ini = int(final[k])
        fin = int(inicio[k])
        # print(ini, fin)
        if ini - fin < timer3:
            rem.append(k)
    inicio = np.delete(inicio, rem)
    final = np.delete(final, rem)

    for k in range(len(inicio)-1):
        ini = int(final[k])
        fin = int(inicio[k + 1])
        #print(ini, fin)
        #f.write(str(fin-ini))
        #f.write("\n")
        out = df.iloc[ini:fin, :]
        nameFich = sepDir + r'\\' + type + r'\\' + user + r'\\' + str(k) + r'\\'+ file+".csv"
        if not os.path.exists(sepDir + r'\\' + type + r'\\' + user + r'\\' + str(k)):
            os.makedirs(sepDir + r'\\' + type + r'\\' + user + r'\\' + str(k))
        print(nameFich)
        #out.to_csv(nameFich, index=False, header=False)
        plt.axvline(x=ini, color="red")
        plt.axvline(x=fin, color="green")
    plt.plot(dfsum)
    #plt.title(nameFich)
    plt.show()

def cutLoose(data, cutpoint, time, dataDir, sepDir, f, type):
    fich = dataDir + r'\\' + data
    initime = -1
    user = data[:-5]
    file = data[-5]
    numFich = 0
    print(fich)
    df = pd.read_csv(fich, header=None)
    df1 = df.copy()
    line = True

    # Corte V1
    dfa = df1.iloc[:, 1]
    np.abs(dfa, dfa)
    dfb = df1.iloc[:, 2]
    np.abs(dfb, dfb)
    dfc = df1.iloc[:, 3]
    np.abs(dfc, dfc)
    prevFich = 0
    c = False
    dfsum = np.add(dfa, np.add(dfb, dfc))
    #print(df1)
    #print(df[0][0])
    for k in range(len(dfa)):
        #print(k)
        if dfsum[k] <= cutpoint:
            if initime == -1:
                # print(df1.shape)
                initime = df[0][k]
            else:
                # print(df1[0][k] - initime)
                if df1[0][k] - initime >= time and line:
                    out = df.iloc[prevFich:k, :]
                    #f.write(str(prevFich-k))
                    #f.write("\n")
                    prevFich = k
                    nameFich = os.path.join(sepDir, type, user, str(numFich), file+".csv")
                    if not os.path.exists(os.path.join(sepDir, type, user, str(numFich))):
                        os.makedirs(os.path.join(sepDir, type, user, str(numFich)))
                    out.to_csv(nameFich, index=False, header=False)
                    numFich = numFich + 1
                    if c:
                        plt.axvline(x=k / 100, color="red")
                    else:
                        plt.axvline(x=k / 100, color="red", label="Punto de corte")
                        c = True
                    line = False
        else:
            initime = -1
            line = True
    out = df.iloc[prevFich:, :]
    nameFich = os.path.join(sepDir, type, user, str(numFich), file)
    if not os.path.exists(os.path.join(sepDir, type, user, str(numFich))):
        os.makedirs(os.path.join(sepDir, type, user, str(numFich)))
    #out.to_csv(nameFich, index=False, header=False)
    print(numFich)
    x = np.arange(0., len(dfsum) / 100, 0.01)
    if x[-1] == len(dfsum)/100:
        x = x[:-1]
    #print(np.arange(0., len(dfsum) / 100, 0.01)[-1])
    plt.plot(x, dfsum, label="Aceleración")
    #plt.xlabel("Tiempo(s)")
    #plt.ylabel("Aceleración(m/s\u00b2)")
    #plt.legend()
    #plt.savefig(r"Images\cutA2\\" + user + file)
    plt.show()


def getFiles(dataDir, sepDir, cutpoint, timer, type):

    dir_list = os.listdir(dataDir)
    f = open('size.txt', 'w')
    for data in dir_list:
        cut(data, cutpoint, timer, dataDir, sepDir, f, type)
    f.close

def getFilesLoose(dataDir, sepDir, cutpoint, timer, type):

    dir_list = os.listdir(dataDir)
    f = open('size.txt', 'w')
    for data in dir_list:
        cutLoose(data, cutpoint, timer, dataDir, sepDir, f, type)
    f.close

