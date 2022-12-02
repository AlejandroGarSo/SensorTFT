import pandas as pd
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import matplotlib
import os
from mpl_toolkits import mplot3d

def getMinIni(vsum, min):
    vsum = abs(vsum)
    prev = vsum[0]
    max = False
    for i in range (len(vsum)):
        curr = vsum[i]
        if not max:
            if curr < prev:
                max = True
        else:
            if curr > prev:
                if min <= 1:
                    return i
                else:
                    min -= 1
                    max = False
        prev = curr


def getMinFin(vsum, min):
    vsum = abs(vsum)
    prev = vsum[-1]
    max = False
    for i in reversed(range(len(vsum))):
        curr = vsum[i]
        if not max:
            if curr < prev:
                max = True
        else:
            if curr > prev:
                if min <= 1:
                    return i
                else:
                    min -= 1
                    max = False
        prev = curr


#Muestra num pantalla
def draw(fileName):
    df = pd.read_csv(fileName, header=None)
    dfax = df.iloc[:, 1]
    dfax = dfax - np.mean(dfax)
    dfay = df.iloc[:, 2]
    dfay = dfay - np.mean(dfay)
    dfaz = df.iloc[:, 3]
    dfaz = dfaz - np.mean(dfaz)

    t=np.arange(0,200,0.01)
    init = 0
    fin = -1
    #print(dfaz.shape)
    #Integramos acceleración con respecto al tiempo para obtener velocidad
    #vx = integrate.cumulative_trapezoid(np.transpose(dfax),t[:len(dfax)],initial=0)
    #vy = integrate.cumulative_trapezoid(np.transpose(dfay),t[:len(dfay)],initial=0)
    #vz = integrate.cumulative_trapezoid(np.transpose(dfaz),t[:len(dfaz)],initial=0)

    vx = integrate.cumulative_trapezoid(np.transpose(dfax))
    vy = integrate.cumulative_trapezoid(np.transpose(dfay))
    vz = integrate.cumulative_trapezoid(np.transpose(dfaz))

    #'''
    vsum = vx+vy+vz
    #plt.subplot(1,2,1)
    #plt.plot(abs(vsum))
    init = getMinIni(vsum, 1)
    fin = getMinFin(vsum, 1)
    #print(init)
    #print(fin)
    #plt.axvline(x=init, color="red")
    #plt.axvline(x=fin, color="red")
    #'''

    vx = np.transpose(vx) - np.mean(vx)
    vy = np.transpose(vy) - np.mean(vy)
    vz = np.transpose(vz) - np.mean(vz)

    # Integramos velocidad con respecto al tiempo para obtener trayectoria
    x=integrate.cumulative_trapezoid(vx*np.diff(t[:len(vx)+1]),initial=0)
    y=integrate.cumulative_trapezoid(vy*np.diff(t[:len(vy)+1]),initial=0)
    z=integrate.cumulative_trapezoid(vz*np.diff(t[:len(vz)+1]),initial=0)

    x1=x-x[1]
    y1=y-y[1]
    z1=z-z[1]

    '''
    print(fileName)
    user = fileName[-7:-4]
    num =fileName[-3]
    id = fileName[-1]
    print(user, num, id)
    if not os.path.exists(r"SepData\F\tray\\"+user+r"\\"+num):
        os.makedirs(r"SepData\F\tray\\"+user+r"\\"+num)
    trayData = [np.arange(0,len(x1),1), x1, y1, z1]
    tdf = pd.DataFrame(data=trayData)
    tdf = tdf.T
    #print(tdf)
    tdf.to_csv(os.path.join(r"SepData\F\tray",user,num,id+".csv"), index=False, header=False)
    #'''


    #plt.subplot(1,2,2)
    #fig = plt.figure()
    #fig.add_subplot(111).plot(-z1[init:fin], -y1[init:fin])
    #plt.plot(-z1[:], -y1[:], label="Sin limpieza de ruido")
    #plt.plot(-z1[init:], -y1[init:], '-', label="Con limpieza de ruido inicial")
    #plt.plot(z1[init:fin], -y1[init:fin], '*-', label="Con limpieza de ruido")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(-z1[init:fin], -y1[init:fin])
    #plt.plot(z1[0:58], y1[0:58], '*')
    #ax = plt.axes(projection='3d')
    #ax.plot3D(z1[init:fin], y1[init:fin], x1[init:fin])
    #ax.view_init(15, 120)
    #'''

    return fig, ax
    #plt.show()

    '''
    plt.plot(abs(vsum), label="Vel")
    vder = np.diff(abs(vsum))
    vder2 = np.diff(abs(vsum), n=2)

    critp = np.argwhere((vder < 0.001) & (vder > -0.001))

    print(critp, vder[critp])
    for i in critp:
        plt.axvline(x=i, color='red')
    plt.axhline(y = 0, color='black')
    plt.plot(vder, label="Der1", ls=':')
    plt.plot(vder2, label="Der2")
    plt.legend()
    #'''

    plt.axis('off')
    #plt.legend()
    #plt.show()

def Draw2d():
    fig = plt.plot(-z1[init:fin], -y1[init:fin])

def Draw3d():
    ax = plt.axes(projection='3d')
    ax.plot3D(z1[init:fin], y1[init:fin], x1[init:fin])
    ax.view_init(15, 120)
