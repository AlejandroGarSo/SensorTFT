import pandas as pd
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import os
import glob
import pnn
import trayectoryDraw
import cutter
import ui
import utils

dataDir = r"RawData\B"
sepDir = r"SepData"
type = "tray"
cutpoint = 0.3
timer = .5
timer2 = 20
timer3 = 50
modelfile = r"model.h5"

#trayectoryDraw.draw(r"SepData/F/accel/D17/3/a")

w = ui.initUI()
model = pnn.loadModel(modelfile)
ui.manageUI(w, model)

#cutter.getFilesLoose(dataDir, sepDir, cutpoint, timer, "accel")
#min, max = utils.getSizeLimits()
#model = pnn.createModel(max)


'''
df = pd.read_csv(r"RawData\A\D10h.csv", header=None)
dfax = df.iloc[:, 1]
dfay = df.iloc[:, 2]
dfaz = df.iloc[:, 3]
dfsum = np.add(dfax, np.add(dfay, dfaz))
x = np.arange(0., len(dfsum) / 100, 0.01)
if x[-1] == len(dfsum)/100:
    x = x[:-1]
plt.plot(x, dfsum, label="Aceleración")
plt.xlabel("Tiempo(s)")
plt.ylabel("Aceleración(m/s\u00b2)")
plt.show()
'''
'''
df = pd.read_csv(r"SepData/F/accel/D17/3/d", header=None)
dfax = df.iloc[:, 1]
dfay = df.iloc[:, 2]
dfaz = df.iloc[:, 3]
vx = integrate.cumulative_trapezoid(np.transpose(dfax))
vy = integrate.cumulative_trapezoid(np.transpose(dfay))
vz = integrate.cumulative_trapezoid(np.transpose(dfaz))
vsum = vx+vy+vz
plt.plot(vsum)
df = pd.read_csv(r"SepData/F/accel/D17/3/e", header=None)
dfax = df.iloc[:, 1]
dfay = df.iloc[:, 2]
dfaz = df.iloc[:, 3]
vx = integrate.cumulative_trapezoid(np.transpose(dfax))
vy = integrate.cumulative_trapezoid(np.transpose(dfay))
vz = integrate.cumulative_trapezoid(np.transpose(dfaz))
vsum = vx+vy+vz
plt.plot(vsum)
plt.ylabel("Velocidad (m/s)")
plt.xlabel("Tiempo(1*10-2 s)")
plt.show()
'''

'''
for file in os.listdir(r"SepData/F/accel/D17/3"):
    df = pd.read_csv(os.path.join(r"SepData/F/accel/D17/3",file), header=None)
    dfax = df.iloc[:, 1]
    dfay = df.iloc[:, 2]
    dfaz = df.iloc[:, 3]
    vx = integrate.cumulative_trapezoid(np.transpose(dfax))
    vy = integrate.cumulative_trapezoid(np.transpose(dfay))
    vz = integrate.cumulative_trapezoid(np.transpose(dfaz))
    vsum = vx + vy + vz
    np.abs(vsum, vsum)
    plt.plot(vsum)
plt.ylabel("Velocidad (m/s)")
plt.xlabel("Tiempo(1*10-2 s)")
plt.show()'''

'''
#f = open('size.txt','w')
for user in os.listdir(os.path.join(sepDir, type)):
    if os.path.isdir(os.path.join(sepDir, type, user)):
        user = os.path.join(sepDir, type, user)
        #for num in os.listdir(user):
            #num = os.path.join(user, num)
            #for file in os.listdir(num):
                #df = pd.read_csv(os.path.join(num,file))
                #f.write(str(len(df.iloc[:,0])))
                #f.write("\n")
                #print(file)
                #trayectoryDraw.draw(os.path.join(num, file))
        cutter.createMatrix(user, 854)
        #'''
'''
trayectoryDraw.draw(r"SepData\F\accel\D17\3\a")
trayectoryDraw.draw(r"SepData\F\accel\D17\3\b")
trayectoryDraw.draw(r"SepData\F\accel\D17\3\c")
trayectoryDraw.draw(r"SepData\F\accel\D17\3\d")
trayectoryDraw.draw(r"SepData\F\accel\D17\3\e")
trayectoryDraw.draw(r"SepData\F\accel\D17\3\f")
trayectoryDraw.draw(r"SepData\F\accel\D17\3\g")
trayectoryDraw.draw(r"SepData\F\accel\D17\3\h")
trayectoryDraw.draw(r"SepData\F\accel\D17\3\i")
trayectoryDraw.draw(r"SepData\F\accel\D17\3\j")
trayectoryDraw.draw(r"SepData\F\accel\D17\3\k")'''
#f.close()
'''for user in os.listdir(r"SepData\F\accel"):
    u = user
    user = os.path.join(r"SepData\F\accel", user)
    if not os.path.isdir(user):
        continue
    for num in os.listdir(user):
        n = num
        num = os.path.join(user, num)
        fig, axs = plt.subplots(3, 1)
        for file in os.listdir(num):
            file = os.path.join(num,file)
            df = pd.read_csv(file, header=None)
            dfax = df.iloc[:, 1]
            dfay = df.iloc[:, 2]
            dfaz = df.iloc[:, 3]
            vx = integrate.cumulative_trapezoid(np.transpose(dfax))
            vy = integrate.cumulative_trapezoid(np.transpose(dfay))
            vz = integrate.cumulative_trapezoid(np.transpose(dfaz))
            axs[0].plot(vx)
            axs[1].plot(vy)
            axs[2].plot(vz)
        fig.suptitle(num)
        plt.savefig(r"Images\velcmpA\\" + n + u)'''
'''
files = os.path.join(r"SepData\tray", 'D**.csv')
files = glob.glob(files)
#print([pd.read_csv(f).shape for f in files])
dfj = pd.concat([pd.read_csv(f, header=None) for f in files], ignore_index=True)
#print(dfj.shape)
dfj.to_csv(r"SepData\tray.csv", index=False, header=False)
#'''
'''
#df = pd.read_csv(r"SepData\accel\D16\3", header=None)
for user in os.listdir(os.path.join(sepDir, "accel")):
    if os.path.isdir(os.path.join(sepDir, "accel", user)):
        u = user
        user = os.path.join(sepDir, "accel", user)
        for num in os.listdir(user):
            n = num
            num = os.path.join(user, num)
            for file in os.listdir(num):
                df = pd.read_csv(os.path.join(num,file))
                trayectoryDraw.draw(os.path.join(num, file))
            plt.title(num)
            plt.savefig(r"Images\\" + n + u)
            plt.clf()
#trayectoryDraw.draw(r"SepData\accel\D16\3\d")
'''

'''
dfax = df.iloc[:, 1]
dfax = dfax - np.mean(dfax)
dfay = df.iloc[:, 2]
dfay = dfay - np.mean(dfay)
dfaz = df.iloc[:, 3]
dfaz = dfaz - np.mean(dfaz)

vx = integrate.cumulative_trapezoid(np.transpose(dfax))
vy = integrate.cumulative_trapezoid(np.transpose(dfay))
vz = integrate.cumulative_trapezoid(np.transpose(dfaz))

vsum = vx+vy+vz
#plt.plot(dfax+dfay+dfaz)
#plt.show()

'''
'''
#df = pd.read_csv(r"Dato1.csv", header=None)
df = pd.read_csv(r"RawData\B\D13d.csv", header=None)
df1 = df.copy()
dir_list = os.listdir(dataDir)
numFich = 0
timer = .5
initime = -1
line = True
#with pd.option_context('display.max_rows', None, 'display.max_columns', None):
#    print(df.head())
'''
#for data in dir_list:
#    getFiles(data, cutpoint, timer2)
'''

#fig, axs = plt.subplots(1, 2)
#for file in os.listdir(r"RawData\B"):
#df1 = pd.read_csv(r"RawData\B\\"+file)
dfa = df1.iloc[:, 1]
np.abs(dfa, dfa)
dfb = df1.iloc[:, 2]
np.abs(dfb, dfb)
dfc = df1.iloc[:, 3]
np.abs(dfc, dfc)
dfsum = np.add(dfa,np.add(dfb,dfc))
#plt.plot(dfsum)
plt.plot(np.arange(0.,len(dfsum)/100,0.01), dfsum, label="Aceleración")
#plt.axvline(x=1050, color="red")
    #plt.title(file)
plt.xlabel("Tiempo(s)")
plt.ylabel("Aceleración(m/s\u00b2)")
plt.legend()
plt.show()
#out = df.iloc[752:1050, :]
#out.to_csv(r"SepData\F\accel\D15\2\e", index=False, header=False)
#out = df.iloc[1050:1344, :]
#out.to_csv(r"SepData\F\accel\D15\3\e", index=False, header=False)
#out = df.iloc[3150:3512, :]
#out.to_csv(r"SepData\F\accel\D10\8\j", index=False, header=False)
#'''

'''
axs[0].plot(dfsum)
dfa = df1.iloc[:, 7]
np.abs(dfa, dfa)
dfb = df1.iloc[:, 8]
np.abs(dfb, dfb)
dfc = df1.iloc[:, 9]
np.abs(dfc, dfc)
dfsum = np.add(dfa,np.add(dfb,dfc))
#plt.plot(dfsum)
axs[1].plot(dfsum)
plt.show()
#'''
'''
order = 6
fs = 12.0       # sample rate, Hz
cutoff = 3.667  # desired cutoff frequency of the filter, Hz

y = butter_lowpass_filter(dfsum, cutoff, fs, order)
plt.plot(y)
#'''

'''
#Muestra datos pantalla
fig, axs = plt.subplots(4, 3)
fig.subplots_adjust(hspace = 0.7, wspace=0.4)
titles = ["Acelerómetro X", "Acelerómetro Y","Acelerómetro Z",
          "Giroscopio X", "Giroscopio Y", "Giroscopio Z",
          "Magnetómetro X", "Magnetómetro Y", "Magnetómetro Z",
          "Yaw", "Roll", "Pitch"]
yaxis = ["Aceleración (m/s\u00b2)", "Velocidad angular (rad/s)", "Fuerza gravitacional (T)", "Rotación(º)"]
#print(df1.get[0])
for i in range(0, 4):
    for j in range(0, 3):
        dfa = df1.iloc[:, 3*i+j+1]
        np.abs(dfa, dfa)
        for k in range(len(dfa)):
            if dfa[k] < cutpoint:
                #if initime == -1:
                initime = df1[0][k]
                #else:
                #df1[0][k] - initime >= timer and
                if line:
                    axs[i,j].axvline(x=k/100, color="red")
                    line = False
            else:
                #initime = -1
                line = True
        axs[i, j].plot(np.arange(0.,len(dfa)/100,0.01), dfa)
        axs[i, j].set_title(titles[3*i+j])
        axs[i, j].set_xlabel("Tiempo(s)")
        axs[i, j].set_ylabel(yaxis[i])
        #axs[i, j].rotate_ylabel(45)
plt.show()
#'''

'''
#Muestra datos pantalla
fig, axs = plt.subplots(3)
fig.subplots_adjust(hspace = 0.7, wspace=0.4)
titles = ["Acelerómetro X", "Acelerómetro Y","Acelerómetro Z",
          "Giroscopio X", "Giroscopio Y", "Giroscopio Z",
          "Magnetómetro X", "Magnetómetro Y", "Magnetómetro Z",
          "Yaw", "Roll", "Pitch"]
yaxis = ["Aceleración (m/s\u00b2)", "Velocidad angular (rad/s)", "Fuerza gravitacional (T)", "Rotación(º)"]
#print(df1.get[0])
i = 3
for j in range(0, 3):
    dfa = df1.iloc[:, 3*i+j+1]
    np.abs(dfa, dfa)
    for k in range(len(dfa)):
        if dfa[k] < cutpoint:
            if initime == -1:
                initime = df1[0][k]
            else:
                if df1[0][k] - initime >= timer and line:
                    axs[j].axvline(x=k/100, color="red")
                    line = False
        else:
            initime = -1
            line = True
    axs[j].plot(np.arange(0.,len(dfa)/100,0.01), dfa)
    axs[j].set_title(titles[3*i+j])
    axs[j].set_xlabel("Tiempo(s)")
    axs[j].set_ylabel(yaxis[i])
    #axs[j].rotate_ylabel(45)
plt.show()
#'''

'''
#Corte V1
dfa = df1.iloc[:, 1]
np.abs(dfa, dfa)
dfb = df1.iloc[:, 2]
np.abs(dfb, dfb)
dfc = df1.iloc[:, 3]
np.abs(dfc, dfc)
prevFich = 0
c = False
dfsum = np.add(dfa,np.add(dfb,dfc))
print(df1)
print(df[0][0])
for k in range(len(dfa)):
    #print(k)
    if dfsum[k] <= cutpoint:
        if initime == -1:
            #print(df1.shape)
            initime = df[0][k]
        else:
            #print(df1[0][k] - initime)
            if df1[0][k] - initime >= timer and line:
                out = df.iloc[prevFich:k, :]
                prevFich = k
                nameFich = r'SepData\\'+str(numFich)+'.csv'
                print(k)
                if numFich >= 0:
                    #out.to_csv(nameFich, index=False, header=False)
                    numFich = numFich+1
                if c:
                    plt.axvline(x=k/100, color="red")
                else:
                    plt.axvline(x=k / 100, color="red", label="Punto de corte")
                    c = True
                line = False
    else:
        initime = -1
        line = True
plt.plot(np.arange(0.,len(dfsum)/100,0.01), dfsum, label="Aceleración")
plt.xlabel("Tiempo(s)")
plt.ylabel("Aceleración(m/s\u00b2)")
plt.legend()
plt.show()
#'''


'''
#Corte V2
#plt.plot(df1.iloc[:,1:4])

dfa = df1.iloc[:, 1]
np.abs(dfa, dfa)
dfb = df1.iloc[:, 2]
np.abs(dfb, dfb)
dfc = df1.iloc[:, 3]
np.abs(dfc, dfc)
dfsum = np.add(dfa,np.add(dfb,dfc))
#dfsum = dfsum/3

#plt.plot(dfsum, color ="black")

#smdat = abs(dfsum) - abs(np.mean(dfsum))
smdat = dfsum - np.mean(dfsum)
indmin = np.argwhere(dfsum.to_numpy() < cutpoint)
inicio = indmin[0]
final = []
for i in range(1,len(indmin)):
    if (indmin[i] - indmin[i-1]) > timer2:
        inicio = np.append(inicio, indmin[i])
        final = np.append(final, indmin[i-1])
final = np.append(final, indmin[-1])
c = False
#print(inicio)
#print(final)
rem = []
for k in range(len(inicio)-1):
    ini = int(final[k])
    fin = int(inicio[k])
    #print(ini, fin)
    if ini-fin < timer3:
        #print("D")
        rem.append(k)
inicio = np.delete(inicio, rem)
final = np.delete(final, rem)

#print(inicio)
#print(final)

for k in range(len(inicio)-1):
    ini = int(final[k])
    fin = int(inicio[k+1])
    #print(ini, fin)
    out = df.iloc[ini:fin, :]
    nameFich = r'SepData\\' + str(k) + '.csv'
    #out.to_csv(nameFich, index=False, header=False)
    if c:
        plt.axvline(x=ini/100, color="red")
        plt.axvline(x=fin/100, color="green")
    else:
        plt.axvline(x=ini / 100, color="red", label="Punto de corte inicial")
        plt.axvline(x=fin / 100, color="green", label="Punto de corte final")
        c = True
plt.plot(np.arange(0.,len(dfsum)/100,0.01), dfsum, label="Aceleración")
plt.xlabel("Tiempo(s)")
plt.ylabel("Aceleración(m/s\u00b2)")
plt.legend()
plt.show()

#'''

'''  
for l in inicio:
    plt.axvline(x=l, color="red")
for l in final:
    plt.axvline(x=l, color="green")
plt.plot(dfsum)

#'''

'''
for k in range(len(dfa)):
        if initime == -1:
            initime = df1[0][k]
        else:
            if line:
                if numFich >= 0:
                    print(inicio, final)
                    out = df1.iloc[prevFich:k, :]
                    prevFich = k
                    nameFich = r'SepData\\'+str(numFich)+'.csv'
                    out.to_csv(nameFich, index=False, header=False)
                numFich = numFich+1
                plt.axvline(x=inicio, color="red")
                plt.axvline(x=final, color="green")
                line = False
    else:
        line = True
'''



'''
#Muestra datos fichero separado
df2 = pd.read_csv(r"SepData\0.csv", header=None)
fig, axs = plt.subplots(3, 3)
for i in range(0, 3):
    for j in range(0, 3):
        dfa = df2.iloc[:, 3*i+j+1]
        np.abs(dfa, dfa)
        for k in range(len(dfa)):
            if dfa[k] < cutpoint:
                if initime == -1:
                    initime = df2[0][k]
                else:
                    if df2[0][k] - initime >= timer and line:
                        axs[i,j].axvline(x=k, color="red")
                        line = False
            else:
                initime = -1
                line = True
        axs[i, j].plot(dfa)
'''

'''
plt.plot(abs(vsum), label="Velocidad")
vder = np.diff(abs(vsum))
vder2 = np.diff(abs(vsum), n=2)

critp = np.argwhere((vder < 0.001) & (vder > -0.001))

print(critp, vder[critp])
#for i in critp:
    #plt.axvline(x=i, color='red')
#plt.axhline(y = 0, color='black')
plt.plot(vder, label="Derivada primera", ls=':')
plt.plot(vder2, label="Derivada segunda")
plt.legend()
#'''

#plt.show()

