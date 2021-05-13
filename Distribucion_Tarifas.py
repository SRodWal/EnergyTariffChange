import pandas as pd
import numpy as np
import datetime as dt
import dateutil # Nos ayuda a agregat meses al datetime
import seaborn as sb
import matplotlib.pyplot as plt
import scipy # Ayuda a crear distribuciones y arreglos

def monthNum(num):
    return {1 : "Jan", 2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",7:"Jul",8:"Aug",9:"Sep",10: "Oct",11:"Nov",12:"Dec"}[num]

data = [ ["May de 2016",	1.4588,	3.6051,	3.9370,	2.3989,	2.2441], 
        ["July de 2016",	1.4588,	3.6051,	3.9370,	2.3989,	2.2441],
        ["November de 2016",	1.4911,	3.6848,	4.0315,	2.4603,	2.2961],
        ["March de 2017",	1.6217,	4.0073,	4.3197,	2.7159,	2.5487],
        ["May de 2017",	1.6214,	4.0065,	4.3114,	2.7068,	2.5401],
        ["August de 2017",	1.6137,	3.9871,	4.2884,	2.6914,	2.5269],
        ["December de 2017",	1.6210,	4.0051,	4.3140,	2.7133,	2.5472],
        ["March de 2018",	1.6427,	4.0588,	4.3140,	2.7133,	2.5472],
        ["June de 2018",	1.6776,	4.1450,	4.3140,	2.7299,	2.5472],
        ["September de 2018",	1.8889,	4.6671,	4.7928,	3.1611,	2.9417],
        ["December de 2018",	3.6430,	4.7404,	4.7373,	3.0883,	2.8755],
        ["March de 2019",	4.0274,	5.2406,	5.3266,	3.4056,	3.1710],
        ["July de 2019",	3.9728,	5.1696,	5.2364,	3.4006,	3.1601],
        ["October de 2019",	3.8678,	5.0330,	5.1195,	3.2387,	2.9952],
        ["January de 2020",	4.0088,	5.2164,	5.1945,	3.3617,	3.1463],
        ["May de 2020",	3.3926,	4.4147,	4.4366,	2.6824,	2.4985],
        ["July de 2020",	3.2679,	4.2524,	4.2868,	2.5040,	2.3356],
        ["October de 2020",	3.3096,	4.3066,	4.3388,	2.5619,	2.3924],
        ["January de 2021",	3.4281,	4.4608,	4.4883,	2.7093,	2.5355],
        ["April de 2021",	3.3657,	4.3796,	4.4082,	2.6437,	2.4725],
        ]
MTpot = [246.5493,246.5493,250.7668,252.3153, 252.4918,252.4920,252.6116, 
          253.4864, 257.6204, 260.6356, 262.8404, 287.7614, 295.9551, 297.2365, 
          297.9744, 304.7970, 305.6773, 310.1292, 311.3043, 313.4589]
cols = ["Fecha","Residencial (< 50 kWh)", "Residencial (> 50 kWh)", "Baja Tensión", "Media Tensión","Potencia @ MT", "Alta Tensión"]
#### Generar dataframe de tarifas
timevec = [dt.datetime(2016,5,1)]
for n in range(1,len(data)):
    timevec.append(timevec[-1]+dateutil.relativedelta.relativedelta(months=3))
datarray = np.array(data)
datarray = np.insert(datarray, 5, MTpot, axis = 1)
df = pd.DataFrame (datarray, columns = cols)
df["Fecha"]=timevec
df = df.set_index(cols[0])

#### Analizar cambios pocentuales de tarifas
datarray = np.delete(datarray, 0,1)
datarray = datarray.astype(float)
Tchlist = [(100*np.divide(np.subtract(datarray[1],datarray[0]),datarray[1])).tolist()]
for i in range(1,len(data)-1):
    trray = (100*np.divide(np.subtract(datarray[i+1],datarray[i]),datarray[i+1])).tolist()
    Tchlist.append(trray)
#Dataframe con cambios porcentuales
dft = pd.DataFrame(Tchlist, columns = cols[1:7])
#Graficar histograma de cambios porcentuales
plt.figure(num = 1, figsize = (6,4))
[sb.distplot(dft[name], rug = True, hist = False) for name in dft.columns]
plt.legend(dft.columns)
plt.show()
# Crear distribuciones
variables = cols[1:7]
normstats = []
skewstats = []
kurstats = []
normfit = []
skewfit = []
burrfit = []
disttype = ["norm", "skewnorm", "burr"]
x = np.linspace(-40,40,100) # Rango de cambios porcentuales
for v in variables:
    nstat = scipy.stats.distributions.norm.fit(dft[v]) # Normal
    skstat = scipy.stats.distributions.skewnorm.fit(dft[v]) #Skew
    kurstat = scipy.stats.burr.fit(dft[v]) #Kurtosis
    normfit.append(scipy.stats.distributions.norm.pdf(x, nstat[0], nstat[1]))
    skewfit.append(scipy.stats.distributions.skewnorm.pdf(x, skstat[0], skstat[1],skstat[2]))
    burrfit.append(scipy.stats.burr.pdf(x, kurstat[0], kurstat[1], kurstat[2], kurstat[3]))
    normstats.append(nstat)
    skewstats.append(skstat)
    kurstats.append(kurstat)

for n, name in zip(range(0, len(variables)), variables):
    plt.figure(num = n)
    plt.title(name)
    sb.distplot(dft[name], rug = True, hist = False)
    for dtype in disttype:
        kstest = scipy.stats.kstest(dft[name], dtype, normstats[n])
        print(kstest)
    plt.plot(x, normfit[n])
    plt.plot(x, skewfit[n])
    plt.plot(x, burrfit[n])
    plt.legend(["Data-Hist","Normal","Skew","Kurtosis"])
    plt.show()
    







