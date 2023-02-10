#%%Import packages
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
from scipy.stats import lognorm
#%%
#Get a list with the names of the actives that want to
#to be downloaded from yahoofinance, and initial
#and an end date and return a list of all the time
#series

def download(nombres,init,end):
    lista=[]
    for nombre in nombres:
        share = yf.download(nombre,start=init,
        end=end)
        lista.append(share)
    return lista

#%%Transform a time series from yahoo finance into
#a simpler one. Takes the open price and the date
#and remove the rest of colums. Returns it in a 
#format that can be used by prophet

def SerTrans(Ser):
    Obj = Ser['Open']
    Obj = Obj.reset_index()
    Obj.columns = ['ds','y']
    return Obj


#%%We compute the rentability as
#  the revenue or loss
#and inversor is getting when buying 
# the active at start date

def GetRent(df,inicio,fin):
    
    cut = df.loc[(df['ds']>=inicio)&(df['ds']<=fin)]
    cut['y'] = (cut['y']-cut['y'].iloc[0])/cut['y'].iloc[0]
    return cut


#%%  In order to optimize the resource
#  distribution we are going to use a Monte Carlo
#  method. We generate random data,
#  acording to the distributions
#  we have computed, and with that data
#  we compute the returnal or loss of the 
#  resource distribution.

def logResourceDist(w,LogNormDists,n):
    data = list(map(lambda d: lognorm.rvs(size=n,
                    s=d[0],loc=d[1],scale=d[2]),LogNormDists))
    df = pd.DataFrame(data)
    
    for i in range(len(w)):
        df.iloc[i,:] = df.iloc[i,:]*w[i]
    
    return(df.sum().mean())
