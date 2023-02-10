#%%Import desired functions and modules:
import datetime
from scipy.stats import lognorm
from scipy.optimize import minimize
from functionsFinance import download,SerTrans,GetRent,logResourceDist

#%%Download the data we want:
#start and end date and the names
#of the actives in yahoofinance
#The reason 4 years are being downloaded 
#is becuase we also intend to study seasonality
#This can be cut down afterwards

start = '2017-11-29'
end = '2022-11-29'
names = ["ORA.PA","AAPL","VOD.L","OXY"]
shares = list()

shares = download2(names,start,end)
#Transform the data
shares = list(map(SerTrans,shares))

#%%Now we fit the data to lognormal distribution
#We can define new start and end dates or keep
#the older ones. Here we define new ones

LastYearInit = datetime.datetime(2022,1,1)
LastYearEnd = datetime.datetime(2022,11,20)

#New rentability functions may be defined to compute rentability  
#in windows of time

Rentabilities = list(map(lambda s: GetRent(s,LastYearInit,
                LastYearEnd), shares))

#Keep only the objective column
Rents = [R[["y"]] for R in Rentabilities]

#Fit to distribution
lognormpar =list(map(lambda r:lognorm.fit(r),Rents))

#Optimize the Resource distribution:
def constraint(x):
    return 1 - sum(x)

# Define the bounds on each element of the vector
#n is number of sample for MonteCarlo method
#the more samples more accuracy in the distribution of resources
#but slower processing

n = 100
w=(0,0,0,0)
bounds = [(0, 1) for i in range(len(w))]
# Call the minimize function with the SLSQP method
result = minimize(logResourceDist, w, args=(lognormpar,n),
                method='SLSQP',
                constraints={'type': 'eq', 'fun': constraint},
                bounds=bounds)

# The optimized vector is stored in the x property of the result object
optimized_vector = result.x