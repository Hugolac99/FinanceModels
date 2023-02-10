#%%
from prophet.diagnostics import cross_validation
import itertools
import numpy as np
import pandas as pd
from prophet.diagnostics import performance_metrics
from prophet import Prophet
#%%
#Fit time series models from finance data, including Covid
#Data using prohpet. This allows to get information
#about it´s trend and seasonality.
#Covid affected actives differntly, that´s why this function
#fits and test using cross validation differnt models
#The objective of this is to define a grid with different set
#of parameters and then use the cross validation implemented
#In prophet to test the best models. The Grid must include
#yearly seasonality fourier parameters, in some casses may include
#weekly and have to diferentiate two cases. One in which COVID
#is treated as a holiday and another in wich we specify the number
#of change points within the grid
#%%
def ValidateProphet(df):
    
    #Model ignoring COVID
    param_grid1 = {  
    'yearly_seasonality': [5,10,15,20,25]
    }
    cutoff1 = pd.to_datetime(['2019-02-24','2022-09-20'])
    # Generate all combinations of parameters
    all_params = [dict(zip(param_grid1.keys(), v))\
         for v in itertools.product(*param_grid1.values())]
    
    #Store the RMSEs for each params here
    rmses1 = []  
    # Use cross validation to evaluate all parameters
    for params in all_params:
        
        # Fit model with given params
        m = Prophet(**params).fit(df)  
        df_cv = cross_validation(m, cutoffs=cutoff1,
             horizon='30 days')

        df_p = performance_metrics(df_cv, rolling_window=1)
        rmses1.append(df_p['rmse'].values[0])
    
    # Find the best parameters
    tuning_results = pd.DataFrame(all_params)
    tuning_results['rmse'] = rmses1
    print(tuning_results)
    BestGrid1 = tuning_results.loc[tuning_results['rmse']==
                tuning_results['rmse'].min()]
    #CrossValidate model with grid1 using Covid as holidays

    #Store the RMSEs for each params here
    rmsescov = [] 

    #Period with covid
    covlock = pd.DataFrame({
    'holiday': 'superbowl',
    'ds': pd.to_datetime(['2020-02-24']),
    'lower_window': 0,
    'upper_window': 24,
    })

    # Use cross validation to evaluate all parameters
    for params in all_params:
        
        # Fit model with given params
        m = Prophet(**params, holidays=covlock).fit(df)  
        df_cv = cross_validation(m, cutoffs=cutoff1,
             horizon='30 days')

        df_p = performance_metrics(df_cv, rolling_window=1)
        rmsescov.append(df_p['rmse'].values[0])


    # Find the best parameters
    tuning_resultscov = pd.DataFrame(all_params)
    tuning_resultscov['rmse'] = rmsescov
    BestGridcov = tuning_resultscov.loc[tuning_resultscov['rmse']==
                tuning_resultscov['rmse'].min()]

    #CrossValidate model with grid2

    param_grid2 = {  
    #'yearly_seasonality': [5,10,15,20,25],
    'seasonality_prior_scale': [0.01,0.1,1,3,5,8,10],
    'n_changepoints': [0,1,2,3,4,5,6]
    }
    #param_grid2 = {  
    #'seasonality_prior_scale': [0.01,0.1,1,3,5,8,10],
    #'changepoint_prior_scale': [0.001,0.01,0.1,0.2,0.3,0.5]
    #}

    # Generate all combinations of parameters
    all_params2 = [dict(zip(param_grid2.keys(), v))\
         for v in itertools.product(*param_grid2.values())]
    
    #Store the RMSEs for each params here
    rmses2 = []  


    # Use cross validation to evaluate all parameters
    for params in all_params2:
        
        # Fit model with given params
        m = Prophet(**params).fit(df)  
        df_cv = cross_validation(m, cutoffs=cutoff1,
             horizon='30 days')

        df_p = performance_metrics(df_cv, rolling_window=1)
        rmses2.append(df_p['rmse'].values[0])

    
    # Find the best parameters
    tuning_results2 = pd.DataFrame(all_params2)
    tuning_results2['rmse'] = rmses2
    print(tuning_results)
    
    BestGrid2 = tuning_results2.loc[tuning_results2['rmse']==
                tuning_results2['rmse'].min()]

    rmsesBest = [float(BestGrid1['rmse']),
    float(BestGridcov['rmse']),float(BestGrid2['rmse'])]

    index = rmsesBest.index(min(rmsesBest))

    if index == 0:
        m = Prophet(yearly_seasonality =
            float(BestGrid1['yearly_seasonality'])).fit(df)

    if index == 1:
    
        m = Prophet(yearly_seasonality =
            float(BestGridcov['yearly_seasonality']),
            holidays=covlock).fit(df)

    if index == 2:

        m = Prophet(seasonality_prior_scale=
        float(BestGrid2['seasonality_prior_scale']),
        n_changepoints=
        int(BestGrid2['n_changepoints'])).fit(df)

    return m