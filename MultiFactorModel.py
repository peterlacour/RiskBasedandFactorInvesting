#!/bin/python3

## Title:        Quantitative Portfolio Management
## Author:       Peter la Cour
## Email:        peter.lacour@student.unisg.ch
## Place, Time:  St. Gallen, 07.03.19
## Description:
##
## Improvements:
## Last changes: Changed sharpe ratio to not annualise standard deviation

#-----------------------------------------------------------------------------#
# Loading Packages
#-----------------------------------------------------------------------------#
import  pandas              as pd
import  numpy               as np
import  datetime
import  os
from    sqlalchemy          import create_engine
import  sqlalchemy          as db
from    sqlalchemy          import update
import  seaborn             as sns
import  matplotlib.pyplot   as plt
import time

# Set plot style
plt.style.use('seaborn')
pd.options.mode.chained_assignment = None

#-----------------------------------------------------------------------------#
# Functions
#-----------------------------------------------------------------------------#

def calculate_scores( dataframe ):
    '''
    Description:
    Inputs:
    Outputs:
    '''
    scores          = dataframe.copy()
    temp            = dataframe.drop('Date', axis = 1)
    for k in scores.columns[1:]:
        for n in range(len(dataframe)):
            if np.isnan(scores[k].loc[n]) == False:
                scores[k].loc[n] = ( temp[k].loc[n] - np.nanmean(temp.loc[n,]) ) / np.nanstd(temp.loc[n,] )
    return scores



#-----------------------------------------------------------------------------#
# Body
#-----------------------------------------------------------------------------#

currentDirectory    = os.getcwd()

# get data into database
engine              = db.create_engine('sqlite:///' + currentDirectory + '/Data/Databases/SP500.db')
connection          = engine.connect()

# SP Price Data
sqlQuery            = "SELECT PERMNO FROM SP500_Prices_1990_2018;"
permno              = pd.read_sql_query(sqlQuery, con = engine)
permno              = permno.PERMNO.unique()
sqlQuery            = "SELECT * FROM SP500_Prices_1990_2018 WHERE PERMNO IN (" + str(permno[0]) + "," + str(permno[1]) + "," + str(permno[2]) + "," + str(permno[3]) + "," + str(permno[4]) + "," + str(permno[5]) + ");"
sample              = pd.read_sql_query(sqlQuery, con = engine)
connection.close()

# create list with sample permnos
permnos             = permno[:5]


# create returns dataframe - starts 1 day ahead
start = time.time()
prices  =  { 'Date': sample['date'][ 0:len( sample[ sample.PERMNO == permnos[0] ]['AdjClose']) ] }
for p in permnos:
    prices[p] = sample[ sample.PERMNO == p ][ 'AdjClose' ].values
prices  = pd.DataFrame(prices)

returns = { 'Date': sample['date'][ 1:len( sample[ sample.PERMNO == permnos[0] ]['AdjClose']) ] }
for p in permnos:
    returns[p] = [ np.log( prices[p].loc[i] / prices[p].loc[i-1] ) for i in range( 1,len(prices) ) ]
returns = pd.DataFrame( returns )
end = time.time()
runtime = end - start
print(runtime)

# momentum - create 12-month trailing returns dataframe - can't use logs - starts 252 days ahead
trailingReturns = { 'Date': sample['date'][ (252):len( sample[ sample.PERMNO == permnos[0] ]['AdjClose'])] }
for p in permnos:
    trailingReturns[p] = [ (prices[p][i] / prices[p][i-252])-1 for i in range( 252, len(prices) ) ]
trailingReturns = pd.DataFrame( trailingReturns )

# low volatility - create returns dataframe for rolling standard deviations dataframe - trailing 60 month = 252 * 5 - starts 252 * 5 days ahead
rollingStd          = { 'Date': sample['date'][(252*5):len( returns[permnos[0]])] }
start = time.time()
for p in permnos:
    rollingStd[p]   = [ returns[p][(i - 252*5 ):i].std() for i in range( 252*5, len(returns[p]) ) ]
rollingStd          = pd.DataFrame( rollingStd )
end = time.time()

runtime = end - start
print(runtime)


# calculate beta






# load SP fundamental data
engine              = db.create_engine('sqlite:///' + currentDirectory + '/Data/Databases/SP500.db')
connection          = engine.connect()
sqlQuery            = "SELECT * FROM SP500_Fundamentals_1990_2018 WHERE PERMNO IN (" + str(permno[0]) + "," + str(permno[1]) + "," + str(permno[2]) + "," + str(permno[3]) + "," + str(permno[4]) + "," + str(permno[5]) + ");"
sample              = pd.read_sql_query(sqlQuery, con = engine)
connection.close()

sample.replace(to_replace=[None], value=np.nan, inplace=True)

for i in range( len(sample['DIVYIELD']) ):
    if  type(sample['DIVYIELD'].loc[i]) == type(np.nan):
        pass
    else:
        sample['DIVYIELD'].loc[i] = float( sample['DIVYIELD'].loc[i].replace('%','') )
sample['DIVYIELD'] = pd.to_numeric(sample['DIVYIELD'])

# VALUE FACTOR - P/E, P/C, P/S, P/Div, P/B
dates               = sample[ sample.permno == permnos[0] ]['adate']

# create value factor dataframe
valueRatios         = { 'P/E': 'pe_exi', 'P/B': 'ptb', 'P/S': 'ps', 'P/C': 'pcf', 'DivYield': 'DIVYIELD'}
valueFactor         = { }
for v in valueRatios:
    valueFactor[v]  = { 'Date': dates }
    for p in permnos:
        valueFactor[v][p] = sample[ sample.permno == p ][ valueRatios[v] ].values
    valueFactor[v]  = pd.DataFrame(valueFactor[v])

valueScores         = { }
for v in valueRatios:
    print(v)
    valueScores[v] = calculate_scores( valueFactor[ v ] )

combinedValueScores = valueScores['P/E'].copy()
for p in permnos:
    for k in range(len(dates)):
        for v in valueRatios:
            combinedValueScores[p].loc[k] += valueScores[v][p][k]

combinedValueScores /= len( valueScores.keys() )


# LOW VOLATILITY
rollingStd.columns
rollingStd          = rollingStd.reset_index()
rollingStd          = rollingStd.drop( ['level_0', 'index'], axis = 1 )
lowVolScore         = calculate_scores( rollingStd )


# MOMENTUM
trailingReturns     = trailingReturns.reset_index()
trailingReturns     = trailingReturns.drop( ['level_0', 'index'], axis = 1 )
trailingReturns     = calculate_scores( trailingReturns )


# SIZE






# quality - ROA, Variability of Earnings, Cash Earings to price, trailing earnings to price












# --------------------------------------------------------------------------------------------------------
# Redundant Code
# --------------------------------------------------------------------------------------------------------

# scores test

dataframe = valueFactor[ 'P/E' ]

scores          = dataframe.copy()
temp            = dataframe.drop('Date', axis = 1)
for k in scores.columns[1:]:
    print(k)
    for n in range(len(dataframe)):
        if np.isnan(scores[k].loc[n]) == False:
            scores[k].loc[n] =  ( temp[k].loc[n] - np.nanmean(temp.loc[n,]) ) / np.nanstd(temp.loc[n,] )
scores




scores[scores.columns[1]].loc[n] - np.nanmean( temp.loc[1,] )


np.nanmean( temp.loc[1,] )

np.nanstd( temp.loc[1,] )








# calculate scores test

valueFactor['P/E']
scores = valueFactor['P/E'].copy()
for k in scores.columns:
    for n in range(len(valueFactor['P/E'])):
        scores[k][n] =  ( scores[k][n] - np.mean(valueFactor['P/E'].loc[n,]) ) / np.std(valueFactor['P/E'].loc[n,] )




valueScores['P/E']



# low vol test

scores = rollingStd.copy()
for k in rollingStd.columns[1:]:
    for n in range(len(rollingStd)):
        scores[k].loc[n] =  ( rollingStd[k][n] - np.mean(rollingStd.loc[n,]) ) / np.std(rollingStd.loc[n,] )
return scores




rollingStd[10104].loc[1]

# old returns calculation
# create returns dataframe - starts 1 day ahead
returns = { 'Date': sample['date'][ 1:len( sample[ sample.PERMNO == permnos[0] ]['AdjClose']) ] }
start = time.time()
for p in permnos:
    returns[p] = [ np.log( sample[ sample.PERMNO == p ]['AdjClose'][i] / sample[ sample.PERMNO == p ]['AdjClose'][i-1] ) for i in range( sample[ sample.PERMNO == p ].index[0] + 1, sample[ sample.PERMNO == p ].index[0] + len( sample[ sample.PERMNO == p ]['AdjClose']) ) ]
returns = pd.DataFrame( returns )
end = time.time()

runtime = end - start
print(runtime)








dataframe = valueFactor['DivYield']

scores          = dataframe.copy()
temp            = dataframe.drop('Date', axis = 1)
for k in scores.columns[1:]:
    for n in range(len(dataframe)):
        #if np.isnan(scores[k].loc[n]) == False:
        scores[k].loc[n] = ( temp[k].loc[n] - np.nanmean( temp.loc[n,])  ) / np.nanstd(temp.loc[n,] )
scores


 np.nanmean( temp.loc[n,])

valueFactor['DivYield']
