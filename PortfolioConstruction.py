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
import os
from sqlalchemy import create_engine
import sqlalchemy as db
from sqlalchemy import update
import seaborn as sns
import matplotlib.pyplot as plt

# Set plot style
plt.style.use('seaborn')

#-----------------------------------------------------------------------------#
# Functions
#-----------------------------------------------------------------------------#

# Portfolio functions

def _equal_weight_portfolio( asset_returns ):
    '''
    Description:
    Inputs:
    Outputs:        portfolio weights
    '''
    N               = len( asset_returns.columns )
    weights         = np.array([-999.99]*N)
    weights.fill(1/N)
    return weights.T

def _minimum_variance_portfolio( asset_returns ):
    '''
    Description:    Calculates weights of the min variance portfolio with:
                    ( inverse of covariance matrix * vector of ones ) / ( transposed vector of ones * inverse of covariance matrix * vector of ones )
                    Vector of ones has size N x 1 where N is the number of assets
    Inputs:         ARRAY of returns ( annualised )
    Outputs:        portfolio weights
    '''
    N               = len( asset_returns.columns )
    covariance      = asset_returns.cov().values * 52
    weights         = ( np.linalg.inv( covariance  ) @ np.ones(N) ) /  ( np.ones(N).T @ np.linalg.inv( covariance ) @ np.ones(N) )

    return weights.T



def _maximum_diversification_portfolio( asset_returns, standard_deviations ):
    '''
    Description:    should this be long only ?
    Inputs:
    Outputs:        portfolio weights
    '''
    covariance      = asset_returns.cov().values * 52
    weights         = ( np.linalg.inv( covariance ) @ (standard_deviations * np.sqrt(52)) ) / ( np.ones(N).T @ np.linalg.inv( covariance  ) @ (standard_deviations * np.sqrt(52)) )

    return weights.T



def _maximum_sharpe_ratio_portfolio( asset_returns, risk_free_rate ):
    '''
    Description:
    Inputs:         ARRAY of returns ( annualised ) and the risk free rate
    Outputs:        portfolio weights
    Improvements:   CALCULATE EXCESS RETURNS BEFORE AND INPUT ONLY EXCESS RETURNS?
    '''
    N                       = len( asset_returns.columns  )
    excessMeanReturns       = np.asarray( [ asset_returns[c].mean() * 52 - risk_free_rate for c in [asset_returns.columns] ] )[0, :].T
    covariance              = asset_returns.cov().values * 52
    weights                 = ( np.linalg.inv( covariance ) @ excessMeanReturns ) /  ( np.ones(N).T @ np.linalg.inv( covariance ) @ excessMeanReturns )

    return weights


def _equal_risk_portfolio( standard_deviations ):
    '''
    Description:
    Inputs:
    Outputs:        portfolio weights
    '''
    risk_contributions = []
    for i in std:
        if i < max(standard_deviations):
            risk_contributions.append( max(standard_deviations) / i )
        else:
            risk_contributions.append(i)
    weights = risk_contributions / sum(risk_contributions)

    return weights



def _inverse_volatility_portfolio( standard_deviations ):
    '''
    Description:
    Inputs:
    Outputs:       portfolio weights
    '''
    weights = np.array( [ (1 / ( sigma * np.sqrt(52) ) ) / sum( 1 / (standard_deviations * np.sqrt(52) ) ) for sigma in standard_deviations ] )
    return weights.T



# General operations

def _calculate_sharpe_ratio( risky_returns , std, risk_free ):
    '''
    Description:
    Inputs:
    Outputs:
    '''
    SR = (risky_returns - risk_free ) / ( std )
    return SR

def _calculate_asset_variances( asset_returns ):
    '''
    Description:
    Inputs:
    Outputs:
    '''
    return pd.Series( [ np.var( asset_returns[c] ) for c in asset_returns.columns ] )


def _calculate_asset_returns():
    '''
    Description:
    Inputs:
    Outputs:
    '''

    # generalising asset returns
    #
    #
    #
    #



def _calculate_portfolio_return( asset_returns, weights, risk_free_rate ):
    '''
    Description:
    Inputs:
    Outputs:
    '''
    if round( sum(weights), 6 ) > 1:
        risk_free_weight = 1 - sum(weights)
    elif round( sum(weights), 6 ) < 1:
        risk_free_weight = 1 - sum(weights)
    else:
        risk_free_weight = 0
    risk_free_return = risk_free_rate * risk_free_weight
    N                   = len( asset_returns.columns  )
    meanReturns         = np.asarray( [ asset_returns[c].mean() for c in [asset_returns.columns] ] )[0, :] * 52
    returns             = (weights.T @ meanReturns)
    return returns, risk_free_weight


def _calculate_portfolio_variance( asset_returns, weights ):
    '''
    Description:
    Inputs:
    Outputs:
    '''
    variance = weights @ ( asset_returns.cov().values * 52 ) @  weights.T
    return variance



def _divsersification_ratio( standard_deviations, asset_returns, weights ):
    '''
    Description:
    Inputs:
    Outputs:
    '''
    return ( weights.T @ ( standard_deviations * np.sqrt(52) ) ) / np.sqrt( ( weights.T @ ( asset_returns.cov().values * 52 ) @ weights ) )


# fundamental stock value weighting




# beta parity portfolio




#-----------------------------------------------------------------------------#
# Body
#-----------------------------------------------------------------------------#


currentDirectory = os.getcwd()

# engine = db.create_engine('sqlite:///' + currentDirectory + '/Dax.db') # sqlite:////Users/PeterlaCour/Documents/Research/News/news.db
# connection = engine.connect( )
# sql_query = "SELECT * FROM Dax_Adjusted_Closes;"
# Dax Data
# dax = pd.read_sql_query(sql_query, con = engine)
# dax = data[4000:]
# dax.dropna(how='any', axis = 1, inplace = True)

# SMI ToF
data_total = pd.read_excel(currentDirectory + '/SMI_ToF.xlsx')
columns     = list(data_total.columns)
data = data_total[ columns[1:] ]

# ...
columns = columns[1:]
N = len(columns)
risk_free_rate = 0.02
daily_returns = pd.DataFrame()

for c in columns:
    daily_returns[c] = [ np.log( data[c][j] / data[c][j-1] ) for j in range( 1, len(data[c]) )]

meanReturns         = np.asarray( [ np.mean( daily_returns[c] ) for c in columns ] ) * 52
returns             = daily_returns * 52
variance            = _calculate_asset_variances( daily_returns )
std                 = daily_returns.std().values
returns_array       = daily_returns.values

mv_weights      = _minimum_variance_portfolio( daily_returns )
sum(mv_weights)

max_sr_weights  = _maximum_sharpe_ratio_portfolio( daily_returns, risk_free_rate )
sum(max_sr_weights)

max_div         = _maximum_diversification_portfolio( daily_returns, std )
sum(max_div)

inv_vol         = _inverse_volatility_portfolio( std )
sum( inv_vol )

ew_weights      = _equal_weight_portfolio( daily_returns )
sum(ew_weights)

erc_weights     = _equal_risk_portfolio(std)
sum(erc_weights)

portfolios = { 'GMV': mv_weights,  'MSR': max_sr_weights, 'MDP': max_div, 'InvVol': inv_vol, 'EW': ew_weights, 'ERC': erc_weights }

for p in portfolios.keys():
    print( p + ' Diversification Ratio: ' + str( round( _divsersification_ratio( std, daily_returns, portfolios[p] ), 7 ) ) )

for p in portfolios.keys():
    print( p + ' Returns: ' + str( _calculate_portfolio_return( daily_returns, portfolios[p], risk_free_rate )[0] ) )

for p in portfolios.keys():
    print( p + ' Risk Free Weight: ' + str( _calculate_portfolio_return( daily_returns, portfolios[p], risk_free_rate )[1] ) )

for p in portfolios.keys():
    print( p + ' Standard Deviation: ' + str( np.sqrt (_calculate_portfolio_variance( daily_returns, portfolios[p] ) )  ) )

for p in portfolios.keys():
    print( p + ' Sharpe Ratio: ' + str( _calculate_sharpe_ratio( _calculate_portfolio_return( daily_returns, portfolios[p], risk_free_rate )[0] , np.sqrt( _calculate_portfolio_variance( daily_returns, portfolios[p] ) ), risk_free_rate )  ) )


# minimum variance frontier
N                       = len( daily_returns.columns  )
excessMeanReturns       = np.asarray( [ daily_returns[c].mean() * 52  for c in [daily_returns.columns] ] )[0, :].T - 0.02
covariance              = daily_returns.cov().values * 52

A = meanReturns.T @ np.linalg.inv( covariance ) @ meanReturns
B = meanReturns.T @ np.linalg.inv( covariance ) @ np.ones(N)
C = np.ones(N) @ np.linalg.inv( covariance ) @ np.ones(N)

requiredReturns = np.arange(risk_free_rate, 1.0, 0.01)

risky_weights = []
for i in requiredReturns:
    lamda    = (C*i- B)/(A*C-B**2)
    delta    = (A-B*i)/(A*C-B**2)
    risky_weights.append( np.linalg.inv( covariance ) @ ( meanReturns * lamda + np.ones(N) * delta ) )

# risky weights
variance    = [ (risky_weights[r] @ ( covariance ) @ risky_weights[r] )**0.5 * 100 for r in range(len(risky_weights)) ]
ret         = [ risky_weights[r] @ ( meanReturns ) for r in range(len(risky_weights)) ]


d = { "Standard Deviation": variance , "Returns": requiredReturns * 100 }
d = pd.DataFrame(d)


risky_and_rf_weights = []
for i in requiredReturns:
    risky_and_rf_weights.append( (i - risk_free_rate) / ( excessMeanReturns @ np.linalg.inv( covariance ) @ excessMeanReturns ) * ( np.linalg.inv( covariance ) @ excessMeanReturns ) )

variance2    = [ (risky_and_rf_weights[r] @ ( covariance ) @ risky_and_rf_weights[r] )**0.5 * 100 for r in range(len(risky_and_rf_weights)) ]
d2 = { "Standard Deviation": variance2 , "Returns": requiredReturns * 100 }
d2 = pd.DataFrame(d2)

portfolios_df = pd.DataFrame( {'GMV': [ _calculate_portfolio_variance( daily_returns, portfolios['GMV'] )**0.5 * 100 , _calculate_portfolio_return( daily_returns, portfolios['GMV'], risk_free_rate )[0] * 100 ],
                               'MSR': [ _calculate_portfolio_variance( daily_returns, portfolios['MSR'] )**0.5 * 100 , _calculate_portfolio_return( daily_returns, portfolios['MSR'], risk_free_rate )[0] * 100 ],
                               'MDP': [ _calculate_portfolio_variance( daily_returns, portfolios['MDP'] )**0.5 * 100 , _calculate_portfolio_return( daily_returns, portfolios['MDP'], risk_free_rate )[0] * 100 ],
                               'EW': [ _calculate_portfolio_variance( daily_returns, portfolios['EW'] )**0.5 * 100 , _calculate_portfolio_return( daily_returns, portfolios['EW'], risk_free_rate )[0] * 100 ],
                               'ERC': [ _calculate_portfolio_variance( daily_returns, portfolios['ERC'] )**0.5 * 100 , _calculate_portfolio_return( daily_returns, portfolios['ERC'], risk_free_rate )[0] * 100 ],
                               'InvVol': [ _calculate_portfolio_variance( daily_returns, portfolios['InvVol'] )**0.5 * 100 , _calculate_portfolio_return( daily_returns, portfolios['InvVol'], risk_free_rate )[0] * 100 ]
                               } )

# plot portfolios

plt.scatter( x = portfolios_df['GMV'][0], y=portfolios_df['GMV'][1],label = 'GMV', marker='x')
plt.scatter( x = portfolios_df['MSR'][0], y=portfolios_df['MSR'][1],label = 'MSR', marker='o')
plt.scatter( x = portfolios_df['MDP'][0], y=portfolios_df['MDP'][1],label = 'MDP', marker='s')
plt.scatter( x = portfolios_df['InvVol'][0], y=portfolios_df['InvVol'][1],label = 'InvVol', marker='v')
plt.scatter( x = portfolios_df['EW'][0], y=portfolios_df['EW'][1],label = 'EW', marker='P')
plt.scatter( x = portfolios_df['ERC'][0], y=portfolios_df['ERC'][1],label = 'ERC', marker='X')
plt.plot( 'Standard Deviation', 'Returns', data=d, marker='', color='blue', linewidth=1, label = "Risky Assets MV")
plt.plot( 'Standard Deviation', 'Returns', data=d2, marker='', color='black', linewidth=1, label = "Risky and Riskfree Assets MV")
plt.title('Mean Variance Frontier', fontdict = {'size': 14, 'fontweight': 'bold'})
plt.xlabel('Standard Deviations (%)')
plt.ylabel('Returns (%)')
plt.legend()
plt.savefig('./Graphs/MeanVariance.png')


# diversification ratio
risky_and_rf_weights_dr     = [ _divsersification_ratio( std, daily_returns, risky_and_rf_weights[r] ) for r in range(1,len(risky_and_rf_weights)) ]
risky_weights_dr            = [ _divsersification_ratio( std, daily_returns, risky_weights[r] ) for r in range(1,len(risky_weights)) ]

df1 = { "Standard Deviation": variance2[1:] , "Diversification Ratio": risky_and_rf_weights_dr }
df2 = { "Standard Deviation": variance[1:] , "Diversification Ratio": risky_weights_dr  }

plt.plot( 'Standard Deviation', 'Diversification Ratio', data=df1, marker='', color='blue', linewidth=1, label = "Risky Assets MV")
plt.plot( 'Standard Deviation', 'Diversification Ratio', data=df2, marker='', color='black', linewidth=1, label = "Risky and Riskfree Assets MV")
plt.scatter( x = portfolios_df['GMV'][0], y = _divsersification_ratio( std, daily_returns, portfolios['GMV'] ),label = 'GMV', marker='x')
plt.scatter( x = portfolios_df['MSR'][0], y = _divsersification_ratio( std, daily_returns, portfolios['MSR'] ),label = 'MSR', marker='o')
plt.scatter( x = portfolios_df['MDP'][0], y = _divsersification_ratio( std, daily_returns, portfolios['MDP'] ),label = 'MDP', marker='s')
plt.scatter( x = portfolios_df['InvVol'][0], y = _divsersification_ratio( std, daily_returns, portfolios['InvVol'] ),label = 'InvVol', marker='v')
plt.scatter( x = portfolios_df['EW'][0], y = _divsersification_ratio( std, daily_returns, portfolios['EW'] ),label = 'EW', marker='+')
plt.scatter( x = portfolios_df['ERC'][0], y = _divsersification_ratio( std, daily_returns, portfolios['ERC'] ),label = 'ERC', marker='X')
plt.title('Diversification Ratios vs. Standard Deviation', fontdict = {'size': 14, 'fontweight': 'bold'})
plt.xlabel('Standard Deviations (%)')
plt.ylabel('Diversification Ratio')
plt.legend()
plt.savefig('./Graphs/DiversificationRatioVsStandardDeviation.png')

df1 = { "Standard Deviation": requiredReturns[1:] * 100 , "Diversification Ratio": risky_and_rf_weights_dr }
df2 = { "Standard Deviation": requiredReturns[1:] * 100 , "Diversification Ratio": risky_weights_dr  }

plt.plot( 'Standard Deviation', 'Diversification Ratio', data=df1, marker='', color='blue', linewidth=1, label = "Risky Assets MV")
plt.plot( 'Standard Deviation', 'Diversification Ratio', data=df2, marker='', color='black', linewidth=1, label = "Risky and Riskfree Assets MV")
plt.scatter( x = portfolios_df['GMV'][1], y = _divsersification_ratio( std, daily_returns, portfolios['GMV'] ),label = 'GMV', marker='x')
plt.scatter( x = portfolios_df['MSR'][1], y = _divsersification_ratio( std, daily_returns, portfolios['MSR'] ),label = 'MSR', marker='o')
plt.scatter( x = portfolios_df['MDP'][1], y = _divsersification_ratio( std, daily_returns, portfolios['MDP'] ),label = 'MDP', marker='s')
plt.scatter( x = portfolios_df['InvVol'][1], y = _divsersification_ratio( std, daily_returns, portfolios['InvVol'] ),label = 'InvVol', marker='v')
plt.scatter( x = portfolios_df['EW'][1], y = _divsersification_ratio( std, daily_returns, portfolios['EW'] ),label = 'EW', marker='+')
plt.scatter( x = portfolios_df['ERC'][1], y = _divsersification_ratio( std, daily_returns, portfolios['ERC'] ),label = 'ERC', marker='X')
plt.title('Diversification Ratios vs. Returns', fontdict = {'size': 14, 'fontweight': 'bold'})
plt.xlabel('Returns (%)')
plt.ylabel('Diversification Ratio')
plt.legend()
plt.savefig('./Graphs/DiversificationRatioVsReturns.png')


# Diversification Vs. Sharpe
#def _calculate_sharpe_ratio( risky_returns , std, risk_free ):

risky_and_rf_weights_sr     =  [ _calculate_sharpe_ratio( requiredReturns[r] * 100, variance2[r], risk_free_rate * 100 ) for r in range(1,len(risky_and_rf_weights)) ]
risky_weights_sr            =  [ _calculate_sharpe_ratio( requiredReturns[r] * 100, variance[r], risk_free_rate * 100 ) for r in range(1,len(risky_weights)) ]

sr1 = { "Diversification Ratio": risky_and_rf_weights_dr , "Sharpe Ratio": risky_and_rf_weights_sr }
sr2 = { "Diversification Ratio": risky_weights_dr , "Sharpe Ratio": risky_weights_sr  }

plt.plot( 'Sharpe Ratio', 'Diversification Ratio', data=sr2, marker='', color='blue', linewidth=1, label = "Risky Assets MV")
plt.scatter( 'Sharpe Ratio', 'Diversification Ratio', data=sr1, marker='x', color='black', linewidth=1, label = "Risky and Riskfree Assets MV")
plt.scatter( x = _calculate_sharpe_ratio(portfolios_df['GMV'][1],portfolios_df['GMV'][0],risk_free_rate*100), y = _divsersification_ratio( std, daily_returns, portfolios['GMV'] ),label = 'GMV', marker='x')
plt.scatter( x = _calculate_sharpe_ratio(portfolios_df['MSR'][1],portfolios_df['MSR'][0],risk_free_rate*100), y = _divsersification_ratio( std, daily_returns, portfolios['MSR'] ),label = 'MSR', marker='o')
plt.scatter( x = _calculate_sharpe_ratio(portfolios_df['MDP'][1],portfolios_df['MDP'][0],risk_free_rate*100), y = _divsersification_ratio( std, daily_returns, portfolios['MDP'] ),label = 'MDP', marker='s')
plt.scatter( x = _calculate_sharpe_ratio(portfolios_df['InvVol'][1],portfolios_df['InvVol'][0],risk_free_rate*100), y = _divsersification_ratio( std, daily_returns, portfolios['InvVol'] ),label = 'InvVol', marker='v')
plt.scatter( x = _calculate_sharpe_ratio(portfolios_df['EW'][1],portfolios_df['EW'][0],risk_free_rate*100), y = _divsersification_ratio( std, daily_returns, portfolios['EW'] ),label = 'EW', marker='+')
plt.scatter( x = _calculate_sharpe_ratio(portfolios_df['ERC'][1],portfolios_df['ERC'][0],risk_free_rate*100), y = _divsersification_ratio( std, daily_returns, portfolios['ERC'] ),label = 'ERC', marker='X')
plt.title('Diversification Ratios vs. Sharpe Ratios', fontdict = {'size': 14, 'fontweight': 'bold'})
plt.xlabel('Sharpe Ratio')
plt.ylabel('Diversification Ratio')
plt.legend()
plt.savefig('./Graphs/DiversificationRatioVsSharpeRatio.png')
