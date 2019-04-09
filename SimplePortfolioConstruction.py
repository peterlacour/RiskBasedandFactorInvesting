
#!/bin/python3

################################################################################
#-------------------------------------------------------------------------------

## Project:      Quantitative Portfolio Management - Risk based and factor investing
## Title:        Class for simple risk-based portfolio construction
## Author:       Peter la Cour
## Email:        peter.lacour@student.unisg.ch
## Place, Time:  St. Gallen, 01.04.19

## Description:
##
##

## Improvements:
## Last changes: Refactored code

#-------------------------------------------------------------------------------
################################################################################

################################################################################
#-------------------------------------------------------------------------------
#--------------------------- Loading Packages ----------------------------------
#-------------------------------------------------------------------------------
################################################################################

import  pandas              as pd
import  numpy               as np


################################################################################
#-------------------------------------------------------------------------------
#---------------- Simple Risk Based Portfolio Construction Class ---------------
#-------------------------------------------------------------------------------
################################################################################

## Class for simple risk based portfolio construction
class RiskBasedPortfolioConstruction():
    # Instantiate class
    def __init__(self):
            pass

    ####################################################################################
    # Simple portfolio construction methods
    ####################################################################################

    def _equal_weight_portfolio( self, assetReturns ):
        '''
        Description:
        Inputs:
        Outputs:        portfolio weights
        '''
        N               = len( assetReturns.columns )
        weights         = np.array([-999.99]*N)
        weights.fill(1/N)
        return weights.T

    def _minimum_variance_portfolio( self, assetReturns, t ):
        '''
        Description:    Calculates weights of the min variance portfolio with:
                        ( inverse of covariance matrix * vector of ones ) / ( transposed vector of ones * inverse of covariance matrix * vector of ones )
                        Vector of ones has size N x 1 where N is the number of assets
        Inputs:         ARRAY of returns ( annualised )
        Outputs:        portfolio weights
        '''
        N               = len( assetReturns.columns )
        covariance      = assetReturns.cov().values * t
        weights         = ( np.linalg.inv( covariance  ) @ np.ones(N) ) /  ( np.ones(N).T @ np.linalg.inv( covariance ) @ np.ones(N) )

        return weights.T



    def _maximum_diversification_portfolio( self, assetReturns, standardDev, t ):
        '''
        Description:    should this be long only ?
        Inputs:
        Outputs:        portfolio weights
        '''
        N               = len( assetReturns.columns )
        covariance      = assetReturns.cov().values * t
        weights         = ( np.linalg.inv( covariance ) @ (standardDev * np.sqrt(t)) ) / ( np.ones(N).T @ np.linalg.inv( covariance  ) @ (standardDev * np.sqrt(t)) )

        return weights.T



    def _maximum_sharpe_ratio_portfolio( self, assetReturns, riskFreeRate, t ):
        '''
        Description:
        Inputs:         ARRAY of returns ( annualised ) and the risk free rate
        Outputs:        portfolio weights
        Improvements:   CALCULATE EXCESS RETURNS BEFORE AND INPUT ONLY EXCESS RETURNS?
        '''
        N                       = len( assetReturns.columns  )
        excessMeanReturns       = np.asarray( [ assetReturns[c].mean() * t - riskFreeRate for c in [assetReturns.columns] ] )[0, :].T
        covariance              = assetReturns.cov().values * t
        weights                 = ( np.linalg.inv( covariance ) @ excessMeanReturns ) /  ( np.ones(N).T @ np.linalg.inv( covariance ) @ excessMeanReturns )

        return weights


    def _equal_risk_portfolio( self, standardDev ):
        '''
        Description:
        Inputs:
        Outputs:        portfolio weights
        '''
        riskContributions = []
        for i in standardDev:
            if i < max(standardDev):
                riskContributions.append( max(standardDev) / i )
            else:
                riskContributions.append(i)
        weights = riskContributions / sum(riskContributions)

        return weights



    def _inverse_volatility_portfolio( self, standardDev, t ):
        '''
        Description:
        Inputs:        :standardDev
                       :t
        Outputs:       portfolio weights
        '''
        weights = np.array( [ (1 / ( sigma * np.sqrt(t) ) ) / sum( 1 / (standardDev * np.sqrt(t) ) ) for sigma in standardDev ] )
        return weights.T


    ####################################################################################
    # General operations
    ####################################################################################

    def _calculate_asset_variances( self, assetReturns ):
        '''
        Description:
        Inputs:
        Outputs:
        '''
        return pd.Series( [ np.var( assetReturns[c] ) for c in assetReturns.columns ] )

    def _calculate_portfolio_return( self, assetReturns, weights, riskFreeRate, t ):
        '''
        Description:
        Inputs:
        Outputs:
        '''
        if round( sum(weights), 6 ) > 1:
            riskFreeWeight = 1 - sum(weights)
        elif round( sum(weights), 6 ) < 1:
            riskFreeWeight = 1 - sum(weights)
        else:
            riskFreeWeight = 0
        # riskFreeReturn      = riskFreeRate * riskFreeWeight
        # N                   = len( assetReturns.columns  )
        meanReturns         = np.asarray( [ assetReturns[c].mean() for c in [assetReturns.columns] ] )[0, :] * t
        returns             = (weights.T @ meanReturns)
        return returns, riskFreeWeight


    def _calculate_portfolio_variance( self, assetReturns, weights, t ):
        '''
        Description:
        Inputs:
        Outputs:
        '''
        variance = weights @ ( assetReturns.cov().values * t ) @  weights.T
        return variance


    ####################################################################################
    # Simple Ratios
    ####################################################################################

    def _divsersification_ratio( self, standardDev, assetReturns, weights, t ):
        '''
        Description:
        Inputs:
        Outputs:
        '''
        return ( weights.T @ ( standardDev * np.sqrt(t) ) ) / np.sqrt( ( weights.T @ ( assetReturns.cov().values * t ) @ weights ) )


    def _calculate_sharpe_ratio( self, riskyReturns , std, riskFreeRate ):
        '''
        Description:
        Inputs:
        Outputs:
        '''
        SR = (riskyReturns - riskFreeRate ) / ( std )
        return SR
