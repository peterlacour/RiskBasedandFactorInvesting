
#!/bin/python3

################################################################################
#-------------------------------------------------------------------------------

## Project:      Quantitative Portfolio Management - Risk based and factor investing
## Title:        Class for calculations used for construction of multi-factor models
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
#--------------------- Factor Model Calculations Class -------------------------
#-------------------------------------------------------------------------------
################################################################################

## Class for factor model calculation methods
class FactorModelCalculations():
    # Instantiate class
    def __init__(self):
            pass


    def _calculate_scores( self, dataframe ):
        '''
        Description: Calculates the standardised values of assets for given factor
                     Standardisation on a given date _t is calculated as:
                          ( asset_t - mean of assets_t ) / standard deviation of assets_t
        Inputs:      Factor / Metric dataframe, e.g. P/E ratios
        Outputs:     Dataframe with standardised values of assets for given factor / metric
        '''
        scores = dataframe.copy()
        temp   = dataframe.drop( ['Date'], axis = 1)
        for k in temp.columns:
            for n in range(len(dataframe)):
                if np.isnan(scores[k].loc[n]) == False:
                    scores[k].loc[n] = ( temp[k].loc[n] - np.nanmean(temp.loc[n,]) ) / np.nanstd(temp.loc[n,] )
        return scores



    def _calculate_ranks( self, dataframe, direction):
        '''
        Description:
        Inputs:
        Outputs:
        '''
        if direction == 'descending':
            bool = False
        elif direction == 'ascending':
            bool = True
        else:
            raise Exception("Please specify the sorting direction: 'ascending' or 'descending'")

        temp = dataframe.drop('Date',axis=1)
        ranks = pd.DataFrame( ) # momentumScore.drop('Date',axis=1).copy()
        for i in range(len(dataframe)):
            ranks = ranks.append( temp.iloc[i].rank( method = 'dense', ascending = bool ), ignore_index = True )
        return ranks



    def _create_equal_market_cap_subportfolios( self, scoresFactorDataframe, ranksFactorDataframe, marketCapDataframe, numberOfSubs, direction ):
        '''
        Description:    Creates equal market cap portfolios given a score dataframe

        Inputs:         :scoresFactorDataframe    => score dataframe of given factor ( date column must be named 'Date' )
                        :marketCapDataframe => market cap dataframe ( date column must be named 'Date' )
                        :numberOfSubs       => number of desired sub-portfolios
                        # :factorMultiplier   => series with multipliers for sub-portfolios

        Outputs:        Returns dictionary with three dataframes:
                        SubWeights:subPortfolioWeights=> dataframe with asset weights given subportfolios and multiplier
                        Subs:subPortfolios      => factor dataframe with each row numbered by sub portfolio
                        SubsCaps:subPortfolioCap    => dataframe with market cap per subportfolio

        Improvements:   optimise calculation...
        '''

        dummyarray = np.empty((len(scoresFactorDataframe), len(scoresFactorDataframe.columns) ))
        dummyarray[:] = np.nan
        subPortfolios = { s:  pd.DataFrame(dummyarray.copy(), columns = scoresFactorDataframe.columns ) for s in range(1, numberOfSubs+1) }

        # Rank Scores
        assetRanks              = self._calculate_ranks( scoresFactorDataframe, direction )


        # Set first date
        #firstDate               = marketCapDataframe[ marketCapDataframe.Date == scoresFactorDataframe.Date.loc[0] ].index[0]

        # Overwrite values in ranked tables if value is NaN in market cap table
        for row in range( len( marketCapDataframe ) ):
            for col in marketCapDataframe.columns:  # remove date column
                if np.isnan( marketCapDataframe[col].loc[row] ):
                    assetRanks[col].loc[ row ] = np.nan
                    #subPortfolios[col].loc[ row ] = np.nan


        for row in range( len( scoresFactorDataframe ) ):
            # Add row to subportfolio cap
            # subPortfolioCap     = subPortfolioCap.append(pd.Series(), ignore_index = True )

            # Sort assets by rank and drop na values
            temp                = assetRanks.loc[row].sort_values(axis = 0)
            notNa               = temp.dropna()

            # Create market cap counter and subportfolio counter
            capCounter          = 0
            subCounter          = 1

            # Calculate total market value of row
            totalMarketValue    =  np.nansum( marketCapDataframe[ list( notNa.index.values  )  ].loc[ row  ] ) / numberOfSubs


            # Sort assets into sub portfolios
            for col in  notNa.index.values:
                singleCap = marketCapDataframe[ col ].loc[ row  ]
                # Check if threshold is exceeded
                if ( singleCap + capCounter ) >= totalMarketValue:
                    # Check if we already reached the last sub-portfolio
                    if subCounter >= numberOfSubs:
                        subCounter = numberOfSubs
                        # Assign market cap to stock in last sub
                        subPortfolios[ subCounter ][ col ].loc[ row ] = singleCap
                        capCounter +=  singleCap + capCounter
                    else:
                        # If last sub-portfolio has not been reached
                        # Calculate the excess amount if another asset was added
                        intermediateCap =  (singleCap + capCounter) - totalMarketValue
                        # Add the market cap of the asset minus the exceeded amount
                        subPortfolios[ subCounter ][ col ].loc[ row ] = singleCap - intermediateCap
                        # Set a new market cap counter
                        capCounter = intermediateCap
                        # Increment the subportfolio
                        subCounter += 1
                        # Assign exceeded amount to asset in incremented sub-portfolio
                        subPortfolios[ subCounter ][ col ].loc[ row ] = intermediateCap
                else:
                    # Just add the asset
                    capCounter += singleCap
                    subPortfolios[ subCounter ][ col ].loc[ row ] = singleCap


        # Calculate market cap weights of sub portfolios ( should be 5% for 20 sub portfolios... )
        subPortfolioAssetWeights = { key: subPortfolios[key].copy() for key in list( subPortfolios.keys() ) }

        # Calculate market cap weights of sub portfolios ( should be 5% for 20 sub portfolios... )
        for key in list(subPortfolioAssetWeights.keys()):
            for row in range( len( subPortfolioAssetWeights[key] ) ):
                total = np.nansum( subPortfolios[key].loc[row] )
                for col in subPortfolioAssetWeights[key].columns:
                        subPortfolioAssetWeights[key][col].loc[row] = subPortfolios[key][col].loc[row] / total

        # Calculate Real market weights
        subPortfolioRealWeights = { key: subPortfolios[key].copy() for key in list( subPortfolios.keys() ) }

        # Calculate market cap weights of sub portfolios ( should be 5% for 20 sub portfolios... )
        totalMarketCap = marketCapDataframe.drop('Date', axis = 1).sum( skipna = True , axis = 1 )
        for key in list(subPortfolioRealWeights.keys()):
            for row in range( len( subPortfolioRealWeights[key] ) ):
                for col in subPortfolioRealWeights[key].columns:
                        subPortfolioRealWeights[key][col].loc[row] =  subPortfolios[key][col].loc[row] / totalMarketCap.loc[row]
            subPortfolioRealWeights[key] = subPortfolioRealWeights[key].fillna(0)


        # Rescale to 1
        for key in list(subPortfolioRealWeights.keys()):
            for row in range( len( subPortfolioRealWeights[key] ) ):
                        subPortfolioRealWeights[key][col].loc[row] *= (1 / np.nansum( [ np.nansum(subPortfolioRealWeights[m].loc[row]) for m in subPortfolioRealWeights.keys() ] ) )




        # calculate sub portfolio cap weights?



        subPortfolioWeightedRanks = { key: subPortfolios[key].copy() for key in list( subPortfolios.keys() ) }
        # calculate subportfolio scores:
        for key in list(subPortfolioWeightedRanks.keys()):
            subPortfolioWeightedRanks[ key ] = pd.DataFrame( ranksFactorDataframe * subPortfolioAssetWeights[ key ].drop('Date', axis = 1) ).sum( axis = 1, skipna = True )
        subPortfolioRanks = pd.DataFrame( subPortfolioWeightedRanks )

        return { 'Subs': subPortfolios, 'SubAssetWeights': subPortfolioAssetWeights, 'SubMarketWeights': subPortfolioRealWeights, 'SubPortfolioRanks': subPortfolioRanks } # , 'SubScores': subPortfolioScores




    def _calculate_drawdown( self, returnsArray ):
        '''
        Description:    Calculates drawdown of cumulative returns series
        Inputs:         :returnsArray   => Array / Series with cumulative returns
        Outputs:        Returns a list with drawdown values
        Improvements:   -
        '''
        highWaterMark       = 0
        # highWaterMarkList   = [ ]
        drawdown            = [ ]
        N                   = len( returnsArray )

        for i in range(N):
            if returnsArray[i] > highWaterMark:
                highWaterMark = returnsArray[i]
                drawdown.append( 0.00 )
            else:
                drawdown.append( ((highWaterMark - returnsArray[i]) / highWaterMark) * 100  )

        drawdown = [ -d for d in drawdown ]
        return drawdown



    def _scale_factor_weights( self, factor, subPortfolios, subPortfolioRanks, subPorfolioMarketWeights, factorRanks, marketWeights, marketCapDataframe, numberOfSubs, power = 1, averageMultiplier = False ):
        '''
        Description:

        Inputs:         :factor                 => Factor name as string
                        :subPortfolioRanks      =>
                        :numberOfSubs           =>
                        :power                  => Default = 1, raises subportfolio multipliers of value, size and quality to the power

        Outputs:        Returns a dataframe with multipliers
                        :multiplier

        Improvements:   :
        '''
        multiplier = pd.DataFrame( columns = range( 1, numberOfSubs+1 ) )
        for row in range( len( subPortfolioRanks ) ):
            # create dataframe with those multipliers
            multiplier.loc[row] = [ k**power for k in list( np.linspace( 0.05, 1.95 , numberOfSubs) ) ]

        '''
        # Multipliers suggested by Bender and Wang....
        if factor.lower() in [ 'momentum', 'quality', 'securitylevel' ]:
            for row in range( len( subPortfolioRanks ) ):
                # create dataframe with those multipliers
                multiplier.loc[row] = [ k**power for k in list( np.linspace( 0.05, 1.95 , numberOfSubs) ) ]

        elif factor.lower() == 'value':
            # Calculate subportfolios market weight rank
            marketWeightsRank = pd.DataFrame( marketWeights * factorRanks )
            marketRankSum     = marketWeightsRank[ marketWeightsRank.columns[0] ]
            for row in range( len( marketWeights ) ):
                marketRankSum.loc[row] = np.nansum( marketWeightsRank.loc[row] )

            # Calculate multiplier
            for row in range( len( subPortfolioRanks ) ):
                multiplier      = multiplier.append(pd.Series(), ignore_index = True )
                multiplier.loc[row] = ( subPortfolioRanks.loc[row] / marketRankSum.loc[row] )**power

        elif factor.lower() in [ 'size', 'volatility' ]:
            # Calculate subportfolios market weight rank
            marketWeightsRank = pd.DataFrame( marketWeights * factorRanks )
            marketRankSum     = marketWeightsRank[ marketWeightsRank.columns[0] ]
            for row in range( len( marketWeights ) ):
                marketRankSum.loc[row] = np.nansum( marketWeightsRank.loc[row] )

            # Calculate multiplier
            for row in range( len( subPortfolioRanks ) ):
                multiplier = multiplier.append(pd.Series(), ignore_index = True )
                multiplier.loc[row] = ( subPortfolioRanks.loc[row] / marketRankSum.loc[row] )**power
            # Replace all values greater than 3.0
            multiplier[ multiplier >= 3.0 ] = 3.0

        else:
            raise Exception("Please specify correct factor: 'value', 'size', 'quality', 'momentum', 'volatility'.")
        '''

        # subPortfolioRealWeights[1]
        scaledFactorWeights = ( subPortfolios[1].div( subPortfolios[1] )- 1).fillna(0)


        for m in range( 1, len(multiplier.columns) +1 ):
            for row in range( len( subPorfolioMarketWeights[m] ) ):
                subPorfolioMarketWeights[m].loc[row] = subPorfolioMarketWeights[m].loc[row] * multiplier[ m ].loc[row]
            scaledFactorWeights = scaledFactorWeights.add( subPorfolioMarketWeights[m], axis = 1 )

        for row  in range( len( scaledFactorWeights ) ):
            scaledFactorWeights.loc[row] = scaledFactorWeights.loc[row] / np.nansum( scaledFactorWeights.loc[row] )

        subPortfoliosWeights = multiplier.copy()
        for m in range( 1, len(multiplier.columns) +1 ):
            for row in range( len( subPortfolios[m] ) ):
                subPortfoliosWeights[m].loc[row] = ( np.nansum(subPortfolios[m].loc[row]) / np.nansum( marketCapDataframe.loc[row] ) ) * multiplier[ m ].loc[row]

        for m in range( 1, len(multiplier.columns) +1 ):
            for row in range( len( subPortfolios[m] ) ):
                subPortfoliosWeights[m].loc[row] = subPortfoliosWeights[m].loc[row] / np.nansum( subPortfoliosWeights.loc[row] )


        return scaledFactorWeights, subPortfoliosWeights, multiplier




    def _calculate_strategy( self, factorWeights, allMarketWeights, rebalanceDates, permnos ):
        '''
        '''
        performanceWeights = allMarketWeights.copy()
        r = 0
        for row in range(len(performanceWeights)):
            for col in performanceWeights[ permnos ].columns:
                if performanceWeights['Date'].loc[row] in rebalanceDates:
                    performanceWeights[col].loc[row] = factorWeights[col].loc[r]
                else:
                    performanceWeights[col].loc[row] = performanceWeights[col].loc[row-1]
            if performanceWeights['Date'].loc[row] in rebalanceDates:
                r += 1
        return performanceWeights



    def _calculate_performance( self, strategy, returns ):
        '''
        Description:
        Inputs:
        Outputs:
        '''
        dict = { }
        dict[ 'Returns' ]            = pd.DataFrame( strategy.drop('Date', axis = 1 ) * returns.drop('Date', axis = 1) )
        dict[ 'TotalReturns' ]       = dict[ 'Returns' ].sum(skipna = True, axis = 1 )
        dict[ 'CumulativeReturns' ]  = (dict[ 'TotalReturns' ] + 1).cumprod( skipna = True)

        return dict
