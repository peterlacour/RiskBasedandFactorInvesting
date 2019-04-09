#!/bin/python3

################################################################################
#-------------------------------------------------------------------------------

## Project:      Quantitative Portfolio Management - Risk Based and Factor Investing
## Title:        Multi-Factor Model for 100 securities in the S&P 500 including a growth factor
## Author:       Peter la Cour
## Email:        peter.lacour@student.unisg.ch
## Place, Time:  St. Gallen, 09.04.19

## Description:
##
##

## Improvements: Refactor
## Last changes: Changed files to csv

#-------------------------------------------------------------------------------
################################################################################

################################################################################
#-------------------------------------------------------------------------------
#--------------------------- Loading Packages ----------------------------------
#-------------------------------------------------------------------------------
################################################################################

# Load files
from    DataCleaning            import *
from    FactorModelCalculations import *

# Load Packages
import  pandas                  as pd
import  numpy                   as np
import  datetime                as dt
import  sqlalchemy              as db
import  seaborn                 as sns
import  matplotlib.pyplot       as plt
import  matplotlib.dates        as mdates
import  os

# Ignore chained assignment
pd.options.mode.chained_assignment = None

# Set plot style
plt.style.use('seaborn')


################################################################################
#-------------------------------------------------------------------------------
#-------------------------------- Functions ------------------------------------
#-------------------------------------------------------------------------------
################################################################################

# Load classes to get methods
fm  = FactorModelCalculations()
dc  = DataClean()


################################################################################
#-------------------------------------------------------------------------------
#---------------------------- Multi factor model -------------------------------
#-------------------------------------------------------------------------------
################################################################################

# Set current working directory
currentDirectory        = os.getcwd()

# Connect to database
engine                  = db.create_engine('sqlite:///' + currentDirectory + '/Data/Databases/SP500.db')
connection              = engine.connect()

# Declare constants
numberOfSubs            = 20
rollingIntervalStd      = 5 * 252
rollingIntervalMom      = 252

power                   = 5
powerString             = str(power).replace('.','_')

csv                     = False
factors                 = [ 'Value', 'Size', 'Quality', 'Momentum', 'Volatility', 'Growth' ]

# Declare value and quality metrics
valueMetrics            = {  'PB': 'ptb', 'PS': 'ps', 'PC': 'pcf', 'DivYield': 'DIVYIELD' }
qualityMetrics          = { 'EarningsVariability': 'epsfxq', 'TrailingEP': 'epsfi12'  } # 'TrailingEP': 'epsf12'
growthMetrics           = { 'PE': 'pe_exi' }

# Load permnos
permnos                 = [ str(p) for p in list( pd.read_csv( currentDirectory + '/CSVs/Permanent_Company_Numbers.csv', index_col = 0  )['0'] ) ]


################################################################################
#--------------- Load Adjusted Closes and calculate Returns --------------------
################################################################################

# Check if just csv's should be loaded to avoid calculations
if csv == False:
    # Load adjusted closes
    adjCloses           = pd.read_csv( currentDirectory + '/CSVs/Adjusted_Closes.csv', index_col = 0 )
    priceDates          = adjCloses['Date']

    # Calculate returns dataframe - starts 1 day ahead
    returns             = { 'Date': adjCloses['Date'][1:] }  # [ 1:len( prices[ priceData.PERMNO == permnos[0] ]['AdjClose']) ] }
    for p in permnos:
        returns[p]      = [ ( adjCloses[p].loc[i] - adjCloses[p].loc[i-1] ) / adjCloses[p].loc[i-1] for i in range( 1,len(adjCloses) ) ]
    returns             = pd.DataFrame( returns )

    # Create backup for asset returns
    returns.to_csv(currentDirectory + '/CSVs/returns.csv')
else:
    # Load adjusted closes
    adjCloses           = pd.read_csv( currentDirectory + '/CSVs/Adjusted_Closes.csv', index_col = 0 )
    priceDates          = adjCloses['Date']
    # Load returns
    returns             = pd.read_csv( currentDirectory + '/CSVs/returns.csv', index_col = 0 )


################################################################################
#----------------------- Retrieve and clean date lists -------------------------
################################################################################

# Retrieve dates and create formatted dates
fundamentalDates        = [ t for t in list( pd.read_csv(currentDirectory + '/CSVs/Dates/S&P_Fundamentals&Financials_Dates_1990_2018.csv',';')['FinancialRatioDates'] ) ]
fundamentalDatesFmt     = [ dt.datetime.strptime( dt.datetime( int( str( int( t ) )[:4] ), int( str(  int( t ) )[4:6] ), int( str( int( t ) )[6:] ) ).strftime('%d/%m/%Y') ,'%d/%m/%Y').date() for t in fundamentalDates ]
priceDatesFmt           = [ dt.datetime.strptime( dt.datetime( int( str( int( t ) )[:4] ), int( str(  int( t )  )[4:6] ), int( str(  int( t )  )[6:] ) ).strftime('%d/%m/%Y') ,'%d/%m/%Y').date() for t in  priceDates  ]
rebalanceDatesAll       = [ t for t in list( pd.read_csv(currentDirectory + '/CSVs/Dates/RebalanceDates_SemiAnnually.csv',';')['RebalanceDates'] ) ]
rebalanceDatesAllFmt    = [ dt.datetime.strptime( dt.datetime( int( str( int( t ) )[:4] ), int( str( int( t ) )[4:6] ), int( str( int( t ) )[6:] ) ).strftime('%d/%m/%Y') ,'%d/%m/%Y').date() for t in rebalanceDatesAll ]
# rebalanceDatesMomentum

# Clean fundamental dates - replace weekends / holidays with following working day
fundamentalDatesFmt,  fundamentalDates  = dc._clean_weekend_dates( fundamentalDatesFmt, priceDatesFmt )
rebalanceDatesAllFmt, rebalanceDatesAll = dc._clean_weekend_dates( rebalanceDatesAllFmt, priceDatesFmt )


################################################################################
#--------- Get market capitalisation and calculate market weights --------------
################################################################################

if csv == False:
    # Load market capitalisation table from the database
    marketCap = pd.read_csv( currentDirectory + '/CSVs/Market_Capitalisation.csv', index_col = 0 )

    # Calculate Market Weights
    marketCap.drop(['Date'], axis = 1, inplace = True)
    marketWeights           = marketCap.copy()

    for row in range( len( marketWeights ) ):
        for col in marketCap.columns:
            marketWeights[col].loc[row] = marketCap[col].loc[row] / np.nansum( marketCap.loc[row] )

    # Add dates back to market cap
    marketCap['Date']       = fundamentalDates
    marketWeights['Date']   = fundamentalDates

    # Create Backups
    marketCap.to_csv(currentDirectory + '/CSVs/Market_Capitalisation.csv')
    marketWeights.to_csv(currentDirectory + '/CSVs/Market_Weights.csv')
else:
    marketCap     = pd.read_csv( currentDirectory + '/CSVs/Market_Capitalisation.csv',  index_col = 0 )
    marketWeights = pd.read_csv( currentDirectory + '/CSVs/Market_Weights.csv',         index_col = 0 )


################################################################################
#----------------------------- Size factor -------------------------------------
################################################################################

if csv == False:
    # Calculate log market cap
    logMarketCap            = np.log( pd.DataFrame(marketCap) )
    logMarketCap[ 'Date' ]  = fundamentalDates

    # Calculate size scores
    sizeScore               = fm._calculate_scores( logMarketCap )
    sizeRanks               = fm._calculate_ranks( sizeScore, 'descending')

    # Backup size scores and ranks
    sizeScore.to_csv(currentDirectory + '/CSVs/Ranks_and_Scores/sizeScore.csv')
    sizeRanks.to_csv(currentDirectory + '/CSVs/Ranks_and_Scores/sizeRanks.csv')

else:
    sizeScore = pd.read_csv(currentDirectory + '/CSVs/Ranks_and_Scores/sizeScore.csv', index_col = 0 )
    sizeRanks = pd.read_csv(currentDirectory + '/CSVs/Ranks_and_Scores/sizeRanks.csv', index_col = 0 )


################################################################################
#--------------------------- Momentum factor -----------------------------------
################################################################################

if csv == False:
    # Momentum - create 12-month trailing returns dataframe - can't use logs - starts 252 days ahead
    trailingReturns         = { 'Date': adjCloses['Date'][ rollingIntervalMom:] }
    for p in permnos:
        trailingReturns[p]  = [ (adjCloses[p].loc[i] / adjCloses[p].loc[ i - rollingIntervalMom ])-1 for i in range( rollingIntervalMom, len(adjCloses) ) ]
    trailingReturns = pd.DataFrame( trailingReturns )

    # Get filter dates for momentum
    fundamentalDates_mom    = [ t for t in fundamentalDates if t >= trailingReturns['Date'].loc[ rollingIntervalMom ] ]

    # Filter for quarterly dates
    trailingReturns         = trailingReturns[ trailingReturns.Date.isin( fundamentalDates_mom  ) ]
    trailingReturns         = trailingReturns.reset_index(drop = True)

    # Calculate momentum scores
    momentumScore           = fm._calculate_scores( trailingReturns )
    momentumRanks           = fm._calculate_ranks( momentumScore, 'ascending')

    # Backup momentum scores and ranks
    momentumScore.to_csv(currentDirectory + '/CSVs/Ranks_and_Scores/momentumScore.csv')
    momentumRanks.to_csv(currentDirectory + '/CSVs/Ranks_and_Scores/momentumRanks.csv')
else:
    momentumScore = pd.read_csv(currentDirectory + '/CSVs/Ranks_and_Scores/momentumScore.csv', index_col = 0 )
    momentumRanks = pd.read_csv(currentDirectory + '/CSVs/Ranks_and_Scores/momentumRanks.csv', index_col = 0 )


################################################################################
#-------------------------- Low volatility factor ------------------------------
################################################################################

if csv == False:
    # Low volatility - create returns dataframe for rolling standard deviations dataframe - trailing 60 month = 252 * 5 - starts 252 * 5 days ahead
    rollingStd              = { 'Date': adjCloses['Date'][ rollingIntervalStd:len( returns[permnos[0]] )] }
    for p in permnos:
        rollingStd[p]       = [ returns[p][(i - rollingIntervalStd ):i].std() for i in range( rollingIntervalStd, len(returns[p]) ) ]
    rollingStd              = pd.DataFrame( rollingStd )

    # Get filter dates for rolling std
    fundamentalDates_vol    = [ t for t in fundamentalDates if t >= rollingStd['Date'].loc[ rollingIntervalStd ] ]

    # Filter for dates
    rollingStd              = rollingStd[ rollingStd.Date.isin( fundamentalDates_vol  ) ]
    rollingStd              = rollingStd.reset_index(drop = True)

    # Calculate low volatility scores
    lowVolScore             = fm._calculate_scores( rollingStd )
    lowVolRanks             = fm._calculate_ranks( lowVolScore, 'descending')

    # Backup low vol scores and ranks
    lowVolScore.to_csv(currentDirectory + '/CSVs/Ranks_and_Scores/lowVolScore.csv')
    lowVolRanks.to_csv(currentDirectory + '/CSVs/Ranks_and_Scores/lowVolRanks.csv')

else:
    lowVolScore = pd.read_csv(currentDirectory + '/CSVs/Ranks_and_Scores/lowVolScore.csv', index_col = 0)
    lowVolRanks = pd.read_csv(currentDirectory + '/CSVs/Ranks_and_Scores/lowVolRanks.csv', index_col = 0)


################################################################################
#------------------------------ Value factor -----------------------------------
################################################################################

if csv == False:
    # Load value factors from database and add to dictionary
    valueFactor                 = { }
    for v in valueMetrics.keys():
        valueFactor[v] = pd.read_csv( currentDirectory + '/CSVs/' + v + '.csv', index_col = 0 )

    # Calculate scores
    valueScores             = { v: fm._calculate_scores( valueFactor[v] ) for v in valueMetrics }
    valueRanks              = { v: fm._calculate_ranks(  valueFactor[v], 'descending' ) for v in valueMetrics }

    # Combine scores
    combinedValueScores = valueScores[ list(valueMetrics.keys())[0] ].copy()
    for p in permnos:
        for k in range(len( combinedValueScores )):
            for v in valueMetrics:
                if np.isnan( valueScores[v][p][k] ) == False:
                    combinedValueScores[p].loc[k] += valueScores[v][p][k]

    combinedValueScores[ permnos ] /= len( valueScores.keys() )
    combinedValueRanks  = fm._calculate_ranks( combinedValueScores, 'descending' )

    combinedValueScores.to_csv(currentDirectory + '/CSVs/Ranks_and_Scores/combinedValueScores_noPE.csv')
    combinedValueRanks.to_csv(currentDirectory + '/CSVs/Ranks_and_Scores/combinedValueRanks_noPE.csv')

else:
    combinedValueScores = pd.read_csv(currentDirectory + '/CSVs/Ranks_and_Scores/combinedValueScores_noPE.csv', index_col = 0)
    combinedValueRanks  = pd.read_csv(currentDirectory + '/CSVs/Ranks_and_Scores/combinedValueRanks_noPE.csv',  index_col = 0)


################################################################################
#------------------------------ Quality factor ---------------------------------
################################################################################

if csv == False:
    # Quarter dates
    # fundamentalQuarters      = [ t for t in list( pd.read_csv(currentDirectory + '/CSVs/Dates/S&P_Fundamentals&Financials_Dates_1990_2018.csv',';')['FundamentalsQuarter'] ) ]

    # Load quality factors from database and add to dictionary
    qualityFactor        = { }
    for v in qualityMetrics.keys():
        qualityFactor[v] = pd.read_csv( currentDirectory + '/CSVs/' + v + '.csv', index_col = 0 )

    # Calculate scores
    qualityScores             = { v: fm._calculate_scores( qualityFactor[v] ) for v in qualityMetrics }
    # Calculate individual ranks ?
    qualityRanks              = { v: fm._calculate_ranks(  qualityFactor[v], 'ascending' ) for v in qualityMetrics }

    combinedQualityScores = qualityScores[ 'EarningsVariability' ].copy() # replace EarningsVariability with something more flexible
    for p in permnos:
        for k in range(len( combinedQualityScores )):
            for v in qualityMetrics:
                if np.isnan( qualityScores[v][p][k] ) == False:
                    combinedQualityScores[p].loc[k] += qualityScores[v][p][k]

    #combinedQualityScores['Date'] = fundamentalDates
    combinedQualityRanks  = fm._calculate_ranks( combinedQualityScores, 'descending' )

    # Create backup
    combinedQualityScores.to_csv( currentDirectory + '/CSVs/Ranks_and_Scores/combinedQualityScores.csv' )
    combinedQualityRanks.to_csv( currentDirectory + '/CSVs/Ranks_and_Scores/combinedQualityRanks.csv' )
else:
    combinedQualityScores       = pd.read_csv(currentDirectory + '/CSVs/Ranks_and_Scores/combinedQualityScores.csv', index_col = 0)
    combinedQualityRanks        = pd.read_csv(currentDirectory + '/CSVs/Ranks_and_Scores/combinedQualityRanks.csv',  index_col = 0)


################################################################################
#------------------------------ Growth factor ---------------------------------
################################################################################

if csv == False:
    # Load value factors from database and add to dictionary
    growthFactor                 = { }
    for v in growthMetrics.keys():
        growthFactor[v] = pd.read_csv( currentDirectory + '/CSVs/' + v + '.csv', index_col = 0 )

    # Calculate scores
    growthScores             = { v: fm._calculate_scores( growthFactor[v] ) for v in growthMetrics }
    growthRanks              = { v: fm._calculate_ranks(  growthFactor[v], 'descending' ) for v in growthMetrics }

    # Combine scores
    combinedGrowthScores = growthScores[ list(growthMetrics.keys())[0] ].copy()
    for p in permnos:
        for k in range(len( combinedGrowthScores )):
            for v in growthMetrics:
                if np.isnan( growthScores[v][p][k] ) == False:
                    combinedGrowthScores[p].loc[k] += growthScores[v][p][k]

    combinedGrowthScores[ permnos ] /= len( valueScores.keys() )
    combinedGrowthRanks  = fm._calculate_ranks( combinedGrowthScores, 'ascending' )

    combinedGrowthScores.to_csv(currentDirectory + '/CSVs/Ranks_and_Scores/combinedGrowthScores.csv')
    combinedGrowthRanks.to_csv(currentDirectory + '/CSVs/Ranks_and_Scores/combinedGrowthRanks.csv')

else:
    combinedGrowthScores = pd.read_csv(currentDirectory + '/CSVs/Ranks_and_Scores/combinedGrowthScores.csv', index_col = 0)
    combinedGrowthRanks  = pd.read_csv(currentDirectory + '/CSVs/Ranks_and_Scores/combinedGrowthRanks.csv',  index_col = 0)





################################################################################
#--------------------------- Create sub portfolios -----------------------------
################################################################################

# Load CSVs




# Make factors the same length

# Find minimum length
minimum                 = min( len(sizeScore) , len(momentumScore), len(lowVolScore), len(combinedValueScores), len(combinedQualityScores), len(combinedGrowthScores) ) - 7
#minimum = len(sizeScore) - 21

# Market Cap
marketCap               = dc._reduce_to_same_dates(marketCap, minimum)
marketWeights           = dc._reduce_to_same_dates(marketWeights, minimum)
# Size
sizeScore               = dc._reduce_to_same_dates(sizeScore, minimum)
sizeRanks               = dc._reduce_to_same_dates(sizeRanks, minimum)
# Momentum
momentumScore           = dc._reduce_to_same_dates(momentumScore, minimum)
momentumRanks           = dc._reduce_to_same_dates(momentumRanks, minimum)
# Value
combinedValueScores     = dc._reduce_to_same_dates(combinedValueScores, minimum)
combinedValueRanks      = dc._reduce_to_same_dates(combinedValueRanks, minimum)
# Quality
combinedQualityScores   = dc._reduce_to_same_dates(combinedQualityScores, minimum)
combinedQualityRanks    = dc._reduce_to_same_dates(combinedQualityRanks, minimum)
# Volatility
lowVolScore             = dc._reduce_to_same_dates(lowVolScore, minimum)
lowVolRanks             = dc._reduce_to_same_dates(lowVolRanks, minimum)
# Growth
combinedGrowthScores    = dc._reduce_to_same_dates(combinedGrowthScores, minimum)
combinedGrowthRanks     = dc._reduce_to_same_dates(combinedGrowthRanks, minimum)



#plt.bar(lowVolRanks[permnos].columns, lowVolRanks[permnos].loc[0].sort_values(axis = 0))


################################################################################
#-------------------- Create equal market cap sub portfolios -------------------
################################################################################

# Size
sizeSubs                = fm._create_equal_market_cap_subportfolios( sizeScore,             sizeRanks,            marketCap, numberOfSubs, 'descending' )
# Momentum
momentumSubs            = fm._create_equal_market_cap_subportfolios( momentumScore,         momentumRanks,        marketCap, numberOfSubs, 'ascending'  )
# Low volatility
volSubs                 = fm._create_equal_market_cap_subportfolios( lowVolScore,           lowVolRanks,          marketCap, numberOfSubs, 'descending' )
# Value
valueSubs               = fm._create_equal_market_cap_subportfolios( combinedValueScores,   combinedValueRanks,   marketCap, numberOfSubs, 'descending' )
# Quality
qualitySubs             = fm._create_equal_market_cap_subportfolios( combinedQualityScores, combinedQualityRanks, marketCap, numberOfSubs, 'ascending'  )
# Growth
growthSubs              = fm._create_equal_market_cap_subportfolios( combinedGrowthScores, combinedGrowthRanks, marketCap, numberOfSubs, 'ascending'  )

################################################################################
#------------------ Calculate and apply factor multipliers ---------------------
################################################################################

# Calculate scaled factor weights using the multipliers
sizeWeights, sizeSubportfolios , sizeMultiplier             = fm._scale_factor_weights( 'size',       sizeSubs['Subs'] ,    sizeSubs['SubPortfolioRanks'],     sizeSubs['SubMarketWeights'] ,     sizeRanks,            marketWeights, marketCap, numberOfSubs,  power )
momentumWeights, momentumSubportfolios, momentumMultiplier  = fm._scale_factor_weights( 'momentum',   momentumSubs['Subs'], momentumSubs['SubPortfolioRanks'], momentumSubs['SubMarketWeights'] , momentumRanks,        marketWeights, marketCap, numberOfSubs,  power )
valueWeights, valueSubportfolios  , valueMultiplier         = fm._scale_factor_weights( 'value',      valueSubs['Subs'],    valueSubs['SubPortfolioRanks'],    valueSubs['SubMarketWeights'] ,    combinedValueRanks,   marketWeights, marketCap, numberOfSubs,  power )
qualityWeights, qualitySubportfolios , qualityMultiplier    = fm._scale_factor_weights( 'quality',    qualitySubs['Subs'],  qualitySubs['SubPortfolioRanks'],  qualitySubs['SubMarketWeights'] ,  combinedQualityRanks, marketWeights, marketCap, numberOfSubs,  power )
volWeights, volSubportfolios ,   volatilityMultiplier       = fm._scale_factor_weights( 'volatility', volSubs['Subs'],      volSubs['SubPortfolioRanks'],      volSubs['SubMarketWeights'] ,      lowVolRanks,          marketWeights, marketCap, numberOfSubs,  power )
growthWeights, growthSubportfolios ,   growthMultiplier     = fm._scale_factor_weights( 'growth',     growthSubs['Subs'],   growthSubs['SubPortfolioRanks'],   growthSubs['SubMarketWeights'] ,   growthRanks,          marketWeights, marketCap, numberOfSubs,  power )


# Calculate combined factor model
combinedMultiFactorWeights                                  = (sizeWeights.fillna(0) + momentumWeights.fillna(0) + valueWeights.fillna(0) + qualityWeights.fillna(0) + volWeights.fillna(0) + growthWeights.fillna(0)) / len(factors)

################################################################################
#---------------------- Weights and Returns for analysis ----------------------
################################################################################

factorDates                     = fundamentalDates[len(fundamentalDates) - minimum:]

if csv == False:
    # Load daily market capitalisation
    marketCapAll                = pd.read_csv( currentDirectory + '/CSVs/Daily_Market_Capitalisation.csv', index_col = 0 )


    # Calculate daily market weights
    marketCapDates              = marketCapAll['Date']
    marketWeightsAll            = marketCapAll.copy()
    marketWeightsAll.drop(['Date'], axis = 1, inplace = True)
    marketCapAll.drop(['Date'], axis = 1, inplace = True)
    for row in range( len( marketWeightsAll ) ):
        for col in marketCapAll.columns:
            marketWeightsAll[col].loc[row] = marketCapAll[col].loc[row] / np.nansum( marketCapAll.loc[row] )

    # Create market weights backup
    marketWeightsAll['Date']    = marketCapDates
    marketWeightsAll.to_csv( currentDirectory + '/CSVs/Daily_Market_Weights.csv' )
else:
    marketCapAll                = pd.read_csv( currentDirectory + '/CSVs/Daily_Market_Capitalisation.csv', index_col = 0 )
    marketWeightsAll            = pd.read_csv( currentDirectory + '/CSVs/Daily_Market_Weights.csv', index_col = 0)

# Get all daily returns
returnsAll                      = returns[ returns.Date >= factorDates[0] ]
returnsAll                      = returnsAll.reset_index( drop = True )


################################################################################
#---------------- Individual and Combined Factor 'Strategies' ------------------
################################################################################

# Calculate individual factors
sizeStrategy                = fm._calculate_strategy( sizeWeights,      marketWeightsAll, rebalanceDatesAll, permnos )
momentumStrategy            = fm._calculate_strategy( momentumWeights,  marketWeightsAll, rebalanceDatesAll, permnos )
valueStrategy               = fm._calculate_strategy( valueWeights,     marketWeightsAll, rebalanceDatesAll, permnos )
volStrategy                 = fm._calculate_strategy( volWeights,       marketWeightsAll, rebalanceDatesAll, permnos )
qualityStrategy             = fm._calculate_strategy( qualityWeights,   marketWeightsAll, rebalanceDatesAll, permnos )
growthStrategy              = fm._calculate_strategy( growthWeights,    marketWeightsAll, rebalanceDatesAll, permnos )

# Calculate combined multi factor strategy
combinedMultiFactorStrategy = fm._calculate_strategy( combinedMultiFactorWeights, marketWeightsAll, rebalanceDatesAll, permnos )


################################################################################
#---------------- Calculate Bottom-Up / Security-Level Strategy -----------------
################################################################################

# Create the bottom up mutli factor model
securityLevelScore                                          = ( ( -1 * sizeScore.drop('Date',axis=1).fillna(0) ) + momentumScore.drop('Date',axis=1).fillna(0) + ( -1 * combinedValueScores.drop('Date',axis=1).fillna(0) ) + ( -1 * lowVolScore.drop('Date',axis=1).fillna(0) ) + combinedQualityScores.drop('Date',axis=1).fillna(0) + combinedGrowthScores.drop('Date',axis=1).fillna(0)) / len(factors)
securityLevelScore.replace(0.0, np.nan)
securityLevelScore['Date']                                  = factorDates
securityLevelRanks                                          = fm._calculate_ranks( securityLevelScore, 'ascending' )
securityLevelSubs                                           = fm._create_equal_market_cap_subportfolios( securityLevelScore, securityLevelRanks, marketCap, numberOfSubs, 'ascending' )
securityLevelRanks['Date']                                  = factorDates
securityLevelWeights, securityLeveSubportfolios, multiplier = fm._scale_factor_weights( 'securityLevel',  securityLevelSubs['Subs'], securityLevelSubs['SubPortfolioRanks'], securityLevelSubs['SubMarketWeights'],  securityLevelRanks, marketWeights, marketCap, numberOfSubs,  power )
securityLevelMultiFactorStrategy                            = fm._calculate_strategy( securityLevelWeights, marketWeightsAll, rebalanceDatesAll, permnos )


################################################################################
#------------------------ Calculate factor performance -------------------------
################################################################################

# Calculate individual factors performance
momentumPerfomance      = fm._calculate_performance( momentumStrategy,                  returnsAll )
sizePerfomance          = fm._calculate_performance( sizeStrategy,                      returnsAll )
valuePerfomance         = fm._calculate_performance( valueStrategy,                     returnsAll )
qualityPerfomance       = fm._calculate_performance( qualityStrategy,                   returnsAll )
volPerfomance           = fm._calculate_performance( volStrategy,                       returnsAll )
growthPerformance       = fm._calculate_performance( growthStrategy,                    returnsAll )

# Calculate multi-Factor performance
combinedPerfomance      = fm._calculate_performance( combinedMultiFactorStrategy,       returnsAll )
securityLevelPerfomance = fm._calculate_performance( securityLevelMultiFactorStrategy,  returnsAll )

# Calculate market performance
marketPerformance       = fm._calculate_performance( marketWeightsAll,                  returnsAll )


################################################################################
#-------------------------- Create time series plots ---------------------------
################################################################################

# Individual factors plot
dates = momentumStrategy['Date']
realDates                = [ dt.datetime( int( str(int(t))[:4] ), int( str(int(t))[4:6] ), int( str(int(t))[6:] ) ).strftime('%d/%m/%Y') for t in dates ]
xDates                   = [ dt.datetime.strptime(d,'%d/%m/%Y').date() for d in realDates ]

years = mdates.YearLocator()   # every year
months = mdates.MonthLocator()  # every month
yearsFmt = mdates.DateFormatter('%Y')

# indvidual factors plot
fig, ax1 = plt.subplots(1,1,sharex=True)
fig.set_size_inches(12, 8)

ax1.set_xlabel("Time", fontdict = {'size': 14 })
ax1.set_ylabel('Cumulative Return', fontdict = {'size': 14 })

# round to nearest years...
datemin = np.datetime64(xDates[0], 'Y')
datemax = np.datetime64(xDates[-1], 'Y') + np.timedelta64(1, 'Y')
ax1.set_xlim(datemin, datemax)

ax1.xaxis.set_major_locator(years)
ax1.xaxis.set_major_formatter(yearsFmt)
ax1.xaxis.set_minor_locator(months)

ax1.format_xdata = mdates.DateFormatter('%Y-%m-%d')
ax1.format_ydata = marketPerformance['CumulativeReturns']
ax1.grid(True)

ax1.set_title('Single factor performance (2010 - 2019) [ power = ' + str( power ) + ' ]', fontdict = {'size': 18, 'fontweight': 'bold'})
fig.autofmt_xdate()

ax1.plot( xDates, marketPerformance['CumulativeReturns'] , 'black', linestyle = ':', label = 'Market' )
ax1.plot( xDates, growthPerformance['CumulativeReturns'], label = 'Growth' )
ax1.legend()
plt.show()
fig.savefig( currentDirectory + '/Graphs/IndividualFactorsWithGrowth_power' + powerString + '.png', dpi = 300)



# Multi-factor plot
dates = momentumStrategy['Date']
realDates                       = [ dt.datetime( int( str(int(t))[:4] ), int( str(int(t))[4:6] ), int( str(int(t))[6:] ) ).strftime('%d/%m/%Y') for t in dates ]
xDates                          = [ dt.datetime.strptime(d,'%d/%m/%Y').date() for d in realDates ]

years = mdates.YearLocator()   # every year
months = mdates.MonthLocator()  # every month
yearsFmt = mdates.DateFormatter('%Y')

# indvidual factors plot
fig, ax1 = plt.subplots(1,1,sharex=True)
fig.set_size_inches(12, 8)

ax1.set_xlabel("Time", fontdict = {'size': 14 })
ax1.set_ylabel('Cumulative Return', fontdict = {'size': 14 })

# round to nearest years...
datemin = np.datetime64(xDates[0], 'Y')
datemax = np.datetime64(xDates[-1], 'Y') + np.timedelta64(1, 'Y')
ax1.set_xlim(datemin, datemax)

ax1.xaxis.set_major_locator(years)
ax1.xaxis.set_major_formatter(yearsFmt)
ax1.xaxis.set_minor_locator(months)

ax1.format_xdata = mdates.DateFormatter('%Y-%m-%d')
ax1.format_ydata = marketPerformance['CumulativeReturns']
ax1.grid(True)

ax1.set_title('Multi-factor performance with growth (2010 - 2019) [ power = ' + str( power ) + ' ]', fontdict = {'size': 18, 'fontweight': 'bold'})
fig.autofmt_xdate()
ax1.plot( xDates, marketPerformance['CumulativeReturns'] ,  'black', linestyle = ':', label = 'Market' )
ax1.plot( xDates, combinedPerfomance['CumulativeReturns'], label = 'Combined Multi-Factor Model' )
ax1.plot( xDates, securityLevelPerfomance['CumulativeReturns'], label = 'Bottom-Up Multi-Factor Model' )
ax1.legend()
plt.show()
fig.savefig( currentDirectory + '/Graphs/MultiFactorWithGrowth_power' + powerString + '.png', dpi = 300)


################################################################################
#--------------------- Calculate performance measures --------------------------
################################################################################

annualisedReturnMarket = ( marketPerformance['CumulativeReturns'].loc[ len(marketPerformance['CumulativeReturns']) - 1 ] )**(1/(len(marketPerformance['CumulativeReturns'])/252)) - 1
annualisedReturnGrowth  = ( growthPerformance['CumulativeReturns'].loc[ len(growthPerformance['TotalReturns']) - 1 ] )**(1/(len(growthPerformance['TotalReturns'])/252)) - 1
annualisedVolatility = marketPerformance['TotalReturns'].std() * np.sqrt(252)
annualisedGrowth = growthPerformance['TotalReturns'].std() * np.sqrt(252)

alphaG = ( growthPerformance['TotalReturns'].mean() - marketPerformance['TotalReturns'].mean() ) * 252
trackingErrorG = ( growthPerformance['TotalReturns'] - marketPerformance['TotalReturns']).std() * np.sqrt(252)
informationRatioG = ( alphaG ) / trackingErrorG
maxDdG = min( fm._calculate_drawdown( growthPerformance['CumulativeReturns'] ) )
maxDdMarket = min( fm._calculate_drawdown( marketPerformance['CumulativeReturns'] ) )

# Create performance table
performanceOutput = pd.DataFrame( columns = [' ', 'Growth', 'Market' ] )
performanceOutput = performanceOutput.append(pd.Series(), ignore_index = True )
performanceOutput[' '].loc[0] = 'Return (p.a.)'
performanceOutput = performanceOutput.append(pd.Series(), ignore_index = True )
performanceOutput[' '].loc[1] = 'Volatility (p.a.)'
performanceOutput = performanceOutput.append(pd.Series(), ignore_index = True )
performanceOutput[' '].loc[2] = 'Max. Drawdown'
performanceOutput = performanceOutput.append(pd.Series(), ignore_index = True )
performanceOutput[' '].loc[3] = 'Alpha'
performanceOutput = performanceOutput.append(pd.Series(), ignore_index = True )
performanceOutput[' '].loc[4] = 'Tracking Error'
performanceOutput = performanceOutput.append(pd.Series(), ignore_index = True )
performanceOutput[' '].loc[5] = 'Information Ratio'
# Returns
performanceOutput['Market'].loc[0] = annualisedReturnMarket * 100
performanceOutput['Growth'].loc[0] = annualisedReturnGrowth * 100
# Volatility
performanceOutput['Market'].loc[1] = annualisedVolatility * 100
performanceOutput['Growth'].loc[1] = annualisedGrowth * 100
# Max DD
performanceOutput['Market'].loc[2] = maxDdMarket * (-1)
performanceOutput['Growth'].loc[2] = maxDdG * (-1)
# Alpha
performanceOutput['Market'].loc[3] = '-'
performanceOutput['Growth'].loc[3] = alphaG * 100
# Tracking Error
performanceOutput['Market'].loc[4] = '-'
performanceOutput['Growth'].loc[4] = trackingErrorG * 100
# Information Ratio
performanceOutput['Market'].loc[5] = '-'
performanceOutput['Growth'].loc[5] = informationRatioG

# Write performance table to excel
performanceOutput.to_excel(currentDirectory + '/CSVs/Performance/Performance_Output_with_Growth_power' + powerString + '.xls')


################################################################################
#----------------- Create Bar Plots of Factor scores and weights ---------------
################################################################################

# Bar plot of scores

# Low Volatility
plt.bar( range( len( permnos ) ), lowVolScore[permnos].mean(axis = 0, skipna = True).sort_values(), facecolor = 'green')
plt.ylabel('Average Standardised Score of Assets', fontdict = {'size': 14 } )
plt.xlabel('Average Asset Rank', fontdict = {'size': 14 })
plt.yticks(np.linspace(-3,14,18))
plt.xticks(np.linspace(0, 100, 5))
plt.title('Low Volatility Scores', fontdict = {'size': 18, 'fontweight': 'bold'})
plt.savefig( currentDirectory + '/Graphs/LowVolScore.png', dpi = 300)

# Momentum
plt.bar( range( len( permnos )),  momentumScore[permnos].mean(axis = 0, skipna = True).sort_values(), facecolor = 'green')
plt.ylabel('Average Standardised Score of Assets', fontdict = {'size': 14 } )
plt.xlabel('Average Asset Rank', fontdict = {'size': 14 })
plt.yticks(np.linspace(-3,14,18))
plt.xticks(np.linspace(0, 100, 5))
plt.title('Momentum Scores', fontdict = {'size': 18, 'fontweight': 'bold'})
plt.savefig( currentDirectory + '/Graphs/Momentum.png', dpi = 300)

# Size scores
plt.bar(range( len( permnos )),  sizeScore[permnos].mean(axis = 0, skipna = True).sort_values(), facecolor = 'green')
plt.ylabel('Average Standardised Score of Assets', fontdict = {'size': 14 } )
plt.xlabel('Average Asset Rank', fontdict = {'size': 14 })
plt.yticks(np.linspace(-3,14,18))
plt.xticks(np.linspace(0, 100, 5))
plt.title('Size Scores', fontdict = {'size': 18, 'fontweight': 'bold'})
plt.savefig( currentDirectory + '/Graphs/SizeScores.png', dpi = 300)

# combinedValueScores
plt.bar(range( len( permnos )),  combinedValueScores[permnos].mean(axis = 0, skipna = True).sort_values(), facecolor = 'green')
plt.ylabel('Average Standardised Score of Assets', fontdict = {'size': 14 } )
plt.yticks(np.linspace(-3,14,18))
plt.xlabel('Average Asset Rank', fontdict = {'size': 14 })
plt.xticks(np.linspace(0, 100, 5))
plt.title('Combined Value Scores', fontdict = {'size': 18, 'fontweight': 'bold'})
plt.savefig( currentDirectory + '/Graphs/ValueScores.png', dpi = 300)

# combinedGrowthScores
plt.bar(range( len( permnos )),  combinedGrowthScores[permnos].mean(axis = 0, skipna = True).sort_values(), facecolor = 'green')
plt.ylabel('Average Standardised Score of Assets', fontdict = {'size': 14 } )
plt.xlabel('Average Asset Rank', fontdict = {'size': 14 })
plt.yticks(np.linspace(-3,14,18))
plt.xticks(np.linspace(0, 100, 5))
plt.title('Combined Growth Scores', fontdict = {'size': 18, 'fontweight': 'bold'})
plt.savefig( currentDirectory + '/Graphs/CombinedGrowthScores.png', dpi = 300)

# combinedQualityScores
plt.bar(range( len( permnos )),  combinedQualityScores[permnos].mean(axis = 0, skipna = True).sort_values(), facecolor = 'green')
plt.ylabel('Average Standardised Score of Assets', fontdict = {'size': 14 } )
plt.xlabel('Average Asset Rank', fontdict = {'size': 14 })
plt.yticks(np.linspace(-3,14,18))
plt.xticks(np.linspace(0, 100, 5))
plt.title('Combined Quality Scores', fontdict = {'size': 18, 'fontweight': 'bold'})
plt.savefig( currentDirectory + '/Graphs/CombinedQualityScores.png', dpi = 300)

# Bottom-Up / Security-Level scores
plt.bar(range( len( permnos )),  securityLevelScore[permnos].mean(axis = 0, skipna = True).sort_values(), facecolor = 'green')
plt.ylabel('Average Standardised Score of Assets', fontdict = {'size': 14 } )
plt.xlabel('Average Asset Rank', fontdict = {'size': 14 })
plt.yticks(np.linspace(-3,14,18))
plt.xticks(np.linspace(0, 100, 5))
plt.title('Bottom-Up Scores', fontdict = {'size': 18, 'fontweight': 'bold'})
plt.savefig( currentDirectory + '/Graphs/BottomUpScores.png', dpi = 300)


################################################################################
#----------------- Create Bar Plots of Factor scores and weights ---------------
################################################################################


plt.bar(sizeSubportfolios.columns, sizeSubportfolios.loc[0] / np.nansum(sizeSubportfolios.loc[0]), facecolor = 'green')
plt.ylabel('Subportfolio Weights', fontdict = {'size': 14 } )
plt.xlabel('# of Subportfolio', fontdict = {'size': 14 })
plt.xticks(range(1,21))
#plt.yticks( np.linspace(0,0.2,21))
plt.title('Rescaled Equal-Market-Cap Subportfolios with power ' + str( power ), fontdict = {'size': 18, 'fontweight': 'bold'})
plt.savefig( currentDirectory + '/Graphs/RescaledSubs' + powerString + '.png', dpi = 300)


################################################################################
#-------------------------------------------------------------------------------
#---------------------------------- End ----------------------------------------
#-------------------------------------------------------------------------------
################################################################################
