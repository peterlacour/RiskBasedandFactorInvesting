
# Version 1.0
#############Installing relevant packages ####################
library(PortfolioAnalytics)
library(zoo)
library(tseries)
library(rvest)
library(ROI)
library(ROI.plugin.quadprog)
library(nloptr)
library(ROI.plugin.alabama)
library(ROI.plugin.ecos)
library(ROI.plugin.glpk)
library(ROI.plugin.ipop)
library(ROI.models.miplib)
library(ROI.plugin.msbinlp)
library(ROI.plugin.nloptr)
library(ROI.plugin.optimx)
library(ROI.plugin.neos)
library(ROI.plugin.scs)
library(ROI.plugin.qpoases)
library(quantmod)
library(roll)
############## SP500 data ####################################################################
SP500 <- get.hist.quote(instrument= "SPY", start="2000-01-01",
                        end= "2019-03-17", quote="AdjClose",
                        provider="yahoo", origin="1970-01-01",
                        compression="w", retclass="zoo") # Downloading weekly data for the SP500 index

SP500ret <- diff(log(SP500)) # Calculating log returns
colnames(SP500) <- "SP500" # Naming the columns

targetvolweights <- function(target, returns, rebalancing_period, leverage, ciao){
  # This function calculates the weights of the target volatility portfolio.
  # Arguments: 
  # target:                   Annual target volatility level
  # returns:                  xts object containing the returns of the risky portfolio
  # rebalancing period:       Periods a rebalancing should take place ("days", "weeks", "months", "quarters", "years")
  #
  # Output:
  # weights:                  An xts object containing the weights and the respective dates of the rebalancing
  
  ep <- endpoints(returns, rebalancing_period)
  
  target <- target/sqrt(252)
  vola <- period.apply(returns, INDEX = ep ,FUN = sd)
  weights <- as.zoo(target/vola)
  
  if (leverage == 0)  {
    weights[weights > 1] <- 1 # Long only restriction
  } else {
    
  }
  weights
}

############################ Rebalancing payoffs #######################################

weightsriskyd <- as.xts(targetvolweights(0.2, SP500ret, "months", leverage = 0)) # Weights in the risky asset
weightsriskyw <- as.xts(targetvolweights(0.2, SP500ret, "quarters", leverage = 0)) # Weights in the risky asset
weightsriskyq <- as.xts(targetvolweights(0.2, SP500ret, "years", leverage = 0)) # Weights in the risky asset

weightsrfd <- 1-weightsriskyd # Respective weights in the money market account
weightsrfw <- 1-weightsriskyw # Respective weights in the money market account
weightsrfq <- 1-weightsriskyq # Respective weights in the money market account

weightsPFd <- merge.xts(weightsriskyd, weightsrfd) # Creating a Timeseries with the weights
weightsPFw <- merge.xts(weightsriskyw, weightsrfw) # Creating a Timeseries with the weights
weightsPFq <- merge.xts(weightsriskyq, weightsrfq) # Creating a Timeseries with the weights

SP500returns <- SP500ret[-c(1:12),] # Creating SP500 returns vector for the risky asset
Moneymarket <- as.zoo(0.00) # Defining the return vector for the money market
Portfret <- merge.zoo(SP500ret, Moneymarket)[-1,] # Merge the returns of the SP500 and the money market
colnames(Portfret) <- c("SP500", "Money Market") # Naming the columns

Portfret <- na.fill(Portfret,0) # Fill the NAs with 0

TVret <- Return.portfolio(Portfret, weights=weightsPFd) # Calculating the rebalancing strategy returns of the Portfolio
TVret2 <- Return.portfolio(Portfret, weights=weightsPFw) # Calculating the rebalancing strategy returns of the Portfolio
TVret3 <- Return.portfolio(Portfret, weights=weightsPFq) # Calculating the rebalancing strategy returns of the Portfolio

portf.minvar <- portfolio.spec(assets=TVret)
TVret <- add.constraint(portfolio=TVret, type="transaction_cost", ptc=50)
colnames(TVret) <- "Target Volatility" # Naming column

Returns <- merge.xts(TVret, TVret2, TVret3) # Create time series object with the return series of the 3 portfolios
colnames(Returns) <- c("Monthly", "Quarter", "Yearly") # Naming columns

charts.PerformanceSummary(Returns, main = "Rebalancing Payoffs") # Create a performance summary graph
charts.RollingPerformance(Returns) # Create a rolling performance summary graph
charts.RollingRegression(TVret, SP500ret) # Create a rolling regression graph
charts.RollingPerformance(Returns) # Create a rolling performance summary graph
charts.RollingRegression(TVret, SP500ret) # Create a rolling regression graph

######### Calculating performance numbers

table.DownsideRisk(Returns) # Calculate risk figures
table.AnnualizedReturns(Returns) # Calculate return figures

############## TRANSACTION COSTS ####################################################################

TRM <- 0.003/4
TRQ <- 0.003/13
TRY <- 0.003/52

TVret1 <- TVret-TRM
TVret2 <- TVret2-TRQ
TVret3 <- TVret3-TRY

#### Analysisi and performance of the strategy

Returns <- merge.xts(TVret1, TVret2, TVret3) # Create time series object with the return series of the 3 portfolios
colnames(Returns) <- c("Monthly", "Quarter", "Yearly") # Naming columns

charts.PerformanceSummary(Returns, main = "Rebalancing Payoffs after TC") # Create a performance summary graph

############## CALM AND STORM ####################################################################
############## GRAPH REGIMES ##############


chart.Bar(Returns^2, main = "Volatility Regimes", legend.loc = TRUE, colorset = (1:10))

##############
rm (SP500)

SP500 <- get.hist.quote(instrument= "SPY", start="2000-01-01",
                        end= "2003-03-17", quote="AdjClose",
                        provider="yahoo", origin="1970-01-01",
                        compression="w", retclass="zoo") # Downloading weekly data for the SP500 index

SP500ret1 <- diff(log(SP500)) # Calculating log returns

rm (SP500)

SP500 <- get.hist.quote(instrument= "SPY", start="2003-03-18",
                        end= "2008-03-17", quote="AdjClose",
                        provider="yahoo", origin="1970-01-01",
                        compression="w", retclass="zoo") # Downloading weekly data for the SP500 index

SP500ret2 <- diff(log(SP500)) # Calculating log returns

rm (SP500)

SP500 <- get.hist.quote(instrument= "SPY", start="2008-03-18",
                        end= "2010-03-17", quote="AdjClose",
                        provider="yahoo", origin="1970-01-01",
                        compression="w", retclass="zoo") # Downloading weekly data for the SP500 index

SP500ret3 <- diff(log(SP500)) # Calculating log returns

rm (SP500)

SP500 <- get.hist.quote(instrument= "SPY", start="2010-03-18",
                        end= "2019-03-17", quote="AdjClose",
                        provider="yahoo", origin="1970-01-01",
                        compression="w", retclass="zoo") # Downloading weekly data for the SP500 index

SP500ret4 <- diff(log(SP500)) # Calculating log returns

weightsriskyl1 <- as.xts(targetvolweights(0.2, SP500ret1, "months", leverage = 0, )) # start="2000-01-01",end= "2003-03-17"
weightsriskyh1 <- as.xts(targetvolweights(0.2, SP500ret2, "years", leverage = 0)) # start="2003-03-18", end= "2008-03-17"
weightsriskyl2 <- as.xts(targetvolweights(0.2, SP500ret3, "months", leverage = 0)) # start="2008-03-18", end= "2010-03-17"
weightsriskyh2 <- as.xts(targetvolweights(0.2, SP500ret4, "years", leverage = 0)) # start="2010-03-18", end= "2019-03-17"

weightsriskyHL <- c(weightsriskyl1, weightsriskyh1, weightsriskyl2, weightsriskyh2)
weightsrfHL <- 1-weightsriskyHL # Respective weights in the money market account
weightsPFHL <- merge.xts(weightsriskyHL, weightsrfHL) # Creating a Timeseries with the weights
TVretHL <- Return.portfolio(Portfret, weights=weightsPFHL) # Calculating the rebalancing strategy returns of the Portfolio

TRP <- 0.003/48
TVretHL <- TVretHL-TRP

Final <- merge.xts(TVret1, TVret2, TVret3, TVretHL) # Create time series object with the return series of the 3 portfolios
colnames(Final) <- c("Monthly", "Quarter", "Yearly", "Optimal") # Naming columns

charts.PerformanceSummary(Final, main = "Dynamic Rebalancing") # Create a performance summary graph


