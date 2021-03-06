#### This file contains the Output for the Risk parity and Target volatility portfolios for the course Quantitative Portfolio management

# Author: Julian Woessner

# Date: 17.03.2019

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
library(ggplot2)
library(plotly)
library(gganimate)
library(gifski)
library(png)
library(transformr)
library(magick)
############ Benchmark data #########################


############### Downloading universe S&P 500 ############################

# Getting all tickers of the S&P 500 constituents
url <- "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
SP500ticker <- url %>%
  read_html() %>%
  html_nodes(xpath='//*[@id="mw-content-text"]/div/table[1]') %>%
  html_table()
SP500ticker <- SP500ticker[[1]]
SP500ticker <- SP500ticker[,2]



ticker.list <- c("MMM", "AMGN", "AXP", "AAPL", "CSCO", "CBS", "KO", "CL", "XOM", "GS", "GE", "K", 
                 "JNJ", "JPM", "NOC", "PFE", "ORCL", "VZ", "WMT", "C")
Pricedata <- get.hist.quote(instrument=ticker.list[1], start="2000-01-01",
                    end= "2019-03-17", quote="AdjClose",
                    provider="yahoo", origin="1970-01-01",
                    compression="d", retclass="zoo")

for (ticker in ticker.list[2:length(ticker.list)]) {
  tmp <- get.hist.quote(instrument=ticker, start="2000-01-01",
                        end= "2019-03-17", quote="AdjClose",
                        provider="yahoo", origin="1970-01-01",
                        compression="d", retclass="zoo")
  Pricedata <- merge.zoo(Pricedata,tmp)
}

colnames(Pricedata) <- ticker.list
head(Pricedata)



# Calculating the log returns of the series
logret <- diff(log(Pricedata))
logret <- logret[!is.na(apply(logret,1,sum)),]



# Defining the log return series for optimization

logretTraining <- logret[1:250]

# Defining the log return series for backtesting

logretTest <- logret[251:nrow(logret)]
### Timeseries starts at 19.11.2013



####################################### Defining portfolios and calculating returns for buy and hold strategies #############################


############################# Creating 1/n portfolio ##########################################################
 Nweight <- NA
for(i in 1:length(ticker.list)) {
  Nweight[i] <- 1/length(ticker.list)

}

Nret <- Return.portfolio(logretTest, weights = Nweight)

############################## Creating Minimum Variance Portfolio of the universe #############################

portf.minvar <- portfolio.spec(assets=ticker.list)
portf.minvar <- add.constraint(portfolio=portf.minvar,type = "weight_sum", min_sum = 0.99, max_sum = 1.01)
portf.minvar <- add.constraint(portfolio=portf.minvar, type= "long_only")
portf.minvar <- add.objective(portfolio=portf.minvar, type="risk", name="StdDev")
portf.minvar

opt.MinVar <- optimize.portfolio(R=logretTraining, portfolio= portf.minvar, 
                                    optimize_method="ROI",trace=TRUE)

opt.MinVar$weights
minVarRet <- Return.portfolio(logretTest, weights = opt.MinVar$weights)



########################### Maximum Sharpe ratio portfolio #########################################################

portf.maxsharpe <- portfolio.spec(assets=ticker.list)
portf.maxsharpe <- add.constraint(portfolio=portf.maxsharpe, type = "weight_sum",min_sum = 0.99, max_sum = 1.01)
portf.maxsharpe <- add.constraint(portfolio=portf.maxsharpe, type="long_only")
portf.maxsharpe <- add.objective(portfolio=portf.maxsharpe, type="return", name="mean")
portf.maxsharpe <- add.objective(portfolio=portf.maxsharpe, type="risk", name="StdDev")

maxSRportf <- optimize.portfolio(R=logretTraining, portfolio=portf.maxsharpe, 
                                   optimize_method="ROI", 
                                   maxSR=TRUE, trace=TRUE)
maxSRportf$weights

# Calculating returns of the maximum sharpe ratio portfolio
maxSRret <- Return.portfolio(logretTest, weights = maxSRportf$weights) # Calculating returns of Max. Sharpe portfolio





############################ Risk Parity Portfolio #########################################################
# objective function
eval_f <- function(w,cov.mat,vol.target) {
  vol <- sqrt(as.numeric(t(w) %*% cov.mat %*% w))
  marginal.contribution <- cov.mat %*% w / vol
  return( sum((vol/length(w) - w * marginal.contribution)^2) )
}

# numerical gradient approximation for solver
eval_grad_f <- function(w,cov.mat,vol.target) {
  out <- w
  for (i in 0:length(w)) {
    up <- dn <- w
    up[i] <- up[i]+.0001
    dn[i] <- dn[i]-.0001
    out[i] = (eval_f(up,cov.mat=cov.mat,vol.target=vol.target) - eval_f(dn,cov.mat=cov.mat,vol.target=vol.target))/.0002
  }
  return(out)
}

Riskparity <- function(returns){
  std <- apply(logretTraining,2,sd)
  cov.mat <- cov(logretTraining)
  x0 <- 1/std/sum(1/std)
        RPportf <- nloptr( x0=x0,
               eval_f=eval_f,
               eval_grad_f=eval_grad_f,
               eval_g_eq=function(w,cov.mat,vol.target) { sum(w) - 1 },
               eval_jac_g_eq=function(w,cov.mat,vol.target) { rep(1,length(std)) },
               lb=rep(0,length(std)),ub=rep(1,length(std)),
               opts = list("algorithm"="NLOPT_LD_SLSQP","print_level" = 3,"xtol_rel"=1.0e-8,"maxeval" = 1000),
               cov.mat = cov.mat,vol.target=2 )
}

RPportf <- Riskparity(logretTraining)
RPportf$solution
# total contributions to risk are equal
  RPportf$solution * cov.mat %*% RPportf$solution

# total portfolio risk
  sum(t(RPportf$solution) %*% cov.mat %*% RPportf$solution)

# Calculating returns of the RP portfolio

  RPweights <- RPportf$solution

  RPret <- Return.portfolio(logretTest, weights = RPweights)

############## Creating plot for the static portfolios ########################################################

  Portfolioreturns <- merge.xts(Nret, minVarRet, maxSRret, RPret) # Combining the returns timeseries of the different portfolios
  
  colnames(Portfolioreturns) <- c("1/N","MinVar", "MaxSR", "RiskParity")
  charts.PerformanceSummary(Portfolioreturns)

# Mean returns
  
  mean_static <- apply(Portfolioreturns, 2, FUN = mean)
  mean_static
  
# Realized standard deviations
  
  sd_static <- apply(Portfolioreturns, 2, FUN = sd)
  sd_static
  

  
  
  
  
  
  
############################# Creating dynamic portfolio strategies and backtesting #############################

  frequency <- "quarters" # Rebalancing frequency
  
  training <- 250 # Training period
  
  
  
  
############################ 1/N ##################################################
  Nweight <- NA
  for(i in 1:length(ticker.list)) {
    Nweight[i] <- 1/length(ticker.list)
    
  }
  
  Nret_dynamic <- Return.portfolio(logret, weights = Nweight, rebalance_on = "quarters")
  Nret_dynamic <- Nret_dynamic[-c(1:18),]
  
############################ Minimum Variance #######################################

 MinVar_dynamic <- optimize.portfolio.rebalancing(logret, portf.minvar, optimize_method = "ROI",
                                                  training_period = training, rolling_window = 90, rebalance_on = frequency,
                                                  traceDE = 0)
  
  MinVar_dynamic_ret <- summary(MinVar_dynamic)$portfolio_returns
  


######################### Maximum Sharpe ratio ######################################
 MaxSR_dynamic <- optimize.portfolio.rebalancing(logret, portf.maxsharpe, optimize_method = "ROI",
                                                   training_period = training, rebalance_on = frequency, rolling_window = 90, search_size = 5000,
                                                   traceDE = 0)
  
  MaxSR_dynamic_ret <- summary(MaxSR_dynamic)$portfolio_returns
  

  

  
  
  
###################### Risk parity ##################################################

  # This function calculates the risk parity portfolio weights for given rebalancing dates
  #
  # Args: 
  #
  # returns:      an xts object containing the timeseries of returns for different assets
  # frequency:    a time object, indicating the rebalancing frequency, e.g. ("days", "weeks", "months", "quarters")
  #
  # Output:       Weights for the Equal risk parity portfolio at the respective rebalancing dates.
  
  Riskparity <- function(returns, frequency){
    
    ep <- endpoints(returns, "quarters")
    ep <- ep[-1]
    RPweights <- returns[ep,]
    j <- NA
    j <- 1
    for (i in ep){
      if(i <= 90){
    std <- apply(returns[c(1:i),],2, FUN = sd)
    cov.mat <- cov(returns[c(1:i),])
    x0 <- 1/std/sum(1/std)
    RPportf <- nloptr( x0=x0,
                       eval_f=eval_f,
                       eval_grad_f=eval_grad_f,
                       eval_g_eq=function(w,cov.mat,vol.target) { sum(w) - 1 },
                       eval_jac_g_eq=function(w,cov.mat,vol.target) { rep(1,length(std)) },
                       lb=rep(0,length(std)),ub=rep(1,length(std)),
                       opts = list("algorithm"="NLOPT_LD_SLSQP","print_level" = 3,"xtol_rel"=1.0e-8,"maxeval" = 1000),
                       cov.mat = cov.mat,vol.target=0 )
      } else {
        std <- apply(returns[c(i-90:i),],2, FUN = sd)
        cov.mat <- cov(returns[c(i-90:i),])
        x0 <- 1/std/sum(1/std)
        RPportf <- nloptr( x0=x0,
                           eval_f=eval_f,
                           eval_grad_f=eval_grad_f,
                           eval_g_eq=function(w,cov.mat,vol.target) { sum(w) - 1 },
                           eval_jac_g_eq=function(w,cov.mat,vol.target) { rep(1,length(std)) },
                           lb=rep(0,length(std)),ub=rep(1,length(std)),
                           opts = list("algorithm"="NLOPT_LD_SLSQP","print_level" = 3,"xtol_rel"=1.0e-8,"maxeval" = 1000),
                           cov.mat = cov.mat,vol.target=0 )
        
      }
    RPweights[j,] <- RPportf$solution 
    
    
    j <- j+1
    }
    RPweights
  }

RPweights_dynamic <- Riskparity(logret, "quarters")   # Calculating the rebalancing weights for the ERC portfolio


ERCret_dynamic <- Return.portfolio(logret, weights = RPweights_dynamic) # Calculating the return series for the ERC portfolio

 



####################### Calculate the performance summary graphs ##########################
Portfolioreturns_dynamic <- merge.xts(Nret_dynamic ,MinVar_dynamic_ret, ERCret_dynamic, MaxSR_dynamic_ret)  # Combining the return series into one xts object
Portfolioreturns_dynamic <- Portfolioreturns_dynamic[!is.na(apply(Portfolioreturns_dynamic, 1, FUN = sum)),] # Deleting NA rows
colnames(Portfolioreturns_dynamic) <- c("1/N","MinVar", "ERC", "MaxSR") # Naming the columns
charts.PerformanceSummary(Portfolioreturns_dynamic) # Performance summary graphs
charts.RollingPerformance(Portfolioreturns_dynamic) # Rolling performance graphs

table.DownsideRisk(Portfolioreturns_dynamic) # Performance figures
table.AnnualizedReturns(Portfolioreturns_dynamic) # Performance figures.



  
############################ Target Volatility #######################################

############## SP500 data ####################################################################
  SP500 <- get.hist.quote(instrument= "SPY", start="2000-01-01",
                      end= "2019-03-17", quote="AdjClose",
                      provider="yahoo", origin="1970-01-01",
                      compression="w", retclass="zoo") # Downloading weekly data for the SP500 index
  
  SP500ret <- diff(log(SP500)) # Calculating log returns
  colnames(SP500) <- "SP500" # Naming the columns
  
targetvolweights <- function(target, returns, rebalancing_period, leverage){
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

### Calculating the returns of the target volatility strategy

weightsrisky <- as.xts(targetvolweights(0.2, SP500ret, "quarters", leverage = 0)) # Weights in the risky asset
weightsrf <- 1-weightsrisky # Respective weights in the money market account
weightsPF <- merge.xts(weightsrisky, weightsrf) # Creating a Timeseries with the weights
colnames(weightsPF) <- c("SP500", "Money Market") # Naming columns.

SP500returns <- SP500ret[-c(1:12),] # Creating SP500 returns vector for the risky asset
Moneymarket <- as.zoo(0.00) # Defining the return vector for the money market

Portfret <- merge.zoo(SP500ret, Moneymarket)[-1,] # Merge the returns of the SP500 and the money market
colnames(Portfret) <- c("SP500", "Money Market") # Naming the columns

Portfret <- na.fill(Portfret,0) # Fill the NAs with 0

TVret <- Return.portfolio(Portfret, weights=weightsPF) # Calculating the rebalancing strategy returns of the Portfolio
colnames(TVret) <- "Target Volatility" # Naming column

#### Calculating returns for the 50/50 strategy

Returns5050 <- SP500returns * 0.5


#### Analysisi and performance of the strategy

Returns <- merge.xts(TVret, SP500returns, Returns5050) # Create time series object with the return series of the 3 portfolios
colnames(Returns) <- c("Target Volatility", "SP500", "50/50") # Naming columns

charts.PerformanceSummary(Returns, main = "Performance Graphs") # Create a performance summary graph
charts.RollingPerformance(Returns) # Create a rolling performance summary graph
charts.RollingRegression(TVret, SP500ret) # Create a rolling regression graph

######### Calculating performance numbers

table.DownsideRisk(Returns) # Calculate risk figures
table.AnnualizedReturns(Returns) # Calculate return figures



## Evaluating why the Risk parity portfolio performs worse than the 1/N -----------------

ERCdyn_Weights <- Riskparity(logret[-c(1:250),], "quarters")
Weights1N <- Nweight
weightsMaxSR <- extractWeights(MaxSR_dynamic)
weightsMinVar <- extractWeights(MinVar_dynamic)
PortfRisk_ERC <- data.frame(NA)
PortfRisk_1N <- data.frame(NA)
PortfRisk_MaxSR <- data.frame(NA)
PortfRisk_MinVar <- data.frame(NA)
ep <- endpoints(logret[-c(1:250),], "quarters")[-1]

j <- 1
  for (i in ep){
    if(i <= 90) {
    cov.mat <- cov(logret[c(1:i),])
    PortfRisk_ERC[j,] <- sum(t(as.vector(ERCdyn_Weights[j,])) %*% cov.mat %*% as.vector(ERCdyn_Weights[j,]))
    PortfRisk_1N[j,] <- sum(t(as.vector(Weights1N)) %*% cov.mat %*% as.vector(Weights1N))
    PortfRisk_MaxSR[j,] <- sum(t(as.vector(weightsMaxSR[j,])) %*% cov.mat %*% as.vector(weightsMaxSR[j,]))
    PortfRisk_MinVar[j,] <- sum(t(as.vector(weightsMinVar[j,])) %*% cov.mat %*% as.vector(weightsMinVar[j,]))
  } else {
    cov.mat <- cov(logret[c(i-90:i),])
    
    PortfRisk_ERC[j,] <- sum(t(as.vector(ERCdyn_Weights[j,])) %*% cov.mat %*% as.vector(ERCdyn_Weights[j,]))
    PortfRisk_1N[j,] <- sum(t(as.vector(Weights1N)) %*% cov.mat %*% as.vector(Weights1N))
    PortfRisk_MaxSR[j,] <- sum(t(as.vector(weightsMaxSR[j,])) %*% cov.mat %*% as.vector(weightsMaxSR[j,]))
    PortfRisk_MinVar[j,] <- sum(t(as.vector(weightsMinVar[j,])) %*% cov.mat %*% as.vector(weightsMinVar[j,]))
  }
    j <- j+1
    }



Risknumbers1nERC <- as.xts(logret[-c(1:250),][ep,-c(5:ncol(logret))])
Risknumbers1nERC$MMM <- t(PortfRisk_ERC)*100
Risknumbers1nERC$AMGN <- t(PortfRisk_1N)*100
Risknumbers1nERC$AXP <- t(PortfRisk_MaxSR)*100
Risknumbers1nERC$AAPL <- t(PortfRisk_MinVar)*100
Risknumbers1nERC <- merge.xts(Risknumbers1nERC,cumprod(1+ERCret_dynamic))
Risknumbers1nERC <- merge.xts(Risknumbers1nERC,cumprod(1+Nret_dynamic))
Risknumbers1nERC <- merge.xts(Risknumbers1nERC,cumprod(1+MaxSR_dynamic_ret))
Risknumbers1nERC <- merge.xts(Risknumbers1nERC,cumprod(1+MinVar_dynamic_ret))
Risknumbers1nERC[251,c(5,6,7,8)] <- 1
Risknumbers1nERC[251,c(1,2,3,4)] <- 0
Risknumbers1nERC <- Risknumbers1nERC[!is.na(apply(Risknumbers1nERC,1,sum)),]
Risknumbers1nERC <- fortify(Risknumbers1nERC)

colnames(Risknumbers1nERC) <- c("Date","ERC","N","MaxSR", "MinVar", "CumRetERC", "CumRet1N", "CumRetMaxSR", "CumRetMinVar")

str(Risknumbers1nERC)

require(scales)


Riskgraph <- ggplot() + geom_line(data = Risknumbers1nERC, aes(x = Date, y = ERC, colour = "ERC")) +
 geom_line(data = Risknumbers1nERC, aes(x = Date, y = N, colour = "1/N")) +
  geom_line(data = Risknumbers1nERC, aes(x = Date, y = MaxSR, colour = "MaxSR")) + 
  geom_line(data = Risknumbers1nERC, aes(x = Date, y = MinVar,colour = "MinVar")) + transition_reveal(Date) +
  scale_x_date(labels = date_format("%Y-%m-%d"), breaks = date_breaks("12 months")) + 
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +labs(x = "Date", y = "Portfolio Risk") + 
  scale_colour_manual("", 
                      breaks = c("ERC", "1/N", "MaxSR", "MinVar"),
                      values = c("black", "green", "blue", "red")) 

Riskgraph
anim_save("Riskgraph.gif", last_animation()) #path = "/Users/julianwossner/Dropbox/QPM - Presentation/Riskgraph.gif")

Returngraph <- ggplot() + geom_line(data = Risknumbers1nERC, aes(x = Date, y = CumRetERC, colour = "ERC")) +
 geom_line(data = Risknumbers1nERC, aes(x = Date, y = CumRet1N, colour = "1/N")) +
  geom_line(data = Risknumbers1nERC, aes(x = Date, y = CumRetMaxSR, colour = "MaxSR")) +
  geom_line(data = Risknumbers1nERC, aes(x = Date, y = CumRetMinVar, colour = "MinVar")) +
  transition_reveal(Date) + 
  scale_x_date(labels = date_format("%Y-%m-%d"), breaks = date_breaks("12 months")) + 
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))+labs(x = "Date", y = "Cumulative Return")+
  scale_colour_manual("", 
                    breaks = c("ERC", "1/N", "MaxSR", "MinVar"),
                    values = c("black", "green", "blue", "red")) 
Returngraph
anim_save("Returngraph.gif", last_animation())#, paht = "/Users/julianwossner/Dropbox/QPM - Presentation/Returngraph.gif")

# ECDF plots
ECDFMaxSR <- ecdf(as.vector(MaxSR_dynamic_ret))
ECDFMinVar <- ecdf(as.vector(MinVar_dynamic_ret))
ECDFERC <- ecdf(as.vector(ERCret_dynamic))
ECDF1N <- ecdf(as.vector(Nret_dynamic))

d <- data.frame(x = c(-0.1,0.1))

ll <- Map(f = stat_function, colour = c("blue", "red", "green", "black"), fun = list(ECDFMaxSR,ECDFMinVar,ECDFERC, ECDF1N), geom = "line")

ggplot(data = d, aes(x = x)) + ll 


# Skewness and Kurtosis of the ERC, MinVar and MaxSR portfolios
skew.1n <- skewness(Nret_dynamic)
  kurt.1n <- kurtosis(Nret_dynamic)
skew.erc <- skewness(ERCret_dynamic)
  kurt.erc <- kurtosis(ERCret_dynamic)
skew.maxsr <- skewness(MaxSR_dynamic_ret)
  kurt.maxsr <- kurtosis(MaxSR_dynamic_ret)

