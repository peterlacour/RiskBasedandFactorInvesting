#!/bin/python3

## Title:        Quantitative Portfolio Management - Database
## Author:       Peter la Cour
## Email:        peter.lacour@student.unisg.ch
## Place, Time:  St. Gallen, 07.03.19
## Description:
##
## Improvements:
## Last changes:

#-----------------------------------------------------------------------------#
# Loading Packages
#-----------------------------------------------------------------------------#
import  pandas     as       pd
import  os
from    sqlalchemy import   create_engine
import  sqlalchemy as       db
from    sqlalchemy import   update

#-----------------------------------------------------------------------------#
# Body
#-----------------------------------------------------------------------------#

currentDirectory    = os.getcwd()

# get data into database
data                = pd.read_csv( currentDirectory + '/Data/SP500_Prices_1990_2018.csv' )
data2               = pd.read_csv(currentDirectory + '/Data/SP_Fundamentals.csv')
engine              = db.create_engine( 'sqlite:///' + currentDirectory + '/Data/Databases/SP500.db' )
connection          = engine.connect()
# calculate adjusted closes
data["AdjClose"]    = [ data['PRC'][k] * data['CFACPR'][k] for k in range(len(data['PRC'])) ]
data["MarketCap"]   = [ data['AdjClose'][k] * data['SHROUT'][k] for k in range(len(data['AdjClose'])) ]
# Write data to database
data.to_sql( name   = "SP500_Prices_1990_2018", con = engine, if_exists='replace' )
data2.to_sql( name  = "SP500_Fundamentals_1990_2018", con = engine, if_exists='replace' )
connection.close()
