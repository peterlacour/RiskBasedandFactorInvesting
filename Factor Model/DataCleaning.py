
#!/bin/python3

################################################################################
#-------------------------------------------------------------------------------

## Project:      Quantitative Portfolio Management - Risk based and factor investing
## Title:        Data cleaning class used for the factor model project
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
import  datetime            as dt
import  sqlalchemy          as db
import  os


################################################################################
#-------------------------------------------------------------------------------
#------------------------ Data Cleaning Class ----------------------------------
#-------------------------------------------------------------------------------
################################################################################

## Class for data cleaning methods
class DataClean():
    # Instantiate class
    def __init__(self):
        currentDirectory        = os.getcwd()
        self.engine             = db.create_engine('sqlite:///' + currentDirectory + '/Data/Databases/SP500.db')
        self.connection         = self.engine.connect()


    def _insert_row( self, row, df, df_insert ):
        '''
        Description:        Inserts a missing row (i.e. missing dates) into a dataframe

        Inputs:             :row                 => Row / index number
                            :df                  => original dataframe
                            :df_insert           => row / dataframe to insert

        Outputs:            Returns original dataframe with desired rows inserted

        Improvements:       /
        '''
        df1 = df.loc[:row-1, ]
        df2 = df.loc[row:, ]
        df  = df1.append(df_insert).append(df2).reset_index(drop = True)
        return df


    def _clean_weekend_dates( self, datesToBeCleanedFmt, datesFmt ):
        '''
        Description:    Sets dates to next working / trading day recorded in price data

        Inputs:         :datesToBeCleanedFmt         => List with dates to be cleaned
                        :datesFmt                    => List with trading days

        Outputs:        Returns two lists:
                        :datesToBeCleanedFmt         => Cleaned dates in date format
                        :datesToBeCleanedIntegers    => Cleaned dates in integer format

        Improvements:   -
        '''
        for d in range( len( datesToBeCleanedFmt ) ):
            while not datesToBeCleanedFmt[d] in list( datesFmt ):
                datesToBeCleanedFmt[d] = datesToBeCleanedFmt[d] + dt.timedelta(days=1)
        datesToBeCleanedIntegers    = [  int(d.strftime('%Y%m%d') ) for d in datesToBeCleanedFmt ]

        return datesToBeCleanedFmt, datesToBeCleanedIntegers



    def _reduce_to_same_dates( self, dataframe, minimum ):
        '''

        '''
        reducedDf = dataframe.loc[ len(dataframe) - minimum: ]
        reducedDf.reset_index( drop = True, inplace = True )
        return reducedDf



    def _date_clean( self, dataframe, dates, dateColumn, companyIdentifier, companyList, metrics, permnos ):
        '''
        Description:        Transforms the long data dataframe from WRDS / CRSP-Compustat
                            to wide data and adds missing dates as NaN values so that all
                            columns have the same length.
                            Then writes the dataframes for each metric to the database.

        Inputs:             :dataframe          => original long data dataframe
                            :dates              => list of required dates
                            :dateColumn         => name of column with dates in dataframe
                            :companyIdentifier  => name of column for company identifier in dataframe
                            :companyList        => list with company identification numbers (i.e. Permno)
                            :metrics            => dictionary (financial ratios or fundamentals)
                                                    or as list (price data)

        Outputs:            Returns no output, writes list of companies for each metric
                            to database in a wide data format

        Improvements:       ...
        '''
        connection              = self.connection
        dataframe               = dataframe[ dataframe[ dateColumn ].isin( dates ) ]
        factor                  = { }

        for v in metrics:
            factor[v]           = { 'Date': dates }
            N                   = len( dates )
            for p in permnos:
                df              = dataframe[ dataframe[ companyIdentifier ] == p ]
                df              = dataframe[ dataframe[companyIdentifier] == p ].drop_duplicates([dateColumn], keep = 'last')
                df              = df.reset_index().drop( ['level_0'], axis = 1 )
                n               = len( df.values )
                if n < N:
                    # fill up dataframe with missing dates and NaN values
                    for d in range(N):
                            # try statement in case the last date is missing
                            # ( if the last date is missing itwill cause index out of
                            # bounds error in if statement below)
                        try:
                            if dates[d] != df[ dateColumn ].loc[d]:
                                if d != 0:
                                    # get index
                                    missingRow                      = pd.DataFrame( columns = dataframe.columns ).append( pd.Series(), ignore_index = True )
                                    missingRow[ companyIdentifier ] = p
                                    missingRow[ dateColumn ]        = dates[d]
                                    df                              = self._insert_row(d, df, missingRow)
                                    factor[v][p]                    = df[ df[ companyIdentifier ] == p ][ metrics[v] ]
                                else:
                                    missingRow                      = pd.DataFrame( columns = dataframe.columns ).append( pd.Series(), ignore_index = True )
                                    missingRow[ companyIdentifier ] = p
                                    missingRow[ dateColumn ]        = dates[d]
                                    df                              = self._insert_row(d, df, missingRow)
                                    factor[v][p]                    = df[ df[ companyIdentifier ] == p ][ metrics[v] ]
                        except:
                            # add the last date
                            missingRow                              = pd.DataFrame( columns = dataframe.columns ).append( pd.Series(), ignore_index = True )
                            missingRow[ companyIdentifier ]         = p
                            missingRow[ dateColumn ]                = dates[d]
                            df                                      = df.append( missingRow, ignore_index = True )
                            factor[v][p]                            = df[ df[ companyIdentifier ] == p ][ metrics[v] ]
                        # See if loop needs to be broken
                        n = len( df.values )
                        if n == N:
                            break
                elif n > N:
                    # make the loop restart with new N
                    print(p)
                    print(n)
                    print('Missing dates in first asset.')

                else:
                    factor[v][p] = df[ df[ companyIdentifier ] == p ][ metrics[v] ] #.values
            pd.DataFrame( factor[v]).to_sql( name = "SP500_1990_2018_complete_" + v , con = self.engine, if_exists='replace' )
            print("Finished.")
        connection.close()
