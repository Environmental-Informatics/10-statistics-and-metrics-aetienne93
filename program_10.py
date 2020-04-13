#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 14:05:37 2020

@author: aetienne
"""
##The sole function of this script is to see if it works in gradescope. If it doesn't work in gradescope there's a major problem 
import pandas as pd
import scipy.stats as stats
import numpy as np

def ReadData( fileName ):
    """This function takes a filename as input, and returns a dataframe with
    raw data read from that file in a Pandas DataFrame.  The DataFrame index
    should be the year, month and day of the observation.  DataFrame headers
    should be "agency_cd", "site_no", "Date", "Discharge", "Quality". The 
    "Date" column should be used as the DataFrame index. The pandas read_csv
    function will automatically replace missing values with np.NaN, but needs
    help identifying other flags used by the USGS to indicate no data is 
    availabiel.  Function returns the completed DataFrame, and a dictionary 
    designed to contain all missing value counts that is initialized with
    days missing between the first and last date of the file."""
    
    # define column names
    colNames = ['agency_cd', 'site_no', 'Date', 'Discharge', 'Quality']

    # open and read the file
    DataDF = pd.read_csv(fileName, header=1, names=colNames,  
                         delimiter=r"\s+",parse_dates=[2], comment='#',
                         na_values=['Eqp'])
    DataDF = DataDF.set_index('Date')
    
    # check for neagtive discharge
    DataDF["Discharge"] = DataDF["Discharge"].where( DataDF["Discharge"] >= 0 )
    
    # quantify the number of missing values
    MissingValues = DataDF["Discharge"].isna().sum()
    
    return( DataDF, MissingValues )

def ClipData( DataDF, startDate, endDate ):
    """This function clips the given time series dataframe to a given range 
    of dates. Function returns the clipped dataframe and and the number of 
    missing values."""
    
    DataDF = DataDF.loc[startDate:endDate]

    # quantify the number of missing values
    MissingValues = DataDF["Discharge"].isna().sum()
    
    return( DataDF, MissingValues )

def CalcTqmean(Qvalues):
    """This function computes the Tqmean of a series of data, typically
       a 1 year time series of streamflow, after filtering out NoData
       values.  Tqmean is the fraction of time that daily streamflow
       exceeds mean streamflow for each year. Tqmean is based on the
       duration rather than the volume of streamflow. The routine returns
       the Tqmean value for the given data array."""
    mean = Qvalues.mean()
    tmpAbove = Qvalues.where( Qvalues > mean )
    return ( ( tmpAbove / tmpAbove ).sum() / len(Qvalues) )

def CalcRBindex(Qvalues):
    """This function computes the Richards-Baker Flashiness Index
       (R-B Index) of an array of values, typically a 1 year time
       series of streamflow, after filtering out the NoData values.
       The index is calculated by dividing the sum of the absolute
       values of day-to-day changes in daily discharge volumes
       (pathlength) by total discharge volumes for each year. The
       routine returns the RBindex value for the given data array."""
    tmpSum = np.abs( Qvalues[:-1].values - Qvalues[1:].values ).sum()
    return ( tmpSum / Qvalues[1:].sum() )

def Calc7Q(Qvalues):
    """This function computes the seven day low flow of an array of 
       values, typically a 1 year time series of streamflow, after 
       filtering out the NoData values. The index is calculated by 
       computing a 7-day moving average for the annual dataset, and 
       picking the lowest average flow in any 7-day period during
       that year.  The routine returns the 7Q (7-day low flow) value
       for the given data array."""
    tmpRolling = Qvalues.rolling(window=7).mean()
    return ( tmpRolling.min() )

def CalcExceed3TimesMedian(Qvalues):
    """This function computes the number of days with flows greater 
       than 3 times the annual median flow. The index is calculated by 
       computing the median flow from the given dataset (or using the value
       provided) and then counting the number of days with flow greater than 
       3 times that value.   The routine returns the count of events greater 
       than 3 times the median annual flow value for the given data array."""
    Qmedian = Qvalues.median()
    tmpCount = Qvalues.where( Qvalues > 3. * Qmedian )
    return ( ( tmpCount / tmpCount ).sum() )

def GetAnnualStatistics(DataDF):
    """This function calculates annual descriptive statistcs and metrics for 
    the given streamflow time series.  Values are retuned as a dataframe of
    annual values for each water year.  Water year, as defined by the USGS,
    starts on October 1."""
    
    colnames = ['site_no','Mean Flow', 'Peak Flow','Median Flow','Coeff Var', 'Skew','Tqmean','R-B Index','7Q','3xMedian']
   #resampling annually starting at the start of october
    WYDataDF=DataDF.resample('AS-OCT').mean() 
    #create empty dataframe
    WYDataDF = pd.DataFrame(0, index=WYDataDF.index,columns=colnames) 
   
    WYDataDF = DataDF.resample('AS-OCT').mean()
    WYDataDF.rename(columns={"Discharge":"Mean Flow"}, inplace=True)
    WYDataDF["Peak Flow"] = DataDF["Discharge"].resample('AS-OCT').max()
    WYDataDF["Median Flow"] = DataDF["Discharge"].resample('AS-OCT').median()
    WYDataDF["Coeff Var"] = DataDF["Discharge"].resample('AS-OCT').std() / WYDataDF["Mean Flow"] * 100.
    WYDataDF["Skew"] = DataDF["Discharge"].resample('AS-OCT').apply(stats.skew)
    WYDataDF["Tqmean"] = DataDF["Discharge"].resample('AS-OCT').apply(CalcTqmean)
    WYDataDF["R-B Index"] = DataDF["Discharge"].resample('AS-OCT').apply(CalcRBindex)
    WYDataDF["7Q"] = DataDF["Discharge"].resample('AS-OCT').apply(Calc7Q)
    
    return ( WYDataDF )

def GetMonthlyStatistics(DataDF):
    """This function calculates monthly descriptive statistics and metrics 
    for the given streamflow time series.  Values are returned as a dataframe
    of monthly values for each year."""

    MoDataDF = DataDF.resample('MS').mean()
    MoDataDF.rename(columns={"Discharge":"Mean Flow"}, inplace=True)
    MoDataDF["Coeff Var"] = DataDF["Discharge"].resample('MS').std() / MoDataDF["Mean Flow"] * 100.
    MoDataDF["Tqmean"] = DataDF["Discharge"].resample('MS').apply(CalcTqmean)
    MoDataDF["R-B Index"] = DataDF["Discharge"].resample('MS').apply(CalcRBindex)

    return ( MoDataDF )

def GetAnnualAverages(WYDataDF):
    """This function calculates annual average values for all statistics and
    metrics.  The routine returns an array of mean values for each metric
    in the original dataframe."""
    
    AnnualAverages = WYDataDF.mean()
    
    return( AnnualAverages )

def GetMonthlyAverages(MoDataDF):
    """This function calculates annual average monthly values for all 
    statistics and metrics.  The routine returns an array of mean values 
    for each metric in the original dataframe."""
    
    months = MoDataDF.index.month
    MonthlyAverages = MoDataDF.groupby(months).mean()
    
    return( MonthlyAverages )

# the following condition checks whether we are running as a script, in which 
# case run the test code, otherwise functions are being imported so do not.
# put the main routines from your code after this conditional check.

if __name__ == '__main__':

    # define filenames as a dictionary
    # NOTE - you could include more than jsut the filename in a dictionary, 
    #  such as full name of the river or gaging site, units, etc. that would
    #  be used later in the program, like when plotting the data.
    fileName = { "Wildcat": "WildcatCreek_Discharge_03335000_19540601-20200315.txt",
                 "Tippe": "TippecanoeRiver_Discharge_03331500_19431001-20200315.txt" }
    
    # define blank dictionaries (these will use the same keys as fileName)
    DataDF = {}
    MissingValues = {}
    WYDataDF = {}
    MoDataDF = {}
    AnnualAverages = {}
    MonthlyAverages = {}
    
    # process input datasets
    for file in fileName.keys():
        
        print( "\n", "="*50, "\n  Working on {} \n".format(file), "="*50, "\n" )
        
        DataDF[file], MissingValues[file] = ReadData(fileName[file])
        print( "-"*50, "\n\nRaw data for {}...\n\n".format(file), DataDF[file].describe(), "\n\nMissing values: {}\n\n".format(MissingValues[file]))
        
        # clip to consistent period
        DataDF[file], MissingValues[file] = ClipData( DataDF[file], '1969-10-01', '2019-09-30' )
        print( "-"*50, "\n\nSelected period data for {}...\n\n".format(file), DataDF[file].describe(), "\n\nMissing values: {}\n\n".format(MissingValues[file]))
        
        # calculate descriptive statistics for each water year
        WYDataDF[file] = GetAnnualStatistics(DataDF[file])
        
        # calcualte the annual average for each stistic or metric
        AnnualAverages[file] = GetAnnualAverages(WYDataDF[file])
        
        print("-"*50, "\n\nSummary of water year metrics...\n\n", WYDataDF[file].describe(), "\n\nAnnual water year averages...\n\n", AnnualAverages[file])

        # calculate descriptive statistics for each month
        MoDataDF[file] = GetMonthlyStatistics(DataDF[file])

        # calculate the annual averages for each statistics on a monthly basis
        MonthlyAverages[file] = GetMonthlyAverages(MoDataDF[file])
        
        print("-"*50, "\n\nSummary of monthly metrics...\n\n", MoDataDF[file].describe(), "\n\nAnnual Monthly Averages...\n\n", MonthlyAverages[file])
        
    WC_WYDataDF = WYDataDF['Wildcat']
    WC_WYDataDF['Station'] = 'Wildcat'
        
    TR_WYDataDF = WYDataDF['Tippe']
    TR_WYDataDF['Station']='Tippe'
    WC_WYDataDF = WC_WYDataDF.append(TR_WYDataDF)
    #Output corrected annual average streamflow data to CSV    
    WC_WYDataDF.to_csv('Annual_Metrics.csv', sep= ',', index =True )

    WC_mo = MoDataDF['Wildcat']
    WC_mo['Station']='Wildcat'
    TR_mo = MoDataDF['Tippe']
    TR_mo['Station']='Tippe'
    WC_mo = WC_mo.append(TR_mo)
    #Output corrected monthly average streamflow data to CSV
    WC_mo.to_csv('Monthly_Metrics.csv', sep= ',', index =True )
      
    WC_anavg = AnnualAverages['Wildcat']
    WC_anavg['Station']='Wildcat'
    TR_anavg = AnnualAverages['Tippe']
    TR_anavg['Station'] = 'Tippe'
    WC_anavg = WC_anavg.append(TR_anavg) 
   #Output corrected annual average streamflow data to TAB delimited .txt file 
    WC_anavg.to_csv('Average_Annual_Metrics.txt', sep='\t', index = True)   
    
    WC_moavg=MonthlyAverages['Wildcat']
    WC_moavg['Station']='Wildcat'
    TR_moavg = MonthlyAverages['Tippe']
    TR_moavg['Station'] = 'Tippe'
    WC_moavg=WC_moavg.append(TR_moavg)
    #Output corrected monthly average streamflow data to TAB delimited .txt file
    WC_moavg.to_csv('Average_Monthly_Metrics.txt', sep='\t', index = True)    
                