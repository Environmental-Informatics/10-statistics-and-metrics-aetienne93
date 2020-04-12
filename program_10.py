#!/bin/env python
# Created on March 25, 2020
#  by Keith Cherkauer
#
# This script servesa as the solution set for assignment-10 on descriptive
# statistics and environmental informatics.  See the assignment documention 
# and repository at:
# https://github.com/Environmental-Informatics/assignment-10.git for more
# details about the assignment.


###Definitions and output files created by Aaron Etienne, in accordance with
###ABE 651 assignment 10 guidelines 
###April 4, 2020
###Github: aetienne Purdue: aetienne

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
    '''
    #'Lack of sanity' check from Cherkauer's test code
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
    '''
   # create global variables 
    #global DataDF
    #global MissingValues
    
    # define column names
    colNames = ['agency_cd', 'site_no', 'Date', 'Discharge', 'Quality']

    # open and read the file
    DataDF = pd.read_csv(fileName, header=1, names=colNames,  
                         delimiter=r"\s+",parse_dates=[2], comment='#',
                         na_values=['Eqp'])
    DataDF = DataDF.set_index('Date')
    
    # quantify the number of missing values
    #I learned this in the last assignment, replaces any values that fall below zero
    #with NAN- and it actually works when graphing
    '''
    for i in range(0, len(DataDF)-1):
        if DataDF['Discharge'].iloc[i] <= 0:
            DataDF['Discharge'].iloc[i] = np.nan
    '''
    #Debugging from cherkauer 
    # check for neagtive discharge
    DataDF["Discharge"] = DataDF["Discharge"].where( DataDF["Discharge"] >= 0 )
    

    
    MissingValues = DataDF["Discharge"].isna().sum()
    
    return( DataDF, MissingValues )
    
    #global DataDF
    #global MissingValues
    print(DataDF)

def ClipData( DataDF, startDate, endDate ):
    """This function clips the given time series dataframe to a given range 
    of dates. Function returns the clipped dataframe and and the number of 
    missing values."""
   
    
    #Make an index specific to the date range in the DataDF
    #startDate = DataDF.index['1969-10-01 00:00:00']
    #endDate = DataDF.index['2019-09-30 00:00:00']
    #DataDF = DataDF.loc['1969-10-01':'2019-09-30']

    #Use pandas.DataFrame.clip maybe it'll work for once 
    #DataFrame.clip(lower=None, upper=None, axis=None, inplace=False, *args, **kwargs))
    DataDF.loc[startDate:endDate]
    #Get sum of NaN values after removing error data 
    MissingValues = DataDF["Discharge"].isna().sum()
    
    return( DataDF, MissingValues )
     
def CalcTqmean(Qvalues):
    """This function computes the TQmean of a series of data, typically
       a 1 year time series of streamflow, after filtering out NoData
       values.  TQmean is the fraction of time that daily streamflow
       exceeds mean streamflow for each year. TQmean is based on the
       duration rather than the volume of streamflow. The routine returns
       the TQmean value for the given data array."""
    '''
    Qvalues = Qvalues.dropna()  
    TQmean = ((Qvalues > Qvalues.mean()).sum()/len(Qvalues))
    return ( TQmean )
    '''
    #Debugging from Cherkauer 
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
    '''
    av = 0
    Qvalues = Qvalues.dropna()
    if sum(Qvalues) > 0:
        for i in range (1, len(Qvalues)):
            av=av+abs(Qvalues[i-1]-Qvalues[i])
        RBindex = av/sum(Qvalues)
    else:
        RBindex = np.nan
    return ( RBindex )
    '''
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
    '''   
    Qvalues = Qvalues.dropna()
    val7Q =  min(Qvalues.resample('7D').mean())    
    return ( val7Q )
    '''
    #tried cherkauer's for debugging
    tmpRolling = Qvalues.rolling(window=7).mean()
    return ( tmpRolling.min() )
def CalcExceed3TimesMedian(Qvalues):
    """This function computes the number of days with flows greater 
       than 3 times the annual median flow. The index is calculated by 
       computing the median flow from the given dataset (or using the value
       provided) and then counting the number of days with flow greater than 
       3 times that value.   The routine returns the count of events greater 
       than 3 times the median annual flow value for the given data array."""
    #Debugging from Cherkauer 
    Qmedian = Qvalues.median()
    tmpCount = Qvalues.where( Qvalues > 3. * Qmedian )
    return ( ( tmpCount / tmpCount ).sum() )
    '''
    Qvalues = Qvalues.dropna()
    median3x = (Qvalues > (Qvalues.median()*3)).sum()
    return ( median3x )
    '''
def GetAnnualStatistics(DataDF):
    """This function calculates annual descriptive statistcs and metrics for 
    the given streamflow time series.  Values are retuned as a dataframe of
    annual values for each water year.  Water year, as defined by the USGS,
    starts on October 1."""
    
    #Per assignment 10 netrics explanation 
    #colNames = ['site_no', 'Mean Flow', 'Peak Flow', 'Median Flow', 'Coeff Var', 
               #'Skew', 'TQmean', 'R-B Index', '7Q', '3xMedian']
    
    
    #Debugging from Cherkauer 
    WYDataDF = DataDF.resample('AS-OCT').mean()
    WYDataDF.rename(columns={"Discharge":"Mean Flow"}, inplace=True)
    WYDataDF["Peak Flow"] = DataDF["Discharge"].resample('AS-OCT').max()
    WYDataDF["Median"] = DataDF["Discharge"].resample('AS-OCT').median()
    WYDataDF["Coeff Var"] = DataDF["Discharge"].resample('AS-OCT').std() / WYDataDF["Mean Flow"] * 100.
    WYDataDF["Skew"] = DataDF["Discharge"].resample('AS-OCT').apply(stats.skew)
    WYDataDF["TQmean"] = DataDF["Discharge"].resample('AS-OCT').apply(CalcTqmean)
    WYDataDF["R-B Index"] = DataDF["Discharge"].resample('AS-OCT').apply(CalcRBindex)
    WYDataDF["7Q"] = DataDF["Discharge"].resample('AS-OCT').apply(Calc7Q)
    WYDataDF["3xMedian"] = DataDF["Discharge"].resample('AS-OCT').apply(CalcExceed3TimesMedian)
    
    '''
    annualData = DataDF.resample('AS-OCT').mean()
    
    WYDataDF = pd.DataFrame(0, index = annualData.index, columns = colNames)
    #use mean for site_no and mean flow
    WYDataDF['site_no'] = DataDF.resample('AS-OCT')['site_no'].mean()
    WYDataDF['Mean Flow'] = DataDF.resample('AS-OCT')['Discharge'].mean()
    #use max discharge for peak flow
    WYDataDF['Peak Flow'] = DataDF.resample('AS-OCT')['Discharge'].max()
    #median discharge for median flow
    WYDataDF['Median Flow'] = DataDF.resample('AS-OCT')['Discharge'].median()
    #std dev/ mean * 100 gives coefficent of variation 
    WYDataDF['Coeff Var'] = (DataDF.resample('AS-OCT')['Discharge'].std()/ 
            DataDF.resample('AS-OCT')['Discharge'].mean())*100
    #Use scipy stats skew function to find sqew
    WYDataDF['Skew'] = DataDF.resample('AS-OCT').apply({'Discharge': lambda x: 
        stats.skew(x, nan_policy='omit', bias=False)}, raw= True)
    #Apply user defined CalcTQmean function to calculate T-Qmean
    WYDataDF['TQmean'] = DataDF.resample('AS-OCT').apply({'Discharge': lambda x: 
        CalcTQmean(x)}) 
    #Apply user defined CalcRBimdex function to calculate Richards-Baker Flashiness Index
    WYDataDF['R-B Index'] = DataDF.resample('AS-OCT').apply({'Discharge': lambda x: 
        CalcRBindex(x)})
    #Apply user defined Calc7Q function to calculate 7-day Low Flow 
    WYDataDF['7Q'] = DataDF.resample('AS-OCT').apply({'Discharge': lambda x: 
        Calc7Q(x)})
    #Apply user defined CalcExceed3TimesMedian function to calculate flow exceeding 3 times median flow 
    WYDataDF['7Q'] = DataDF.resample('AS-OCT').apply({'Discharge': lambda x: 
        CalcExceed3TimesMedian(x)})  
    '''
   

    
    return ( WYDataDF )


def GetMonthlyStatistics(DataDF):
    """This function calculates monthly descriptive statistics and metrics 
    for the given streamflow time series.  Values are returned as a dataframe
    of monthly values for each year."""
    
    #deubugging from Cherkauer 
    
    MoDataDF = DataDF.resample('MS').mean()
    MoDataDF.rename(columns={"Discharge":"Mean Flow"}, inplace=True)
    MoDataDF["Coeff Var"] = DataDF["Discharge"].resample('MS').std() / MoDataDF["Mean Flow"] * 100.
    MoDataDF["TQmean"] = DataDF["Discharge"].resample('MS').apply(CalcTqmean)
    MoDataDF["R-B Index"] = DataDF["Discharge"].resample('MS').apply(CalcRBindex)
    
    
    '''
    colNames = ['site_no', 'Mean Flow', 'Coeff Var', 'TQmean', 'R-B Index']
    monthStat = DataDF.resample('M').mean()
    #Structure the month data dataframe 
    MoDataDF = pd.DataFrame(0, index=monthStat.index, columns=colNames)
    #Resample site number for month 
    MoDataDF['site_no']=DataDF.resample('M')['site_no'].mean()
    #resample discharge mean to monthly for mean flow 
    MoDataDF['Mean Flow']=DataDF.resample('M')['Discharge'].mean()
    #resample coefficent variable created in last definition to monthly
    MoDataDF['Coeff Var'] = (DataDF.resample('M')['Discharge'].std()/ 
            DataDF.resample('M')['Discharge'].mean())*100
    #resample TQmean created in last definition to monthly 
    MoDataDF['TQmean'] = DataDF.resample('M').apply({'Discharge': lambda x: 
        CalcTQmean(x)})
    #resample R-Bindex created in last definition to monthly 
    MoDataDF['R-B Index'] = DataDF.resample('M').apply({'Discharge': lambda x: 
        CalcRBindex(x)})
    '''
    return ( MoDataDF )

def GetAnnualAverages(WYDataDF):
    """This function calculates annual average values for all statistics and
    metrics.  The routine returns an array of mean values for each metric
    in the original dataframe."""
    #specify WYDataDF mean axis to be zero. This gives annual average
    
    #AnnualAverages = WYDataDF.mean(axis=0)
    AnnualAverages = WYDataDF.mean()
    return( AnnualAverages )

def GetMonthlyAverages(MoDataDF):
    """This function calculates annual average monthly values for all 
    statistics and metrics.  The routine returns an array of mean values 
    for each metric in the original dataframe."""
    #Calculate averages for all 12 months of 'site_no', 'Mean Flow', 
    #'Coeff Var', 'TQmean', 'R-B Index'
    #Debugging from cherkauer 
    months = MoDataDF.index.month
    MonthlyAverages = MoDataDF.groupby(months).mean()
    
    
    '''
    colNames = ['site_no', 'Mean Flow', 'Coeff Var', 'TQmean', 'R-B Index']
    MonthlyAverages = pd.DataFrame(0, index=range(1,13), columns = colNames)
    a=[3,4,5,6,7,8,9,10,11,0,1,2]
    idx=0
    for i in range(12):
        MonthlyAverages.iloc[idx,0] = MoDataDF['site_no'][::12].mean()
        MonthlyAverages.iloc[idx,1] = MoDataDF['Mean Flow'][a[idx]::12].mean()
        MonthlyAverages.iloc[idx,2] = MoDataDF['Coeff Var'][a[idx]::12].mean()
        MonthlyAverages.iloc[idx,3] = MoDataDF['TQmean'][a[idx]::12].mean()
        MonthlyAverages.iloc[idx,4] = MoDataDF['R-B Index'][a[idx]::12].mean()
        idx +=1
    '''
    '''
    #run iterativelythrough each column of the third row in the DataFrame. 
    #Site number should be the same for each month here 
    #Start with the site number 
    MonthlyAverages.iloc[0,0]=MoDataDF['site_no'][::12].mean()
    MonthlyAverages.iloc[1,0]=MoDataDF['site_no'][::12].mean()
    MonthlyAverages.iloc[2,0]=MoDataDF['site_no'][::12].mean()
    MonthlyAverages.iloc[3,0]=MoDataDF['site_no'][::12].mean()
    MonthlyAverages.iloc[4,0]=MoDataDF['site_no'][::12].mean()
    MonthlyAverages.iloc[5,0]=MoDataDF['site_no'][::12].mean()
    MonthlyAverages.iloc[6,0]=MoDataDF['site_no'][::12].mean()
    MonthlyAverages.iloc[7,0]=MoDataDF['site_no'][::12].mean()
    MonthlyAverages.iloc[8,0]=MoDataDF['site_no'][::12].mean()
    MonthlyAverages.iloc[9,0]=MoDataDF['site_no'][::12].mean()
    MonthlyAverages.iloc[10,0]=MoDataDF['site_no'][::12].mean()
    MonthlyAverages.iloc[11,0]=MoDataDF['site_no'][::12].mean()
    
    #Get the mean flow for each month 
    MonthlyAverages.iloc[0,1]=MoDataDF['Mean Flow'][3::12].mean()
    MonthlyAverages.iloc[1,1]=MoDataDF['Mean Flow'][4::12].mean()
    MonthlyAverages.iloc[2,1]=MoDataDF['Mean Flow'][5::12].mean()
    MonthlyAverages.iloc[3,1]=MoDataDF['Mean Flow'][6::12].mean()
    MonthlyAverages.iloc[4,1]=MoDataDF['Mean Flow'][7::12].mean()
    MonthlyAverages.iloc[5,1]=MoDataDF['Mean Flow'][8::12].mean()
    MonthlyAverages.iloc[6,1]=MoDataDF['Mean Flow'][9::12].mean()
    MonthlyAverages.iloc[7,1]=MoDataDF['Mean Flow'][10::12].mean()
    MonthlyAverages.iloc[8,1]=MoDataDF['Mean Flow'][11::12].mean() 
    MonthlyAverages.iloc[9,1]=MoDataDF['Mean Flow'][::12].mean()
    MonthlyAverages.iloc[10,1]=MoDataDF['Mean Flow'][1::12].mean()
    MonthlyAverages.iloc[11,1]=MoDataDF['Mean Flow'][2::12].mean()
    
    #Get the coefficient of variation for each month 
    MonthlyAverages.iloc[0,2]=MoDataDF['Coeff Var'][3::12].mean()
    MonthlyAverages.iloc[1,2]=MoDataDF['Coeff Var'][4::12].mean()
    MonthlyAverages.iloc[2,2]=MoDataDF['Coeff Var'][5::12].mean()
    MonthlyAverages.iloc[3,2]=MoDataDF['Coeff Var'][6::12].mean()
    MonthlyAverages.iloc[4,2]=MoDataDF['Coeff Var'][7::12].mean()
    MonthlyAverages.iloc[5,2]=MoDataDF['Coeff Var'][8::12].mean()
    MonthlyAverages.iloc[6,2]=MoDataDF['Coeff Var'][9::12].mean()
    MonthlyAverages.iloc[7,2]=MoDataDF['Coeff Var'][10::12].mean()
    MonthlyAverages.iloc[8,2]=MoDataDF['Coeff Var'][11::12].mean() 
    MonthlyAverages.iloc[9,2]=MoDataDF['Coeff Var'][::12].mean()
    MonthlyAverages.iloc[10,2]=MoDataDF['Coeff Var'][1::12].mean()
    MonthlyAverages.iloc[11,2]=MoDataDF['Coeff Var'][2::12].mean()
    
    #Get the TQmean for each month 
    MonthlyAverages.iloc[0,3]=MoDataDF['TQmean'][3::12].mean()
    MonthlyAverages.iloc[1,3]=MoDataDF['TQmean'][4::12].mean()
    MonthlyAverages.iloc[2,3]=MoDataDF['TQmean'][5::12].mean()
    MonthlyAverages.iloc[3,3]=MoDataDF['TQmean'][6::12].mean()
    MonthlyAverages.iloc[4,3]=MoDataDF['TQmean'][7::12].mean()
    MonthlyAverages.iloc[5,3]=MoDataDF['TQmean'][8::12].mean()
    MonthlyAverages.iloc[6,3]=MoDataDF['TQmean'][9::12].mean()
    MonthlyAverages.iloc[7,3]=MoDataDF['TQmean'][10::12].mean()
    MonthlyAverages.iloc[8,3]=MoDataDF['TQmean'][11::12].mean() 
    MonthlyAverages.iloc[9,3]=MoDataDF['TQmean'][::12].mean()
    MonthlyAverages.iloc[10,3]=MoDataDF['TQmean'][1::12].mean()
    MonthlyAverages.iloc[11,3]=MoDataDF['TQmean'][2::12].mean()    
    
    #Get R-B index for each  month 
    MonthlyAverages.iloc[0,4]=MoDataDF['R-B Index'][3::12].mean()
    MonthlyAverages.iloc[1,4]=MoDataDF['R-B Index'][4::12].mean()
    MonthlyAverages.iloc[2,4]=MoDataDF['R-B Index'][5::12].mean()
    MonthlyAverages.iloc[3,4]=MoDataDF['R-B Index'][6::12].mean()
    MonthlyAverages.iloc[4,4]=MoDataDF['R-B Index'][7::12].mean()
    MonthlyAverages.iloc[5,4]=MoDataDF['R-B Index'][8::12].mean()
    MonthlyAverages.iloc[6,4]=MoDataDF['R-B Index'][9::12].mean()
    MonthlyAverages.iloc[7,4]=MoDataDF['R-B Index'][10::12].mean()
    MonthlyAverages.iloc[8,4]=MoDataDF['R-B Index'][11::12].mean() 
    MonthlyAverages.iloc[9,4]=MoDataDF['R-B Index'][::12].mean()
    MonthlyAverages.iloc[10,4]=MoDataDF['R-B Index'][1::12].mean()
    MonthlyAverages.iloc[11,4]=MoDataDF['R-B Index'][2::12].mean()     
    '''
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
        
        
    #Test user defined functions for each data set
     
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
                