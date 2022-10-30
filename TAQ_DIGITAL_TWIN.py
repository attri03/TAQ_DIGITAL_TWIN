#!/usr/bin/env python
# coding: utf-8

# # TAQ 5005

# In[1]:


import pandas as pd
df = pd.read_csv('E:/5005.csv')
df.head(50)

#Splitting the Time variable into date and time
data2 = df.Time.apply(lambda x: pd.Series(str(x).split(" ")))
df['Day...'] = data2[0]

#Removing the duplicates
df = df.drop_duplicates()

#Creating datasets with only values having non zero current
df=df[df['Current Load(A)']!=0]

#Dropping unwanted columns to remove curse of dimensionality
df = df.drop(['Cycles Remaining','Time To Discharge','Time To Full Charge'],axis=1)

#Setting 'Capacity' variable to a Value of 30
df['Capacity'] = 30

#Adding one more column named 'Current Capacity' and assigning it a value using formula
df['Current Capacity'] = df['Capacity'] * df['Soc(%)']/100

#Finding number of 'Days' using substraction method for dates
from datetime import datetime
df['Day...'] =  pd.to_datetime(df['Day...'], infer_datetime_format=True)
df['Day...']=pd.to_datetime(df['Day...'], format='%d-%m-%Y')
a = []

for ind in df.index:
    delta = df['Day...'][ind]-df['Day...'].iloc[0]
    a.append(delta.days)
    
df['Days'] = a

#Converting 'Days' so that we are multiplying it by 24
df['Days'] = df['Days']*24

#Converting Time to 'DateTime' format
df['Time'] =  pd.to_datetime(df['Time'], infer_datetime_format=True)
df['Time']=pd.to_datetime(df['Time'], format='%d-%m-%Y hh:mm:ss')

#Making a variable and storing 'Time' in it
data3 = df.Time.apply(lambda x: pd.Series(str(x).split(" ")))
df['Time_'] = data3[1]

data4 = df.Time_.apply(lambda x: pd.Series(str(x).split(":")))

data4[0]=data4[0].astype(int)

df['hh'] = data4[0]
df['mm'] = data4[1]
df['ss'] = data4[2]

df['hh'] = df['hh']+df['Days']

df['Time2'] = df['hh'].astype(str) + ':' + df['mm'] + ':' + df['ss']

df = df.drop(['hh','mm','ss','Time_','Day...','Time'],axis=1)
df = df.drop('SoH(%)',axis=1)

#Creating 'HH:MM' feature which is basically converting existing time to float for further engineering

data6 = df.Time2.apply(lambda x: pd.Series(str(x).split(":")))
df['hh'] = data6[0]
df['mm'] = data6[1]
df['ss'] = data6[2]
df['mm'] = df['mm'].astype(float)
df['mm']= df['mm']*(1/60)
df['mm'] = df['mm'].round(2)
data8 = df['mm'].apply(lambda x: pd.Series(str(x).split(".")))
df['MM'] = data8[1]
df['HH:MM'] = df['hh'] + '.' + df['MM'].astype(str)
df = df.drop(['hh','mm','ss','MM','Geolocation'],axis=1)

df1 =df
df1['HH:MM'] = df1['HH:MM'].astype(float)

# Differentiating cycles using 'HH:MM' to get best results of differentiations 
r_time=[]
a_time = df1.index[0]
x_time = df1.index[0]
for n_time in df1.index:
    if df1['HH:MM'][n_time] >= (df1['HH:MM'][a_time] + 1):
        dataset = df1.loc[x_time:a_time]
        arr_time = dataset.to_numpy()
        r_time.append(arr_time)
        x_time = n_time
    a_time=n_time

#Converting the numpy array dataframe to pandas
for t_time in range(0,len(r_time)):
    r_time[t_time] = pd.DataFrame(r_time[t_time])
    
#Adding the column names to the dataset
for x in range (0,len(r_time)):
    r_time[x].columns =['Voltage(V)', 'Soc(%)', 'Current Load(A)', 'Charging Current(A)','Battery Temperature','Capacity','Cycles Used','Speed(Kmph)','Total Distance(Km)','Trip Distance(Km)','Power(Watts)','Current Capacity','Days','Time2','HH:MM']
    
# Seperating the charging the discharging from cycles
charging_r_time = []
discharging_r_time = []
for r in range(0,len(r_time)):
    dataset_charging = r_time[r][r_time[r]['Current Load(A)'] > 0]
    arr_charging = dataset_charging.to_numpy()
    charging_r_time.append(arr_charging)
    dataset_discharging = r_time[r][r_time[r]['Current Load(A)'] < 0]
    arr_discharging = dataset_discharging.to_numpy()
    discharging_r_time.append(arr_discharging)
    

#Converting all the array datasets to pandas dataframe
for t in range(0,len(charging_r_time)):
    charging_r_time[t] = pd.DataFrame(charging_r_time[t])
    
for t in range(0,len(discharging_r_time)):
    discharging_r_time[t] = pd.DataFrame(discharging_r_time[t])
    
#Adding the column names to the dataset 
for x in range (0,len(r_time)):
    charging_r_time[x].columns =['Voltage(V)', 'Soc(%)', 'Current Load(A)', 'Charging Current(A)','Battery Temperature','Capacity','Cycles Used','Speed(Kmph)','Total Distance(Km)','Trip Distance(Km)','Power(Watts)','Current Capacity','Days','Time2','HH:MM']
    discharging_r_time[x].columns =['Voltage(V)', 'Soc(%)', 'Current Load(A)', 'Charging Current(A)','Battery Temperature','Capacity','Cycles Used','Speed(Kmph)','Total Distance(Km)','Trip Distance(Km)','Power(Watts)','Current Capacity','Days','Time2','HH:MM']
    
data = charging_r_time[0].append(charging_r_time[1])
data = data.append(charging_r_time[3])
data = data.append(charging_r_time[4])
data = data.append(charging_r_time[5])
data = data.append(charging_r_time[6])
data = data.append(charging_r_time[9])
data = data.append(charging_r_time[20])
data = data.append(charging_r_time[22])
data = data.append(charging_r_time[23])
data = data.append(charging_r_time[24])
data = data.append(charging_r_time[25])
data = data.append(charging_r_time[26])
data = data.append(charging_r_time[27])
data = data.append(charging_r_time[28])
data = data.append(charging_r_time[29])

charging_set_1 = data

for ind in charging_set_1.columns:
    if ind != 'Time2':
        charging_set_1[ind] = charging_set_1[ind].astype(float)


# # SOME DATA ANALYSIS OF TAQ 5005

# In[2]:


#Storing charging_set_1 in another variable so that it does not effect our original charging_set_1
data_analysis = charging_set_1
data_analysis = charging_set_1.reset_index()
data_analysis = data_analysis.drop('index',axis=1)
data_analysis


# In[3]:


#We are creating histogram to actually know about the distribution of our data points
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

for ind in data_analysis.columns:
    plt.hist(data_analysis[ind])
    plt.title(ind)
    plt.show()


# In[4]:


#We are plotting scatter points to actually know about the effect of change of different variables with Current Capacity, most relevant one is Voltage(V) vs Current Capacity
for ind in data_analysis.columns:
    plt.scatter(x = data_analysis[ind], y = data_analysis['Current Capacity'])
    plt.xlabel(ind)
    plt.ylabel('Current Capacity')
    plt.title(ind)
    plt.show()


# In[5]:


#Creating Heat Map to find which all functions are actually correlated with each other
import seaborn as sns
uuu = data_analysis.drop(['Charging Current(A)','Capacity','Cycles Used','Days','HH:MM'], axis = 1)
corrmat = uuu.corr()
top_corr_features = corrmat.index
plt.figure(figsize = (10,6))
g = sns.heatmap(uuu[top_corr_features].corr(), annot = True)


# # TAQ 5000

# In[6]:


import pandas as pd
df = pd.read_csv('E:/5000.csv')

#Splitting the Time variable into date and time
data2 = df.Time.apply(lambda x: pd.Series(str(x).split(" ")))
df['Day...'] = data2[0]

#Removing the duplicates
df = df.drop_duplicates()

#Creating datasets with only values having non zero current
df=df[df['Current Load(A)']!=0]

#Dropping unwanted columns to remove curse of dimensionality
df = df.drop(['Cycles Remaining','Time To Discharge','Time To Full Charge'],axis=1)

#Setting 'Capacity' variable to a Value of 30
df['Capacity'] = 30

#Adding one more column named 'Current Capacity' and assigning it a value using formula
df['Current Capacity'] = df['Capacity'] * df['Soc(%)']/100

#Finding number of 'Days' using substraction method for dates
from datetime import datetime
df['Day...'] =  pd.to_datetime(df['Day...'], infer_datetime_format=True)
df['Day...']=pd.to_datetime(df['Day...'], format='%d-%m-%Y')

a = []
for ind in df.index:
    r = df['Day...'][ind] - df['Day...'].iloc[0]
    a.append(r.days)
    
df['Days'] = a

#Converting 'Days' so that we are multiplying it by 24
df['Days'] = df['Days']*24

#Converting Time to 'DateTime' format
df['Time'] =  pd.to_datetime(df['Time'], infer_datetime_format=True)
df['Time']=pd.to_datetime(df['Time'], format='%d-%m-%Y hh:mm:ss')

#Making a variable and storing 'Time' in it
data3 = df.Time.apply(lambda x: pd.Series(str(x).split(" ")))
df['Time_'] = data3[1]

data4 = df.Time_.apply(lambda x: pd.Series(str(x).split(":")))

data4[0]=data4[0].astype(int)

df['hh'] = data4[0]
df['mm'] = data4[1]
df['ss'] = data4[2]

df['hh'] = df['hh']+df['Days']

df['Time2'] = df['hh'].astype(str) + ':' + df['mm'] + ':' + df['ss']

df = df.drop(['hh','mm','ss','Time_','Day...','Time'],axis=1)

#Creating 'HH:MM' feature which is basically converting existing time to float for further engineering
data6 = df.Time2.apply(lambda x: pd.Series(str(x).split(":")))
df['hh'] = data6[0]
df['mm'] = data6[1]
df['ss'] = data6[2]
df['mm'] = df['mm'].astype(float)
df['mm']= df['mm']*(1/60)
df['mm'] = df['mm'].round(2)
data8 = df['mm'].apply(lambda x: pd.Series(str(x).split(".")))
df['MM'] = data8[1]
df['HH:MM'] = df['hh'] + '.' + df['MM'].astype(str)
df = df.drop(['hh','mm','ss','MM','Geolocation'],axis=1)

#Saving data set to another dataset for further engineering and converting 'HH:MM' to float 

df1 =df
df1['HH:MM'] = df1['HH:MM'].astype(float)

# Differentiating cycles using 'HH:MM' to get best results of differentiations 
r_time=[]
a_time = df1.index[0]
x_time = df1.index[0]
for n_time in df1.index:
    if df1['HH:MM'][n_time] >= (df1['HH:MM'][a_time] + 1):
        dataset = df1.loc[x_time:a_time]
        arr_time = dataset.to_numpy()
        r_time.append(arr_time)
        x_time = n_time
    a_time=n_time
    
#Converting the numpy array dataframe to pandas
for t_time in range(0,len(r_time)):
    r_time[t_time] = pd.DataFrame(r_time[t_time])
    
#Adding the column names to the dataset
for x in range (0,len(r_time)):
    r_time[x].columns =['Voltage(V)', 'Soc(%)', 'Current Load(A)', 'Charging Current(A)','Battery Temperature','Capacity','Cycles Used','Speed(Kmph)','Total Distance(Km)','Trip Distance(Km)','Power(Watts)','Current Capacity','Days','Time2','HH:MM']
    
# Seperating the charging the discharging from cycles
charging_r_time = []
discharging_r_time = []
for r in range(0,len(r_time)):
    dataset_charging = r_time[r][r_time[r]['Current Load(A)'] > 0]
    arr_charging = dataset_charging.to_numpy()
    charging_r_time.append(arr_charging)
    dataset_discharging = r_time[r][r_time[r]['Current Load(A)'] < 0]
    arr_discharging = dataset_discharging.to_numpy()
    discharging_r_time.append(arr_discharging)
    
#Converting all the array datasets to pandas dataframe
for t in range(0,len(charging_r_time)):
    charging_r_time[t] = pd.DataFrame(charging_r_time[t])
    
for t in range(0,len(discharging_r_time)):
    discharging_r_time[t] = pd.DataFrame(discharging_r_time[t])
    
#Adding the column names to the dataset 
for x in range (0,len(r_time)):
    charging_r_time[x].columns =['Voltage(V)', 'Soc(%)', 'Current Load(A)', 'Charging Current(A)','Battery Temperature','Capacity','Cycles Used','Speed(Kmph)','Total Distance(Km)','Trip Distance(Km)','Power(Watts)','Current Capacity','Days','Time2','HH:MM']
    discharging_r_time[x].columns =['Voltage(V)', 'Soc(%)', 'Current Load(A)', 'Charging Current(A)','Battery Temperature','Capacity','Cycles Used','Speed(Kmph)','Total Distance(Km)','Trip Distance(Km)','Power(Watts)','Current Capacity','Days','Time2','HH:MM']
    
charging_set_2 = df1[df1['Current Load(A)'] > 0]

for ind in charging_set_2.columns:
    if ind != 'Time2':
        charging_set_2[ind] = charging_set_2[ind].astype(float)


# # SOME DATA ANALYSIS OF TAQ 5000

# In[7]:


#Storing charging_set_1 in another variable so that it does not effect our original charging_set_1
data_analysis = charging_set_2
data_analysis = charging_set_2.reset_index()
data_analysis = data_analysis.drop('index',axis=1)
data_analysis


# In[8]:


#We are creating histogram to actually know about the distribution of our data points
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

for ind in data_analysis.columns:
    plt.hist(data_analysis[ind])
    plt.title(ind)
    plt.show()


# In[9]:


#We are plotting scatter points to actually know about the effect of change of different variables with Current Capacity, most relevant one is Voltage(V) vs Current Capacity
for ind in data_analysis.columns:
    plt.scatter(x = data_analysis[ind], y = data_analysis['Current Capacity'])
    plt.xlabel(ind)
    plt.ylabel('Current Capacity')
    plt.title(ind)
    plt.show()


# In[10]:


#Creating Heat Map to find which all functions are actually correlated with each other
import seaborn as sns
uuu = data_analysis.drop(['Charging Current(A)','Capacity','Cycles Used','Days','HH:MM'], axis = 1)
corrmat = uuu.corr()
top_corr_features = corrmat.index
plt.figure(figsize = (10,6))
g = sns.heatmap(uuu[top_corr_features].corr(), annot = True)


# # TAQ 5002

# In[11]:


import pandas as pd
df = pd.read_csv('E:/5002.csv')

#Splitting the Time variable into date and time
data2 = df.Time.apply(lambda x: pd.Series(str(x).split(" ")))
df['Day...'] = data2[0]

#Removing the duplicates
df = df.drop_duplicates()

#Creating datasets with only values having non zero current
df=df[df['Current Load(A)']!=0]

#Removing the 'SoH'
df = df.drop('SoH(%)',axis=1)

#Dropping unwanted columns to remove curse of dimensionality
df = df.drop(['Cycles Remaining','Time To Discharge','Time To Full Charge'],axis=1)

#Setting 'Capacity' variable to a Value of 30
df['Capacity'] = 30

#Adding one more column named 'Current Capacity' and assigning it a value using formula
df['Current Capacity'] = df['Capacity'] * df['Soc(%)']/100

#Finding number of 'Days' using substraction method for dates
from datetime import datetime
df['Day...'] =  pd.to_datetime(df['Day...'], infer_datetime_format=True)
df['Day...']=pd.to_datetime(df['Day...'], format='%d-%m-%Y')

a = []
for ind in df.index:
    r = df['Day...'][ind] - df['Day...'].iloc[0]
    a.append(r.days)
    
df['Days'] = a

#Converting 'Days' so that we are multiplying it by 24
df['Days'] = df['Days']*24

#Converting Time to 'DateTime' format
df['Time'] =  pd.to_datetime(df['Time'], infer_datetime_format=True)
df['Time']=pd.to_datetime(df['Time'], format='%d-%m-%Y hh:mm:ss')

#Making a variable and storing 'Time' in it
data3 = df.Time.apply(lambda x: pd.Series(str(x).split(" ")))
df['Time_'] = data3[1]

data4 = df.Time_.apply(lambda x: pd.Series(str(x).split(":")))

data4[0]=data4[0].astype(int)

df['hh'] = data4[0]
df['mm'] = data4[1]
df['ss'] = data4[2]

df['hh'] = df['hh']+df['Days']

df['Time2'] = df['hh'].astype(str) + ':' + df['mm'] + ':' + df['ss']

df = df.drop(['hh','mm','ss','Time_','Day...','Time'],axis=1)

#Creating 'HH:MM' feature which is basically converting existing time to float for further engineering
data6 = df.Time2.apply(lambda x: pd.Series(str(x).split(":")))
df['hh'] = data6[0]
df['mm'] = data6[1]
df['ss'] = data6[2]
df['mm'] = df['mm'].astype(float)
df['mm']= df['mm']*(1/60)
df['mm'] = df['mm'].round(2)
data8 = df['mm'].apply(lambda x: pd.Series(str(x).split(".")))
df['MM'] = data8[1]
df['HH:MM'] = df['hh'] + '.' + df['MM'].astype(str)
df = df.drop(['hh','mm','ss','MM','Geolocation'],axis=1)

#Saving data set to another dataset for further engineering and converting 'HH:MM' to float 

df1 =df
df1['HH:MM'] = df1['HH:MM'].astype(float)

df1.head(50)

# Differentiating cycles using 'HH:MM' to get best results of differentiations 
r_time=[]
a_time = df1.index[0]
x_time = df1.index[0]
for n_time in df1.index:
    if df1['HH:MM'][n_time] >= (df1['HH:MM'][a_time] + 1):
        dataset = df1.loc[x_time:a_time]
        arr_time = dataset.to_numpy()
        r_time.append(arr_time)
        x_time = n_time
    a_time=n_time

#Converting the numpy array dataframe to pandas
for t_time in range(0,len(r_time)):
    r_time[t_time] = pd.DataFrame(r_time[t_time])

#Adding the column names to the dataset
for x in range (0,len(r_time)):
    r_time[x].columns =['Voltage(V)', 'Soc(%)', 'Current Load(A)', 'Charging Current(A)','Battery Temperature','Capacity','Cycles Used','Speed(Kmph)','Total Distance(Km)','Trip Distance(Km)','Power(Watts)','Current Capacity','Days','Time2','HH:MM']

# Seperating the charging the discharging from cycles
charging_r_time = []
discharging_r_time = []
for r in range(0,len(r_time)):
    dataset_charging = r_time[r][r_time[r]['Current Load(A)'] > 0]
    arr_charging = dataset_charging.to_numpy()
    charging_r_time.append(arr_charging)
    dataset_discharging = r_time[r][r_time[r]['Current Load(A)'] < 0]
    arr_discharging = dataset_discharging.to_numpy()
    discharging_r_time.append(arr_discharging)

#Converting all the array datasets to pandas dataframe
for t in range(0,len(charging_r_time)):
    charging_r_time[t] = pd.DataFrame(charging_r_time[t])
    
for t in range(0,len(discharging_r_time)):
    discharging_r_time[t] = pd.DataFrame(discharging_r_time[t])

#Adding the column names to the dataset 
for x in range (0,len(r_time)):
    charging_r_time[x].columns =['Voltage(V)', 'Soc(%)', 'Current Load(A)', 'Charging Current(A)','Battery Temperature','Capacity','Cycles Used','Speed(Kmph)','Total Distance(Km)','Trip Distance(Km)','Power(Watts)','Current Capacity','Days','Time2','HH:MM']
    discharging_r_time[x].columns =['Voltage(V)', 'Soc(%)', 'Current Load(A)', 'Charging Current(A)','Battery Temperature','Capacity','Cycles Used','Speed(Kmph)','Total Distance(Km)','Trip Distance(Km)','Power(Watts)','Current Capacity','Days','Time2','HH:MM']
    
data = charging_r_time[4].append(charging_r_time[5])
data = data.append(charging_r_time[6])
data = data.append(charging_r_time[7])
data = data.append(charging_r_time[8])
data = data.append(charging_r_time[9])
data = data.append(charging_r_time[12])
data = data.append(charging_r_time[13])
data = data.append(charging_r_time[14])
data = data.append(charging_r_time[15])
data = data.append(charging_r_time[17])
data = data.append(charging_r_time[18])
data = data.append(charging_r_time[19])
data = data.append(charging_r_time[20])
data = data.append(charging_r_time[21])
data = data.append(charging_r_time[22])
data = data.append(charging_r_time[23])
data = data.append(charging_r_time[29])
data = data.append(charging_r_time[30])
data = data.append(charging_r_time[31])
data = data.append(charging_r_time[32])

charging_set_3 = data

for ind in charging_set_3.columns:
    if ind != 'Time2':
        charging_set_3[ind] = charging_set_3[ind].astype(float)


# # SOME DATA ANALYSIS OF TAQ 5002

# In[12]:


#Storing charging_set_1 in another variable so that it does not effect our original charging_set_1
data_analysis = charging_set_3
data_analysis = charging_set_3.reset_index()
data_analysis = data_analysis.drop('index',axis=1)
data_analysis


# In[13]:


#We are creating histogram to actually know about the distribution of our data points
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

for ind in data_analysis.columns:
    plt.hist(data_analysis[ind])
    plt.title(ind)
    plt.show()


# In[14]:


#We are plotting scatter points to actually know about the effect of change of different variables with Current Capacity, most relevant one is Voltage(V) vs Current Capacity
for ind in data_analysis.columns:
    plt.scatter(x = data_analysis[ind], y = data_analysis['Current Capacity'])
    plt.xlabel(ind)
    plt.ylabel('Current Capacity')
    plt.title(ind)
    plt.show()


# In[15]:


#Creating Heat Map to find which all functions are actually correlated with each other
import seaborn as sns
uuu = data_analysis.drop(['Charging Current(A)','Capacity','Cycles Used','Days','HH:MM'], axis = 1)
corrmat = uuu.corr()
top_corr_features = corrmat.index
plt.figure(figsize = (10,6))
g = sns.heatmap(uuu[top_corr_features].corr(), annot = True)


# # TAQ 5007

# In[16]:


import pandas as pd
df = pd.read_csv('E:/5007.csv')

#Splitting the Time variable into date and time
data2 = df.Time.apply(lambda x: pd.Series(str(x).split(" ")))
df['Day...'] = data2[0]

#Removing the duplicates
df = df.drop_duplicates()

#Creating datasets with only values having non zero current
df=df[df['Current Load(A)']!=0]

#Removing the 'SoH'
df = df.drop('SoH(%)',axis=1)

#Dropping unwanted columns to remove curse of dimensionality
df = df.drop(['Cycles Remaining','Time To Discharge','Time To Full Charge'],axis=1)

#Setting 'Capacity' variable to a Value of 30
df['Capacity'] = 30

#Adding one more column named 'Current Capacity' and assigning it a value using formula
df['Current Capacity'] = df['Capacity'] * df['Soc(%)']/100

#Finding number of 'Days' using substraction method for dates
from datetime import datetime
df['Day...'] =  pd.to_datetime(df['Day...'], infer_datetime_format=True)
df['Day...']=pd.to_datetime(df['Day...'], format='%d-%m-%Y')

a = []
for ind in df.index:
    r = df['Day...'][ind] - df['Day...'].iloc[0]
    a.append(r.days)
    
df['Days'] = a

#Converting 'Days' so that we are multiplying it by 24
df['Days'] = df['Days']*24

#Converting Time to 'DateTime' format
df['Time'] =  pd.to_datetime(df['Time'], infer_datetime_format=True)
df['Time']=pd.to_datetime(df['Time'], format='%d-%m-%Y hh:mm:ss')

#Making a variable and storing 'Time' in it
data3 = df.Time.apply(lambda x: pd.Series(str(x).split(" ")))
df['Time_'] = data3[1]

data4 = df.Time_.apply(lambda x: pd.Series(str(x).split(":")))

data4[0]=data4[0].astype(int)

df['hh'] = data4[0]
df['mm'] = data4[1]
df['ss'] = data4[2]

df['hh'] = df['hh']+df['Days']

df['Time2'] = df['hh'].astype(str) + ':' + df['mm'] + ':' + df['ss']

df = df.drop(['hh','mm','ss','Time_','Day...','Time'],axis=1)

#Creating 'HH:MM' feature which is basically converting existing time to float for further engineering
data6 = df.Time2.apply(lambda x: pd.Series(str(x).split(":")))
df['hh'] = data6[0]
df['mm'] = data6[1]
df['ss'] = data6[2]
df['mm'] = df['mm'].astype(float)
df['mm']= df['mm']*(1/60)
df['mm'] = df['mm'].round(2)
data8 = df['mm'].apply(lambda x: pd.Series(str(x).split(".")))
df['MM'] = data8[1]
df['HH:MM'] = df['hh'] + '.' + df['MM'].astype(str)
df = df.drop(['hh','mm','ss','MM','Geolocation'],axis=1)

#Saving data set to another dataset for further engineering and converting 'HH:MM' to float 

df1 =df
df1['HH:MM'] = df1['HH:MM'].astype(float)

df1.head(50)

# Differentiating cycles using 'HH:MM' to get best results of differentiations 
r_time=[]
a_time = df1.index[0]
x_time = df1.index[0]
for n_time in df1.index:
    if df1['HH:MM'][n_time] >= (df1['HH:MM'][a_time] + 1):
        dataset = df1.loc[x_time:a_time]
        arr_time = dataset.to_numpy()
        r_time.append(arr_time)
        x_time = n_time
    a_time=n_time

#Converting the numpy array dataframe to pandas
for t_time in range(0,len(r_time)):
    r_time[t_time] = pd.DataFrame(r_time[t_time])

#Adding the column names to the dataset
for x in range (0,len(r_time)):
    r_time[x].columns =['Voltage(V)', 'Soc(%)', 'Current Load(A)', 'Charging Current(A)','Battery Temperature','Capacity','Cycles Used','Speed(Kmph)','Total Distance(Km)','Trip Distance(Km)','Power(Watts)','Current Capacity','Days','Time2','HH:MM']

# Seperating the charging the discharging from cycles
charging_r_time = []
discharging_r_time = []
for r in range(0,len(r_time)):
    dataset_charging = r_time[r][r_time[r]['Current Load(A)'] > 0]
    arr_charging = dataset_charging.to_numpy()
    charging_r_time.append(arr_charging)
    dataset_discharging = r_time[r][r_time[r]['Current Load(A)'] < 0]
    arr_discharging = dataset_discharging.to_numpy()
    discharging_r_time.append(arr_discharging)

#Converting all the array datasets to pandas dataframe
for t in range(0,len(charging_r_time)):
    charging_r_time[t] = pd.DataFrame(charging_r_time[t])
    
for t in range(0,len(discharging_r_time)):
    discharging_r_time[t] = pd.DataFrame(discharging_r_time[t])

#Adding the column names to the dataset 
for x in range (0,len(r_time)):
    charging_r_time[x].columns =['Voltage(V)', 'Soc(%)', 'Current Load(A)', 'Charging Current(A)','Battery Temperature','Capacity','Cycles Used','Speed(Kmph)','Total Distance(Km)','Trip Distance(Km)','Power(Watts)','Current Capacity','Days','Time2','HH:MM']
    discharging_r_time[x].columns =['Voltage(V)', 'Soc(%)', 'Current Load(A)', 'Charging Current(A)','Battery Temperature','Capacity','Cycles Used','Speed(Kmph)','Total Distance(Km)','Trip Distance(Km)','Power(Watts)','Current Capacity','Days','Time2','HH:MM']
    
data = charging_r_time[4].append(charging_r_time[5])
data = data.append(charging_r_time[6])
data = data.append(charging_r_time[7])
data = data.append(charging_r_time[8])
data = data.append(charging_r_time[9])
data = data.append(charging_r_time[12])
data = data.append(charging_r_time[13])
data = data.append(charging_r_time[14])
data = data.append(charging_r_time[15])
data = data.append(charging_r_time[17])
data = data.append(charging_r_time[18])
data = data.append(charging_r_time[19])
data = data.append(charging_r_time[20])
data = data.append(charging_r_time[21])
data = data.append(charging_r_time[22])
data = data.append(charging_r_time[23])
data = data.append(charging_r_time[29])
data = data.append(charging_r_time[30])
data = data.append(charging_r_time[31])
data = data.append(charging_r_time[32])

charging_set_4 = data

for ind in charging_set_4.columns:
    if ind != 'Time2':
        charging_set_4[ind] = charging_set_4[ind].astype(float)


# # SOME DATA ANALYSIS ON TAQ 5007

# In[17]:


#Storing charging_set_1 in another variable so that it does not effect our original charging_set_1
data_analysis = charging_set_4
data_analysis = charging_set_4.reset_index()
data_analysis = data_analysis.drop('index',axis=1)
data_analysis


# In[18]:


#We are creating histogram to actually know about the distribution of our data points
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

for ind in data_analysis.columns:
    plt.hist(data_analysis[ind])
    plt.title(ind)
    plt.show()


# In[19]:


#We are plotting scatter points to actually know about the effect of change of different variables with Current Capacity, most relevant one is Voltage(V) vs Current Capacity
for ind in data_analysis.columns:
    plt.scatter(x = data_analysis[ind], y = data_analysis['Current Capacity'])
    plt.xlabel(ind)
    plt.ylabel('Current Capacity')
    plt.title(ind)
    plt.show()


# In[20]:


#Creating Heat Map to find which all functions are actually correlated with each other
import seaborn as sns
uuu = data_analysis.drop(['Charging Current(A)','Capacity','Cycles Used','Days','HH:MM'], axis = 1)
corrmat = uuu.corr()
top_corr_features = corrmat.index
plt.figure(figsize = (10,6))
g = sns.heatmap(uuu[top_corr_features].corr(), annot = True)


# # TAQ 5041

# In[21]:


import pandas as pd
df = pd.read_csv('E:/5041.csv')

#Splitting the Time variable into date and time
data2 = df.Time.apply(lambda x: pd.Series(str(x).split(" ")))
df['Day...'] = data2[0]

#Removing the duplicates
df = df.drop_duplicates()

#Creating datasets with only values having non zero current
df=df[df['Current Load(A)']!=0]

#Removing the 'SoH'
df = df.drop(['SoH(%)','Unnamed: 0'],axis=1)

#Dropping unwanted columns to remove curse of dimensionality
df = df.drop(['Cycles Remaining','Time To Discharge','Time To Full Charge'],axis=1)

#Setting 'Capacity' variable to a Value of 30
df['Capacity'] = 30

#Adding one more column named 'Current Capacity' and assigning it a value using formula
df['Current Capacity'] = df['Capacity'] * df['Soc(%)']/100

#Finding number of 'Days' using substraction method for dates
from datetime import datetime
df['Day...'] =  pd.to_datetime(df['Day...'], infer_datetime_format=True)
df['Day...']=pd.to_datetime(df['Day...'], format='%d-%m-%Y')

a = []
for ind in df.index:
    r = df['Day...'][ind] - df['Day...'].iloc[0]
    a.append(r.days)
    
df['Days'] = a

#Converting 'Days' so that we are multiplying it by 24
df['Days'] = df['Days']*24

#Converting Time to 'DateTime' format
df['Time'] =  pd.to_datetime(df['Time'], infer_datetime_format=True)
df['Time']=pd.to_datetime(df['Time'], format='%d-%m-%Y hh:mm:ss')

#Making a variable and storing 'Time' in it
data3 = df.Time.apply(lambda x: pd.Series(str(x).split(" ")))
df['Time_'] = data3[1]

data4 = df.Time_.apply(lambda x: pd.Series(str(x).split(":")))

data4[0]=data4[0].astype(int)

df['hh'] = data4[0]
df['mm'] = data4[1]
df['ss'] = data4[2]

df['hh'] = df['hh']+df['Days']

df['Time2'] = df['hh'].astype(str) + ':' + df['mm'] + ':' + df['ss']

df = df.drop(['hh','mm','ss','Time_','Day...','Time'],axis=1)

#Creating 'HH:MM' feature which is basically converting existing time to float for further engineering
data6 = df.Time2.apply(lambda x: pd.Series(str(x).split(":")))
df['hh'] = data6[0]
df['mm'] = data6[1]
df['ss'] = data6[2]
df['mm'] = df['mm'].astype(float)
df['mm']= df['mm']*(1/60)
df['mm'] = df['mm'].round(2)
data8 = df['mm'].apply(lambda x: pd.Series(str(x).split(".")))
df['MM'] = data8[1]
df['HH:MM'] = df['hh'] + '.' + df['MM'].astype(str)
df = df.drop(['hh','mm','ss','MM','Geolocation'],axis=1)

#Saving data set to another dataset for further engineering and converting 'HH:MM' to float 

df1 =df
df1['HH:MM'] = df1['HH:MM'].astype(float)

df1.head(50)

# Differentiating cycles using 'HH:MM' to get best results of differentiations 
r_time=[]
a_time = df1.index[0]
x_time = df1.index[0]
for n_time in df1.index:
    if df1['HH:MM'][n_time] >= (df1['HH:MM'][a_time] + 1):
        dataset = df1.loc[x_time:a_time]
        arr_time = dataset.to_numpy()
        r_time.append(arr_time)
        x_time = n_time
    a_time=n_time

#Converting the numpy array dataframe to pandas
for t_time in range(0,len(r_time)):
    r_time[t_time] = pd.DataFrame(r_time[t_time])

#Adding the column names to the dataset
for x in range (0,len(r_time)):
    r_time[x].columns =['Voltage(V)', 'Soc(%)', 'Current Load(A)', 'Charging Current(A)','Battery Temperature','Capacity','Cycles Used','Speed(Kmph)','Total Distance(Km)','Trip Distance(Km)','Power(Watts)','Current Capacity','Days','Time2','HH:MM']

# Seperating the charging the discharging from cycles
charging_r_time = []
discharging_r_time = []
for r in range(0,len(r_time)):
    dataset_charging = r_time[r][r_time[r]['Current Load(A)'] > 0]
    arr_charging = dataset_charging.to_numpy()
    charging_r_time.append(arr_charging)
    dataset_discharging = r_time[r][r_time[r]['Current Load(A)'] < 0]
    arr_discharging = dataset_discharging.to_numpy()
    discharging_r_time.append(arr_discharging)

#Converting all the array datasets to pandas dataframe
for t in range(0,len(charging_r_time)):
    charging_r_time[t] = pd.DataFrame(charging_r_time[t])
    
for t in range(0,len(discharging_r_time)):
    discharging_r_time[t] = pd.DataFrame(discharging_r_time[t])

#Adding the column names to the dataset 
for x in range (0,len(r_time)):
    charging_r_time[x].columns =['Voltage(V)', 'Soc(%)', 'Current Load(A)', 'Charging Current(A)','Battery Temperature','Capacity','Cycles Used','Speed(Kmph)','Total Distance(Km)','Trip Distance(Km)','Power(Watts)','Current Capacity','Days','Time2','HH:MM']
    discharging_r_time[x].columns =['Voltage(V)', 'Soc(%)', 'Current Load(A)', 'Charging Current(A)','Battery Temperature','Capacity','Cycles Used','Speed(Kmph)','Total Distance(Km)','Trip Distance(Km)','Power(Watts)','Current Capacity','Days','Time2','HH:MM']
    
data = charging_r_time[2].append(charging_r_time[3])
data = data.append(charging_r_time[4])
data = data.append(charging_r_time[5])
data = data.append(charging_r_time[6])
data = data.append(charging_r_time[10])
data = data.append(charging_r_time[11])
data = data.append(charging_r_time[12])
data = data.append(charging_r_time[13])

charging_set_5 = data

for ind in charging_set_5.columns:
    if ind != 'Time2':
        charging_set_5[ind] = charging_set_5[ind].astype(float)


# # SOME DATA ANALYSIS ON TAQ 5041

# In[22]:


#Storing charging_set_1 in another variable so that it does not effect our original charging_set_1
data_analysis = charging_set_5
data_analysis = charging_set_5.reset_index()
data_analysis = data_analysis.drop('index',axis=1)
data_analysis


# In[23]:


#We are creating histogram to actually know about the distribution of our data points
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

for ind in data_analysis.columns:
    plt.hist(data_analysis[ind])
    plt.title(ind)
    plt.show()


# In[24]:


#We are plotting scatter points to actually know about the effect of change of different variables with Current Capacity, most relevant one is Voltage(V) vs Current Capacity
for ind in data_analysis.columns:
    plt.scatter(x = data_analysis[ind], y = data_analysis['Current Capacity'])
    plt.xlabel(ind)
    plt.ylabel('Current Capacity')
    plt.title(ind)
    plt.show()


# In[25]:


#Creating Heat Map to find which all functions are actually correlated with each other
import seaborn as sns
uuu = data_analysis.drop(['Charging Current(A)','Capacity','Cycles Used','Days','HH:MM'], axis = 1)
corrmat = uuu.corr()
top_corr_features = corrmat.index
plt.figure(figsize = (10,6))
g = sns.heatmap(uuu[top_corr_features].corr(), annot = True)


# # TAQ 5069

# In[26]:


import pandas as pd
df = pd.read_csv('E:/5041.csv')

#Splitting the Time variable into date and time
data2 = df.Time.apply(lambda x: pd.Series(str(x).split(" ")))
df['Day...'] = data2[0]

#Removing the duplicates
df = df.drop_duplicates()

#Creating datasets with only values having non zero current
df=df[df['Current Load(A)']!=0]

#Removing the 'SoH'
df = df.drop(['SoH(%)','Unnamed: 0'],axis=1)

#Dropping unwanted columns to remove curse of dimensionality
df = df.drop(['Cycles Remaining','Time To Discharge','Time To Full Charge'],axis=1)

#Setting 'Capacity' variable to a Value of 30
df['Capacity'] = 30

#Adding one more column named 'Current Capacity' and assigning it a value using formula
df['Current Capacity'] = df['Capacity'] * df['Soc(%)']/100

#Finding number of 'Days' using substraction method for dates
from datetime import datetime
df['Day...'] =  pd.to_datetime(df['Day...'], infer_datetime_format=True)
df['Day...']=pd.to_datetime(df['Day...'], format='%d-%m-%Y')

a = []
for ind in df.index:
    r = df['Day...'][ind] - df['Day...'].iloc[0]
    a.append(r.days)
    
df['Days'] = a

#Converting 'Days' so that we are multiplying it by 24
df['Days'] = df['Days']*24

#Converting Time to 'DateTime' format
df['Time'] =  pd.to_datetime(df['Time'], infer_datetime_format=True)
df['Time']=pd.to_datetime(df['Time'], format='%d-%m-%Y hh:mm:ss')

#Making a variable and storing 'Time' in it
data3 = df.Time.apply(lambda x: pd.Series(str(x).split(" ")))
df['Time_'] = data3[1]

data4 = df.Time_.apply(lambda x: pd.Series(str(x).split(":")))

data4[0]=data4[0].astype(int)

df['hh'] = data4[0]
df['mm'] = data4[1]
df['ss'] = data4[2]

df['hh'] = df['hh']+df['Days']

df['Time2'] = df['hh'].astype(str) + ':' + df['mm'] + ':' + df['ss']

df = df.drop(['hh','mm','ss','Time_','Day...','Time'],axis=1)

#Creating 'HH:MM' feature which is basically converting existing time to float for further engineering
data6 = df.Time2.apply(lambda x: pd.Series(str(x).split(":")))
df['hh'] = data6[0]
df['mm'] = data6[1]
df['ss'] = data6[2]
df['mm'] = df['mm'].astype(float)
df['mm']= df['mm']*(1/60)
df['mm'] = df['mm'].round(2)
data8 = df['mm'].apply(lambda x: pd.Series(str(x).split(".")))
df['MM'] = data8[1]
df['HH:MM'] = df['hh'] + '.' + df['MM'].astype(str)
df = df.drop(['hh','mm','ss','MM','Geolocation'],axis=1)

#Saving data set to another dataset for further engineering and converting 'HH:MM' to float 

df1 =df
df1['HH:MM'] = df1['HH:MM'].astype(float)

df1.head(50)

# Differentiating cycles using 'HH:MM' to get best results of differentiations 
r_time=[]
a_time = df1.index[0]
x_time = df1.index[0]
for n_time in df1.index:
    if df1['HH:MM'][n_time] >= (df1['HH:MM'][a_time] + 1):
        dataset = df1.loc[x_time:a_time]
        arr_time = dataset.to_numpy()
        r_time.append(arr_time)
        x_time = n_time
    a_time=n_time

#Converting the numpy array dataframe to pandas
for t_time in range(0,len(r_time)):
    r_time[t_time] = pd.DataFrame(r_time[t_time])

#Adding the column names to the dataset
for x in range (0,len(r_time)):
    r_time[x].columns =['Voltage(V)', 'Soc(%)', 'Current Load(A)', 'Charging Current(A)','Battery Temperature','Capacity','Cycles Used','Speed(Kmph)','Total Distance(Km)','Trip Distance(Km)','Power(Watts)','Current Capacity','Days','Time2','HH:MM']

# Seperating the charging the discharging from cycles
charging_r_time = []
discharging_r_time = []
for r in range(0,len(r_time)):
    dataset_charging = r_time[r][r_time[r]['Current Load(A)'] > 0]
    arr_charging = dataset_charging.to_numpy()
    charging_r_time.append(arr_charging)
    dataset_discharging = r_time[r][r_time[r]['Current Load(A)'] < 0]
    arr_discharging = dataset_discharging.to_numpy()
    discharging_r_time.append(arr_discharging)

#Converting all the array datasets to pandas dataframe
for t in range(0,len(charging_r_time)):
    charging_r_time[t] = pd.DataFrame(charging_r_time[t])
    
for t in range(0,len(discharging_r_time)):
    discharging_r_time[t] = pd.DataFrame(discharging_r_time[t])

#Adding the column names to the dataset 
for x in range (0,len(r_time)):
    charging_r_time[x].columns =['Voltage(V)', 'Soc(%)', 'Current Load(A)', 'Charging Current(A)','Battery Temperature','Capacity','Cycles Used','Speed(Kmph)','Total Distance(Km)','Trip Distance(Km)','Power(Watts)','Current Capacity','Days','Time2','HH:MM']
    discharging_r_time[x].columns =['Voltage(V)', 'Soc(%)', 'Current Load(A)', 'Charging Current(A)','Battery Temperature','Capacity','Cycles Used','Speed(Kmph)','Total Distance(Km)','Trip Distance(Km)','Power(Watts)','Current Capacity','Days','Time2','HH:MM']
    
data = charging_r_time[4].append(charging_r_time[5])
data = data.append(charging_r_time[6])
data = data.append(charging_r_time[10])
data = data.append(charging_r_time[11])
data = data.append(charging_r_time[12])
data = data.append(charging_r_time[13])

charging_set_6 = data

for ind in charging_set_6.columns:
    if ind != 'Time2':
        charging_set_6[ind] = charging_set_6[ind].astype(float)


# # SOME DATA ANALYSIS ON TAQ 5069

# In[27]:


#Storing charging_set_1 in another variable so that it does not effect our original charging_set_1
data_analysis = charging_set_6
data_analysis = charging_set_6.reset_index()
data_analysis = data_analysis.drop('index',axis=1)
data_analysis


# In[28]:


#We are creating histogram to actually know about the distribution of our data points
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

for ind in data_analysis.columns:
    plt.hist(data_analysis[ind])
    plt.title(ind)
    plt.show()


# In[29]:


#We are plotting scatter points to actually know about the effect of change of different variables with Current Capacity, most relevant one is Voltage(V) vs Current Capacity
for ind in data_analysis.columns:
    plt.scatter(x = data_analysis[ind], y = data_analysis['Current Capacity'])
    plt.xlabel(ind)
    plt.ylabel('Current Capacity')
    plt.title(ind)
    plt.show()


# In[30]:


#Creating Heat Map to find which all functions are actually correlated with each other
import seaborn as sns
uuu = data_analysis.drop(['Charging Current(A)','Capacity','Cycles Used','Days','HH:MM'], axis = 1)
corrmat = uuu.corr()
top_corr_features = corrmat.index
plt.figure(figsize = (10,6))
g = sns.heatmap(uuu[top_corr_features].corr(), annot = True)


# # TAQ 5070 TEST SET

# In[31]:


import pandas as pd
df = pd.read_csv('E:/5041.csv')

#Splitting the Time variable into date and time
data2 = df.Time.apply(lambda x: pd.Series(str(x).split(" ")))
df['Day...'] = data2[0]

#Removing the duplicates
df = df.drop_duplicates()

#Creating datasets with only values having non zero current
df=df[df['Current Load(A)']!=0]

#Removing the 'SoH'
df = df.drop(['SoH(%)','Unnamed: 0'],axis=1)

#Dropping unwanted columns to remove curse of dimensionality
df = df.drop(['Cycles Remaining','Time To Discharge','Time To Full Charge'],axis=1)

#Setting 'Capacity' variable to a Value of 30
df['Capacity'] = 30

#Adding one more column named 'Current Capacity' and assigning it a value using formula
df['Current Capacity'] = df['Capacity'] * df['Soc(%)']/100

#Finding number of 'Days' using substraction method for dates
from datetime import datetime
df['Day...'] =  pd.to_datetime(df['Day...'], infer_datetime_format=True)
df['Day...']=pd.to_datetime(df['Day...'], format='%d-%m-%Y')

a = []
for ind in df.index:
    r = df['Day...'][ind] - df['Day...'].iloc[0]
    a.append(r.days)
    
df['Days'] = a

#Converting 'Days' so that we are multiplying it by 24
df['Days'] = df['Days']*24

#Converting Time to 'DateTime' format
df['Time'] =  pd.to_datetime(df['Time'], infer_datetime_format=True)
df['Time']=pd.to_datetime(df['Time'], format='%d-%m-%Y hh:mm:ss')

#Making a variable and storing 'Time' in it
data3 = df.Time.apply(lambda x: pd.Series(str(x).split(" ")))
df['Time_'] = data3[1]

data4 = df.Time_.apply(lambda x: pd.Series(str(x).split(":")))

data4[0]=data4[0].astype(int)

df['hh'] = data4[0]
df['mm'] = data4[1]
df['ss'] = data4[2]

df['hh'] = df['hh']+df['Days']

df['Time2'] = df['hh'].astype(str) + ':' + df['mm'] + ':' + df['ss']

df = df.drop(['hh','mm','ss','Time_','Day...','Time'],axis=1)

#Creating 'HH:MM' feature which is basically converting existing time to float for further engineering
data6 = df.Time2.apply(lambda x: pd.Series(str(x).split(":")))
df['hh'] = data6[0]
df['mm'] = data6[1]
df['ss'] = data6[2]
df['mm'] = df['mm'].astype(float)
df['mm']= df['mm']*(1/60)
df['mm'] = df['mm'].round(2)
data8 = df['mm'].apply(lambda x: pd.Series(str(x).split(".")))
df['MM'] = data8[1]
df['HH:MM'] = df['hh'] + '.' + df['MM'].astype(str)
df = df.drop(['hh','mm','ss','MM','Geolocation'],axis=1)

#Saving data set to another dataset for further engineering and converting 'HH:MM' to float 

df1 =df
df1['HH:MM'] = df1['HH:MM'].astype(float)

df1.head(50)

# Differentiating cycles using 'HH:MM' to get best results of differentiations 
r_time=[]
a_time = df1.index[0]
x_time = df1.index[0]
for n_time in df1.index:
    if df1['HH:MM'][n_time] >= (df1['HH:MM'][a_time] + 1):
        dataset = df1.loc[x_time:a_time]
        arr_time = dataset.to_numpy()
        r_time.append(arr_time)
        x_time = n_time
    a_time=n_time

#Converting the numpy array dataframe to pandas
for t_time in range(0,len(r_time)):
    r_time[t_time] = pd.DataFrame(r_time[t_time])

#Adding the column names to the dataset
for x in range (0,len(r_time)):
    r_time[x].columns =['Voltage(V)', 'Soc(%)', 'Current Load(A)', 'Charging Current(A)','Battery Temperature','Capacity','Cycles Used','Speed(Kmph)','Total Distance(Km)','Trip Distance(Km)','Power(Watts)','Current Capacity','Days','Time2','HH:MM']

# Seperating the charging the discharging from cycles
charging_r_time = []
discharging_r_time = []
for r in range(0,len(r_time)):
    dataset_charging = r_time[r][r_time[r]['Current Load(A)'] > 0]
    arr_charging = dataset_charging.to_numpy()
    charging_r_time.append(arr_charging)
    dataset_discharging = r_time[r][r_time[r]['Current Load(A)'] < 0]
    arr_discharging = dataset_discharging.to_numpy()
    discharging_r_time.append(arr_discharging)

#Converting all the array datasets to pandas dataframe
for t in range(0,len(charging_r_time)):
    charging_r_time[t] = pd.DataFrame(charging_r_time[t])
    
for t in range(0,len(discharging_r_time)):
    discharging_r_time[t] = pd.DataFrame(discharging_r_time[t])

#Adding the column names to the dataset 
for x in range (0,len(r_time)):
    charging_r_time[x].columns =['Voltage(V)', 'Soc(%)', 'Current Load(A)', 'Charging Current(A)','Battery Temperature','Capacity','Cycles Used','Speed(Kmph)','Total Distance(Km)','Trip Distance(Km)','Power(Watts)','Current Capacity','Days','Time2','HH:MM']
    discharging_r_time[x].columns =['Voltage(V)', 'Soc(%)', 'Current Load(A)', 'Charging Current(A)','Battery Temperature','Capacity','Cycles Used','Speed(Kmph)','Total Distance(Km)','Trip Distance(Km)','Power(Watts)','Current Capacity','Days','Time2','HH:MM']
    
charging_test_set = df1[df1['Current Load(A)'] > 0]

for ind in charging_test_set.columns:
    if ind != 'Time2':
        charging_test_set[ind] = charging_test_set[ind].astype(float)


# In[32]:


#Storing charging_set_1 in another variable so that it does not effect our original charging_set_1
data_analysis = charging_test_set
data_analysis = charging_test_set.reset_index()
data_analysis = data_analysis.drop('index',axis=1)
data_analysis


# In[33]:


#We are creating histogram to actually know about the distribution of our data points
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

for ind in data_analysis.columns:
    plt.hist(data_analysis[ind])
    plt.title(ind)
    plt.show()


# In[34]:


#We are plotting scatter points to actually know about the effect of change of different variables with Current Capacity, most relevant one is Voltage(V) vs Current Capacity
for ind in data_analysis.columns:
    plt.scatter(x = data_analysis[ind], y = data_analysis['Current Capacity'])
    plt.xlabel(ind)
    plt.ylabel('Current Capacity')
    plt.title(ind)
    plt.show()


# In[35]:


#Creating Heat Map to find which all functions are actually correlated with each other
import seaborn as sns
uuu = data_analysis.drop(['Charging Current(A)','Capacity','Cycles Used','Days','HH:MM'], axis = 1)
corrmat = uuu.corr()
top_corr_features = corrmat.index
plt.figure(figsize = (10,6))
g = sns.heatmap(uuu[top_corr_features].corr(), annot = True)


# # MERGING ALL TRAIN DATASETS

# In[36]:


train_data = charging_set_1.append(charging_set_2)
train_data = train_data.append(charging_set_3)
train_data = train_data.append(charging_set_4)
train_data = train_data.append(charging_set_5)
train_data = train_data.append(charging_set_6)

train_data = train_data.reset_index()

train_data = train_data.drop('index',axis=1)


# # Preparing the Training data 

# In[37]:


train_data.corr()


# In[38]:


#Creating Heat Map to find which all functions are actually correlated with each other
import seaborn as sns
corrmat = data.corr()
top_corr_features = corrmat.index
plt.figure(figsize = (10,6))
g = sns.heatmap(data[top_corr_features].corr(), annot = True)


# In[39]:


#This is the kind of Feature Selection on the basis of heat map and correlations
#Since it is the time to get the final dataset for our model we need to get rid of unwanted variables to avoid curse of dimensionality
train_data = train_data.drop(['Charging Current(A)','Capacity','Cycles Used','Days','Time2','HH:MM','Total Distance(Km)','Soc(%)'],axis=1)


# In[40]:


train_data.head()


# # Preparing the TESTING DATA 

# In[41]:


test_data = charging_test_set
test_data = test_data.reset_index()
test_data = test_data.drop('index',axis=1)


# In[42]:


test_data = test_data[['Voltage(V)','Current Load(A)','Battery Temperature','Speed(Kmph)','Trip Distance(Km)','Power(Watts)','Current Capacity']]


# In[43]:


test_data.head()


# # Preparing the X_train, X_test, y_train, y_test set for our MACHINE LEARNING MODEL

# In[44]:


X_train = train_data.drop('Current Capacity',axis=1)
y_train = train_data['Current Capacity']

X_test = test_data.drop('Current Capacity',axis=1)
y_test = test_data['Current Capacity']


# # Machine Learning Model Using DecisionTreeRegressor

# In[46]:


#Creating the Model and training it with our Training Dataset
from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()
model.fit(X_train, y_train)


# In[47]:


#Making Predictions with our model 
prediction = model.predict(X_test)


# In[48]:


#Checking the Accuracy of our Model
from sklearn.metrics import r2_score
score = r2_score(y_test, prediction)
score


# In[49]:


#Plotting the predicted value and the actual test value to visualize our predictions
plt.scatter(y_test, prediction)


# In[51]:


#Checking the residual basically the standard deviation of our prediction that how much our predicted value deviates from the actual value and we can see that most of the value has '0' deviation which shows the accuracy of the model
residuals = y_test - prediction
sns.displot(residuals, kind = 'kde')


# In[ ]:


#We can predict our value manually by giving manual values
predictions = model.predict([[]])


# In[ ]:




