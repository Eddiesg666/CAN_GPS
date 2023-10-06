#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('pip', 'install strym')


# In[2]:


pip install strym matplotlib


# In[3]:


get_ipython().system('pip install gmaps')


# In[138]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
import os
import cantools
import warnings
warnings.filterwarnings("ignore")
import datetime
import strym
import strym as s
from strym import strymread
from strym import strymmap
from strym import meta



# In[5]:


# process Toyota CAN data first
toyota_can = pd.read_csv('2020-07-08-20-22-14_Toyota_CAN_Messages.csv')


# In[6]:


toyota_can.head(1)
toyota_can.tail(1)


# In[7]:


toyota_can


# ### Minitest 1: Toyota CAN and GPS data

# In[8]:


# the start and end datetimes of this file:

# The start datetime is Thu Jul 09 2020 03:22:14 GMT+0000

# The end datetime is Thu Jul 09 2020 03:37:53 GMT+0000

start_time = toyota_can.Time.min()
end_time = toyota_can.Time.max()
print(start_time)
print(end_time)


# In[9]:


# the duration of the file (in seconds)
print(round(end_time - start_time))


# In[10]:


# unique messages of this file?
print(toyota_can.Message.nunique())
print(toyota_can.MessageID.nunique())
print(np.array(sorted(toyota_can.MessageID.unique())))

# unique_messages = df.drop_duplicates(subset=['MessageID', 'Message'])[['MessageID', 'Message']]

# output = unique_messages.apply(lambda x: f"Message ID: {x['MessageID']}, Message: {x['Message']}", axis=1).tolist()
# print(output)


# In[11]:


# data size
print(toyota_can.shape[0])


# In[12]:


# process Toyota GPS data
toyota_gps = pd.read_csv('2020-07-08-20-22-14_Toyota_GPS_Messages.csv')


# In[13]:


toyota_gps.head(1)


# In[14]:


toyota_gps


# ### Minitest 1: Honda CAN and GPS data

# In[18]:


# process CAN data of the Honda vehicle
honda_can = pd.read_csv('2020-07-08-20-22-16_Honda_CAN_Messages.csv')


# In[19]:


honda_can.head(1)


# In[20]:


honda_can


# In[ ]:


# process GPS data of the Honda vehicle
honda_gps = pd.read_csv('2020-07-08-20-22-16_Honda_GPS_Messages.csv')


# In[ ]:


honda_gps.head(1)


# In[27]:


honda_gps


# ### Comparison on CAN and GPS data across vehicle types

# In[31]:


# common message types for each of the cars, based on MessageID
MessageID_toyota = sorted(toyota_can.MessageID.unique())
MessageID_honda = sorted(honda_can.MessageID.unique())
common_message = set(MessageID_toyota).intersection(MessageID_honda)
print(common_message)
print(len(common_message))


# Single car analysis: Minitest 1, Toyota CAN Data only

# In[32]:


toyota_can = pd.read_csv('2020-07-08-20-22-14_Toyota_CAN_Messages.csv')


# In[33]:


toyota_can


# In[34]:


# The vehicle first appears at Thu Jul 09 2020 03:22:14 GMT+0000
start_time_toyota_can = toyota_can.Time.min()
print(start_time_toyota_can)


# In[35]:


# The vehicle exits the roadway at Thu Jul 09 2020 03:37:53 GMT+0000
end_time_toyota_can = toyota_can.Time.max()
print(end_time_toyota_can)


# In[36]:


toyota = "2020-07-08-20-22-14_Toyota"
toyota_can_file = toyota + "_CAN_Messages.csv"

dbcfile = 'toyota_rav4_2019.dbc'
csvdata = toyota_can_file
r = strymread(csvfile=csvdata, dbcfile=dbcfile, createdb=True, verbose=False)


# In[37]:


r.dataframe


# In[38]:


# extract and plot the speed
t_speed = r.speed()
# plot relative timeseries since first data value received.
plt.plot(t_speed.Time-t_speed.iloc[0].Time,t_speed.Message)
plt.xlabel('Time')
plt.ylabel('Speed (km/hr)')
plt.title('Figure 1: Speed of Toyota Vehicle (km/hr)')


# In[39]:


# calculate the avg speed
avg_speed = t_speed.Message.sum()/t_speed.shape[0]
print(avg_speed)


# In[40]:


# estimate the total distance

total_time = 939 / 3600
total_dist = total_time * avg_speed
print(total_dist)


# In[41]:


# get the maximum and minimum speed of the vehicle

max_speed = t_speed.Message.max()
print(max_speed)
min_speed = t_speed.Message.min()
print(min_speed)


# In[43]:


# Extract and plot the space gap using the lead_distance method in strym

r =strymread(csvfile=csvdata, dbcfile = dbcfile, createdb = True, verbose=False)
lead_distance = r.lead_distance()
lead_distance.reset_index(inplace = True)
lead_distance.head(1)


# In[44]:


t_dist = lead_distance.iloc[0].Time
plt.scatter(lead_distance.Time-t_dist,lead_distance.Message, s=5)
plt.xlabel("Time (sec)")
plt.ylabel("Space Gap (m)")
plt.title("Figure 2: Space Gap Diagram")
plt.show()


# In[45]:


# Determine how many times the lead vehicle changes (either a cut-in or a cut-out) there are during this drive, 
# using information about the length of an average car, along with discontinuities in the dataframe

# Calculate the change in distance
lead_distance['distance_change'] = lead_distance['Message'].diff()

# Plot the data
plt.scatter(lead_distance['Time'], lead_distance['distance_change'])
plt.xlabel("Time")
plt.ylabel("Distance_change (m)")
plt.title("Figure 3: Lead Vehicle Relative Distance Change")
plt.show()


# In[46]:


# Calculate the absolute value of distance change
lead_distance['abs_distance_change'] = lead_distance['distance_change'].abs()

avg_car_length = 4.5

# Identify rows where the absolute distance change exceeds the threshold
vehicle_changes = lead_distance[lead_distance['abs_distance_change'] > avg_car_length]

# Count the number of times the lead vehicle changes
num_changes = len(vehicle_changes)

print(num_changes)


# In[47]:


# calculate the frequency of the speed and lead\_distance data

fq_speed = t_speed.shape[0] / (t_speed.Time.max() - t_speed.Time.min())
fq_lead_distance = lead_distance.shape[0] / (lead_distance.Time.max() - lead_distance.Time.min())
print(fq_speed)
print(fq_lead_distance)


# In[48]:


# Merge the speed and lead\_distance dataframes, using a zero-order-hold approach to acquire the speed and space gap.


# In[96]:


lead_distance_copy = r.lead_distance()


# In[106]:


lead_distance_copy.drop(columns=['MessageLength'], inplace=True)


# In[113]:


lead_distance_copy.reset_index(inplace=True)


# In[114]:


lead_distance_copy.head(1)


# In[115]:


lead_distance_copy.columns


# In[104]:


t_speed = r.speed()
t_speed.reset_index(inplace=True)
t_speed.drop(columns=['MessageLength'], inplace=True)
t_speed.rename(columns={'Message': 'Speed'}, inplace=True)

t_speed.head(1)


# In[105]:


t_speed.columns


# In[116]:


t_speed.columns = ['Clock','Time','Speed']
lead_distance_copy.columns = ['Clock', 'Time','LeadDistance']


# In[122]:


merge_df = pd.merge_asof(t_speed[['Clock','Time','Speed']],lead_distance_copy[['Clock','LeadDistance']],on='Clock',allow_exact_matches=False,direction='nearest')


# In[123]:


merge_df


# In[124]:


merge_df.shape,t_speed.shape,lead_distance_copy.shape


# In[126]:


# Calculate and plot the time gap using the unified dataframe

merge_df['speed_mps'] = merge_df.Speed * 1000 / 3600
merge_df['time_gap'] = merge_df.LeadDistance / merge_df.speed_mps
plt.plot(merge_df.Time, merge_df.time_gap)
plt.xlabel("Time (sec)")
plt.ylabel("Time Gap")
plt.title("Figure 4: Time Gap Over Time")
plt.show()


# In[131]:


# Integrate the speed signal to estimate the total distance

def calculate_odometer(t_speed):
    shifted_speed = t_speed.shift(1)
    t_speed['total_distance'] = (t_speed['Speed'] + shifted_speed['Speed']) * (t_speed['Time'] - shifted_speed['Time']) / 3600 / 2
    t_speed['total_distance_cum'] = t_speed['total_distance'].cumsum().fillna(0)
    return t_speed

def plot_odometer(df):
    t0 = df.iloc[0].Time
    plt.plot(df.Time - t0, df.total_distance_cum)
    plt.xlabel("Time")
    plt.ylabel("Cumulative Distance (km)")
    plt.title("Figure 5: Estimated Total Distance Traveled")
    plt.show()

# Execute the functions
speed_data = calculate_odometer(t_speed)
plot_odometer(speed_data)


# In[135]:


# Plot the trajectory of this vehicle in a time space diagram

def plot_time_space_diagram(df):
    x = df.iloc[0].Time
    plt.plot(df.Time - x, df.total_distance_cum)
    plt.xlabel("Time (sec)")
    plt.ylabel("Space (km)")
    plt.title("Figure 6: Time Space Diagram")
    plt.show()
    
speed_data = calculate_odometer(t_speed)
plot_time_space_diagram(speed_data)


# In[136]:


# plot vehicle speed

plt.plot(merge_df.Time, merge_df.speed_mps)
plt.xlabel("Time (sec)")
plt.ylabel("Speed [m/s]")
plt.title("Figure 7: Vehicle Speed Over Time")
plt.show()


# In[145]:


# cruise state validation

db = cantools.database.Database()
with open(dbcfile,'r') as path:    
    db = cantools.database.load(path)
drive = pd.read_csv(csvdata)

cruise_state = s.convertData('PCM_CRUISE','CRUISE_ACTIVE',drive,db)


# In[165]:


x = cruise_state.iloc[0].Time
plt.scatter(cruise_state.Time-x,cruise_state.Message,s=2)
plt.xlabel("Time (sec)")
plt.ylabel("Cruise Control State")
plt.title("Figure 8: Cruise Control State: (0=off; 1=on)")
plt.show()


# In[159]:


# Plot Lead vehicle relative distance and Adaptive Cruise Control State and compare with the decoded speed signal
# Lead vehicle relative distance v.s speed
merge_df['speed_mps'] = merge_df.Speed * 1000/3600 # convert the unit of velocity from km/h to m/s
x = merge_df.iloc[0].Time
plt.scatter(merge_df.Time-x,merge_df.speed_mps,c='m',s=4,label='Speed (m/s)')
plt.scatter(merge_df.Time-x,merge_df.LeadDistance,c='b',s=4,label='Leading Distance')
plt.legend(["Speed (m/s)", "Leading Distance"], fontsize="15", loc ="right")
plt.xlabel("Time (sec)")
plt.ylabel("Speed and Leading Distance")
plt.title("Figure 9: Speed and Leading Distance diagram")
plt.show()


# In[166]:


cruise_state.head(1)


# In[173]:


# Adaptive Cruise Control State v.s speed.

# Setting column names
cruise_state.columns = ['Time', 'CruiseState', 'Bus']

# Merging datasets
merged_df2 = pd.merge_asof(
    t_speed[['Clock', 'Time', 'Speed']],
    cruise_state[['Time', 'CruiseState']],
    on='Time',
    allow_exact_matches=False,
    direction='nearest'
)

# Convert speed to m/s
merged_df2['speed_mps'] = merged_df2.Speed * 1000 / 3600

# Time adjustment
x = merged_df2.iloc[0].Time
merged_df2.Time -= x

# Plotting
for state, color, label in [(1, 'k', 'Cruise Control On'), (0, 'w', 'Cruise Control Off')]:
    subset = merged_df2[merged_df2.CruiseState == state]
    plt.scatter(subset.Time, subset.speed_mps, c=color, s=4, label=label)

plt.legend()
plt.xlabel("Time (sec)")
plt.ylabel("Speed")
plt.title("Figure 10: Speed Under Different Cruise States Diagram")
plt.show()


# ### 3.3 Aggregate Data: All Toyota Data across many files

# In[213]:


folder = "Data/"

dataset = []
sub_path = os.listdir(folder)
for i in sub_path:
    file_list = os.listdir(folder + i + '/')
    for j in file_list:
        if "CAN" not in j or "Toyota" not in j:
            continue
        dataset.append(folder + i + '/' + j)


# In[228]:


# Steering angle absolute value and speed
dbcfile  = 'toyota_rav4_2019.dbc'
s_list = []

s_a_list = []
r_list = []

for csv in dataset:
    print(f"\nProcessing {csv}")
    r = strymread(csvfile=csv, dbcfile=dbcfile)

    # Filter out files recorded with Python based on burst flag
    if r.success and not r.burst:
        r_list.append(r)
        
        speed = r.speed()
        speed['Message'] /= 3.6  
        
        sa = r.steer_angle()
        sa['Message'] = abs(sa.Message)
    
        s_list.append(speed)
        s_a_list.append(steering_angle)


# In[242]:


speed.head(1)


# In[243]:


sa.head(1)


# In[ ]:


r_speeds = []

for s, angle in zip(s_list, s_a_list):
    if s.shape[0] == 0:
        continue
    s_new, angle_new = strymread.ts_sync(s, angle, rate="second")
    df = pd.DataFrame({
        'Time': s_new['Time'],
        'Speed': s_new['Message'],
        'Steering_Angle': angle_new['Message']
    })
    r_speeds.append(df)
steering_angle = pd.concat(r_speeds)


# In[ ]:


figure, val = strymread.create_fig(1)
figure.set_size_inches(10, 10)
val[0].scatter(x = 'Speed', y = 'Steering_Angle', data = steering_angle, s = 1, color = 'r')
val[0].set_xlabel('Speed (m/s)')
val[0].set_ylabel('Steering Angle (degrees)')
val[0].set_title('Figure 11: Speed and Steering Angle Relationship')
plt.show()


# In[252]:


# Acceleration vs speed
dbcfile  = 'toyota_rav4_2019.dbc'
s_list = []

a_list = []
r_list = []

for csv in dataset:
    print(f"\nProcessing {csv}")
    r = strymread(csvfile=csv, dbcfile=dbcfile)

    # Filter out files recorded with Python based on burst flag
    if r.success and not r.burst:
        r_list.append(r)
        
        speed = r.speed()
        speed['Message'] /= 3.6  
        accelx = r.accelx()
    
        s_list.append(speed)
        a_list.append(accelx)


# In[ ]:


r_speeds = []

for speed, acc in zip(s_list, a_list):
    if speed.shape[0] == 0:
        continue
    s_new, a_new = strymread.ts_sync(speed, acc, rate="second")
    df = pd.DataFrame({
        'Time': s_new['Time'],
        'Speed': s_new['Message'],
        'Acceleration': a_new['Message']
    })
    r_speeds.append(df)
s_acc = pd.concat(r_speeds)


# In[ ]:


figure, val = strymread.create_fig(1)
figure.set_size_inches(10, 10)
val[0].scatter(x = 'Speed', y = 'Acceleration', data = s_acc, s = 1, color = 'r')
val[0].set_xlabel('Speed (m/s)')
val[0].set_ylabel('Acceleration (m/s^2)')
val[0].set_title('Figure 12: Speed and Acceleration Relationship')
plt.show()


# In[ ]:


# Histogram of all positive acceleration

params = np.linspace(0, 4, 40)  

plt.hist(s_acc[s_acc.Acceleration > 0].Acceleration, bins=params)
plt.xlabel("Acceleration (m/s^2)")
plt.ylabel("Number count")
plt.title("Figure 13: Histogram: All Positive Acceleration")
plt.show()


# In[ ]:


# Histogram of all negative acceleration

params2 = np.linspace(-4, 0, 40)  

plt.hist(s_acc[s_acc.Acceleration < 0].Acceleration, bins=params2)
plt.xlabel("Acceleration (m/s^2)")
plt.ylabel("Number count")
plt.title("Figure 14: Histogram: All Negative Acceleration")
plt.show()

