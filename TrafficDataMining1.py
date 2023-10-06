#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv('trajectories-0805am-0820am.csv')


# In[3]:


df


# In[4]:


# calculate the max and min speed of the vehicle

vel_513 = df.loc[df['Vehicle_ID'] == 513, 'v_Vel']
max_vel_513 = vel_513.max()
min_vel_513 = vel_513.min()

print(max_vel_513)
print(min_vel_513)


# In[5]:


# unique leaders of the vehicle

precede = df.loc[df["Vehicle_ID"] == 513, "Preceeding"]
print(set(precede))


# In[6]:


# trajectory of a single vehicle

car513 = df[df["Vehicle_ID"] == 513]


# In[7]:


plt.rcParams['figure.dpi'] =500
plt.rcParams['figure.figsize'] = [12,3]


# In[8]:


# color code and mark the axis
plt.scatter(x=car513.Global_Time,y=car513.Local_Y,c=car513.v_Vel,sizes=[0.75])
plt.colorbar()
plt.title('Timespace diagram for Car 513')
plt.xlabel('Time (s)')
plt.ylabel('Distance (m)')


# In[9]:


# add the trajectories of preceding vehicles

def plot_trajectory(vehicle_df, vehicle_id, label, color=None):
    plt.plot(vehicle_df.Global_Time / 1000.0, vehicle_df.Local_Y, label=label, color=color)

def main():
    preceding_vehicle_ids = car513['Preceeding'].unique()

    # Create a figure and plot the trajectory of vehicle 513
    plt.figure(figsize=(12, 3))

    plot_trajectory(car513, 513, 'Vehicle 513', 'blue')

    # Plot the trajectories of all preceding vehicles
    for preceding_id in preceding_vehicle_ids:
        preceding_vehicle = df[df['Vehicle_ID'] == preceding_id]
        plot_trajectory(preceding_vehicle, preceding_id, f'Preceding Vehicle {preceding_id}')

    plt.xlabel('Time(s)')
    plt.ylabel('Distance(ft)')
    plt.title('Trajectories of Vehicle 513 and Preceding Vehicles')
    plt.legend()
    plt.show()

main()


# In[10]:


# Plot the space-gap of a single vehicle

def compute_space_gaps(vehicle, dataframe):
    time_intervals = []
    space_gaps = []

    for _, row in vehicle.iterrows():
        preceding_vehicle_id = row['Preceeding'] 
        if preceding_vehicle_id != 0:
            # Get the row for the preceding vehicle at the same timestamp
            preceding_vehicle_row = dataframe[
                (dataframe['Global_Time'] == row['Global_Time']) & 
                (dataframe['Vehicle_ID'] == preceding_vehicle_id)
            ]

            if not preceding_vehicle_row.empty:
                # Calculate the space gap and relative time
                space_gap = (preceding_vehicle_row.iloc[0]['Local_Y'] - 
                             row['Local_Y'] - 
                             preceding_vehicle_row.iloc[0]['v_Length'])
                relative_time = (row['Global_Time'] - vehicle['Global_Time'].min()) / 1000.0

                # Append results to lists
                time_intervals.append(relative_time)
                space_gaps.append(space_gap)

    return time_intervals, space_gaps

def plot_space_gaps(time_intervals, space_gaps):
    plt.figure(figsize=(12, 3))
    plt.scatter(x=time_intervals, y=space_gaps)
    plt.xlabel('Time (s)')
    plt.ylabel('Space-gap (ft)')
    plt.title('Space-gap of Vehicle 513 to the Vehicle Ahead')
    plt.show()

def main():
    time_intervals, space_gaps = compute_space_gaps(car513, df)
    plot_space_gaps(time_intervals, space_gaps)

main()


# In[11]:


# Plot the spacing of a single vehicle

def get_spacings_and_intervals(vehicle, dataframe):
    time_intervals = []
    spacings = []

    for _, row in vehicle.iterrows():
        preceding_vehicle_id = row['Preceeding'] 
        if preceding_vehicle_id != 0:
            # Fetch the row for the preceding vehicle at the same timestamp
            preceding_vehicle_row = dataframe[
                (dataframe['Global_Time'] == row['Global_Time']) & 
                (dataframe['Vehicle_ID'] == preceding_vehicle_id)
            ]

            if not preceding_vehicle_row.empty:
                # Calculate the space gap and relative time
                spacing = preceding_vehicle_row.iloc[0]['Local_Y'] - row['Local_Y']
                relative_time = (row['Global_Time'] - vehicle['Global_Time'].min()) / 1000.0

                # Append results to lists
                time_intervals.append(relative_time)
                spacings.append(spacing)

    return time_intervals, spacings

def plot_spacing_vs_time(time_intervals, spacings):
    plt.figure(figsize=(12, 3))
    plt.scatter(x=time_intervals, y=spacings)
    plt.xlabel('Time (s)')
    plt.ylabel('Spacing (ft)')
    plt.title('Spacing of Vehicle 513 to the Vehicle Ahead')
    plt.show()

def main():
    time_intervals, spacings = get_spacings_and_intervals(car513, df)
    plot_spacing_vs_time(time_intervals, spacings)

main()


# In[12]:


# compute the flow over an area using the trajectory data

# Parameters
delta_t = 60 * 1000  # 60 seconds in milliseconds
delta_x = 300  # 300 feet
center_y = 1000  # center at Local_Y=1000
low_y = center_y - delta_x/2
high_y = center_y + delta_x/2
area = (delta_t / 3600000) * (delta_x / 5280)  # in hours*miles
lanes_of_interest = [1, 2, 3, 4, 5, 6, 7, 8]
start_time = df['Global_Time'].min()
end_time = df['Global_Time'].max()

def compute_flow_for_box(box_df):
    total_distance = sum(box_df.groupby('Vehicle_ID', group_keys=True)['Local_Y'].apply(lambda y: y.max() - y.min()))

    total_distance_miles = total_distance / 5280
    return total_distance_miles / area

def filter_dataframe_by_time_and_lane(dataframe, t, lane_id):
    time_filtered_df = dataframe[(dataframe['Global_Time'] >= t) & (dataframe['Global_Time'] < t + delta_t)]
    lane_filtered_df = time_filtered_df[time_filtered_df['Lane_ID'] == lane_id]
    return lane_filtered_df[(lane_filtered_df['Local_Y'] >= low_y) & (lane_filtered_df['Local_Y'] <= high_y)]

def compute_flow_by_lane(lane_id):
    return [{'Time_Start': t, 'Flow': compute_flow_for_box(filter_dataframe_by_time_and_lane(df, t, lane_id))}
            for t in range(start_time, end_time, delta_t)]

def plot_flow_results(results_by_lane):
    plt.figure(figsize=(12, 3))
    for lane_id, flow_df in results_by_lane.items():
        plt.plot(flow_df['Time_Start'], flow_df['Flow'], label=f"Lane {lane_id}")

    plt.title("Flow over Time by Lane")
    plt.xlabel("Time (milliseconds since start)")
    plt.ylabel("Flow (vehicles/hour)")
    plt.legend()
    plt.grid(True)
    plt.show()


# Main code
flow_results_by_lane = {lane_id: pd.DataFrame(compute_flow_by_lane(lane_id)) for lane_id in lanes_of_interest}

for lane_id, flow_df in flow_results_by_lane.items():
    print(f"Flow for Lane {lane_id} (vehicles/hour):")
    print(flow_df)

plot_flow_results(flow_results_by_lane)


# In[13]:


# Compute the density of vehicles using the area-based definitions

def filter_data_by_time_and_lane(df, t, delta_t, lane_id):
    time_filtered_df = df[(df['Global_Time'] >= t) & (df['Global_Time'] < t + delta_t)]
    lane_filtered_df = time_filtered_df[time_filtered_df['Lane_ID'] == lane_id]
    return lane_filtered_df[(lane_filtered_df['Local_Y'] >= low_y) & (lane_filtered_df['Local_Y'] <= high_y)]

def compute_vehicle_time_in_box(box_df, vehicle_id):
    vehicle_data = box_df[box_df['Vehicle_ID'] == vehicle_id]
    time_entry = vehicle_data['Global_Time'].min()
    time_exit = vehicle_data['Global_Time'].max()
    return time_exit - time_entry

def compute_density_for_lane(lane_id, df, start_time, end_time, delta_t, area):
    density_results = []
    for t in range(start_time, end_time, delta_t):
        box_df = filter_data_by_time_and_lane(df, t, delta_t, lane_id)
        
        total_time = sum(compute_vehicle_time_in_box(box_df, vehicle_id) for vehicle_id in box_df['Vehicle_ID'].unique())
        
        # Convert total_time to minutes for the density calculation
        total_time_minutes = total_time / 3600000
        
        # Compute the density
        density = total_time_minutes / area  # vehicles per minute-mile
        density_results.append({'Time_Start': t, 'Density': density})
    
    return pd.DataFrame(density_results)

def plot_density_by_lane(density_results_by_lane):
    plt.figure(figsize=(12, 3))
    for lane_id, flow_df in density_results_by_lane.items():
        plt.plot(flow_df['Time_Start'], flow_df['Density'], label=f"Lane {lane_id}")

    plt.title("Density over Time by Lane")
    plt.xlabel("Time (milliseconds since start)")
    plt.ylabel("Density (vehicles/miles)")
    plt.legend()
    plt.grid(True)
    plt.show()

# Main execution
density_results_by_lane = {lane_id: compute_density_for_lane(lane_id, df, start_time, end_time, delta_t, area) for lane_id in lanes_of_interest}

for lane_id, density_df in density_results_by_lane.items():
    print(f"Density for Lane {lane_id} (vehicles/miles):")
    print(density_df)

plot_density_by_lane(density_results_by_lane)


# In[14]:


# compute the average speed of traffic

def compute_speed_for_lane(lane_id, flow_results, density_results):
    flow_df = flow_results[lane_id]
    density_df = density_results[lane_id]
    merged_df = pd.merge(flow_df, density_df, on='Time_Start')
    merged_df['Speed'] = merged_df['Flow'] / merged_df['Density']
    return merged_df[['Time_Start', 'Speed']]

def display_speed_results(speed_results):
    for lane_id, speed_df in speed_results.items():
        print(f"Average Speed for Lane {lane_id}:")
        print(speed_df)

def plot_speed_by_lane(speed_results):
    plt.figure(figsize=(12, 3))
    
    # Create a color map to get a range of colors for the lanes
    color_map = plt.get_cmap('viridis', len(lanes_of_interest))
    
    for idx, (lane_id, speed_df) in enumerate(speed_results.items()):
        plt.plot(speed_df['Time_Start'], speed_df['Speed'], label=f"Lane {lane_id}", 
                 linestyle='--', color=color_map(idx), alpha=0.8)  # changed line style and added transparency

    plt.title("Speed over Time by Lane")
    plt.xlabel("Time (milliseconds since start)")
    plt.ylabel("Speed (miles/hour)")
    plt.legend()
    plt.grid(True)
    plt.show()

# Main execution
speed_results_by_lane = {lane_id: compute_speed_for_lane(lane_id, flow_results_by_lane, density_results_by_lane) for lane_id in lanes_of_interest}

display_speed_results(speed_results_by_lane)
plot_speed_by_lane(speed_results_by_lane)


# The average speed of traffic is shown in the output.

# In[15]:


# plot each flow,density pair in a single fundamental diagram

plt.figure(figsize=(14, 6))  

# Create a color map to get a range of colors for the lanes
color_map = plt.get_cmap('viridis', len(lanes_of_interest))

# Plot flow vs density for each lane
for idx, lane_id in enumerate(lanes_of_interest):
    flow_df = flow_results_by_lane[lane_id]
    density_df = density_results_by_lane[lane_id]

    # Merge dataframes
    merged_df = pd.merge(flow_df, density_df, on='Time_Start')

    plt.scatter(merged_df.Density, merged_df.Flow, label=f"Lane {lane_id}", 
                color=color_map(idx), alpha=0.7, s=50)  # Adjusted color, transparency, and size

# Add title, labels and legend
plt.title("Fundamental Diagram")
plt.xlabel("Density (vehicles per hour)")
plt.ylabel("Flow (vehicles per mile)")
plt.grid(True)  # Added grid
plt.legend(loc='upper right')  # Added legend

# Display the plot
plt.tight_layout()  # Adjust layout for better spacing
plt.show()


# In[16]:


# Plot a time space diagram (one for each lane) containing all vehicles in the dataset

# Set consistent y-limits for all lanes
y_min = df['Local_Y'].min() - 50
y_max = df['Local_Y'].max() + 50

# Create a color map to get a range of colors for the lanes
color_map = plt.get_cmap('viridis', 9)  # 9 colors for 9 lanes

for lane in range(1, 9):
    # Filter the dataset for the current lane
    lane_df = df[df['Lane_ID'] == lane]

    # Create a time-space diagram for the current lane
    plt.figure(figsize=(14, 5))  # Adjusted figure size
    plt.scatter(lane_df.Global_Time / 10000, lane_df.Local_Y, s=2, alpha=0.6, color=color_map(lane))  # Adjusted size, transparency, and color
    plt.title(f'Time-Space Diagram for Lane {lane}')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Distance (feet)')
    plt.ylim(y_min, y_max)  # Consistent y-axis
    plt.grid(True)  # Added grid
    plt.tight_layout()  # Adjust layout for better spacing
    plt.show()


# In[17]:


# calculate flows and densities every 300 ft for the full distance of the roadway
# Plot all of the flow-density pairs in a single fundamental diagram

import matplotlib.pyplot as plt

def calc_segment_flow_density(segment_df, area):
    """Calculate flow and density for a given segment."""
    # Calculate Flow
    total_distance = segment_df.groupby('Vehicle_ID', group_keys=True)['Local_Y'].apply(lambda y: y.max() - y.min()).sum()
    total_distance_miles = total_distance / 5280
    flow = total_distance_miles / area
    
    # Calculate Density
    total_time = segment_df.groupby('Vehicle_ID', group_keys=True)['Global_Time'].apply(lambda times: times.max() - times.min()).sum()
    total_time_hours = total_time / 3600000
    density = total_time_hours / area
    
    return flow, density

def generate_segment_filters(t, delta_t, lower_bound, upper_bound):
    """Generate filters for DataFrame based on time and space."""
    time_filter = (df['Global_Time'] >= t) & (df['Global_Time'] < t + delta_t)
    space_filter = (df['Local_Y'] >= lower_bound) & (df['Local_Y'] <= upper_bound)
    return time_filter & space_filter

def plot_fundamental_diagram(all_densities, all_flows):
    """Plot the flow-density relationship."""
    plt.figure(figsize=(12, 8))
    plt.scatter(all_densities, all_flows, s=40, c=all_densities, cmap='viridis', edgecolor='k', alpha=0.7)
    plt.colorbar(label='Density (vehicles per hour*mile)')
    plt.title("Fundamental Diagram")
    plt.xlabel("Density (vehicles per hour*mile)")
    plt.ylabel("Flow (vehicles per hour*mile)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# Set parameters
min_y, max_y = df['Local_Y'].min(), df['Local_Y'].max()
delta_t = 60 * 1000
delta_x = 300
area = (delta_t / 3600000) * (delta_x / 5280)
start_time = df['Global_Time'].min()
end_time = df['Global_Time'].max()

all_flows = []
all_densities = []

# Loop over space and time to calculate flow and density
for lower_bound in range(int(min_y), int(max_y - delta_x), delta_x):
    upper_bound = lower_bound + delta_x
    
    for t in range(start_time, end_time, delta_t):
        segment_df = df[generate_segment_filters(t, delta_t, lower_bound, upper_bound)]
        flow, density = calc_segment_flow_density(segment_df, area)
        
        all_flows.append(flow)
        all_densities.append(density)

# Plot results
plot_fundamental_diagram(all_densities, all_flows)


