#!/usr/bin/env python
# coding: utf-8

# IMPORTING LIBRARIES

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# IMPORTING DATA

# In[ ]:


import os

directory = r'C:\Users\kandp\OneDrive\Desktop\PROJECT\CYCLIST\FINAL DATA'
dfs = []
files = os.listdir(directory)
print("Files in directory:", files)

for filename in files:
    if filename.endswith(".xlsx") or filename.endswith(".xls"):
        file_path = os.path.join(directory, filename)
        print(f"Reading file: {file_path}")
        df = pd.read_excel(file_path)
        dfs.append(df)

if not dfs:
    print("No Excel files were found in the directory.")
else:
    combined_df = pd.concat(dfs, ignore_index=True)
    print(combined_df.head())


# DESCRIPTIVE STATISTICS

# In[ ]:


combined_df_encoded = pd.get_dummies(combined_df, columns=['rideable_type', 'member_casual'])
print(combined_df_encoded.head())


# In[ ]:


combined_df['started_at'] = pd.to_datetime(combined_df['started_at'])
combined_df['ended_at'] = pd.to_datetime(combined_df['ended_at'])
combined_df['ride_duration'] = (combined_df['ended_at'] - combined_df['started_at']).dt.total_seconds()
print(combined_df.head())
#duration is in seconds


# In[ ]:


combined_df['started_at'] = pd.to_datetime(combined_df['started_at'])
combined_df['ended_at'] = pd.to_datetime(combined_df['ended_at'])
combined_df['ride_duration'] = (combined_df['ended_at'] - combined_df['started_at']).dt.total_seconds()
combined_df = combined_df[combined_df['ride_duration'] > 0]
print(combined_df.head())


# In[ ]:





# In[ ]:


# Group by member type and calculate descriptive statistics for ride duration
ride_duration_by_member_type = combined_df.groupby('member_casual')['ride_duration'].describe()
print(ride_duration_by_member_type)


# In[ ]:


total_classic_bike = combined_df_encoded['rideable_type_classic_bike'].sum()
total_electric_bike = combined_df_encoded['rideable_type_electric_bike'].sum()
total_docked_bike = combined_df_encoded['rideable_type_docked_bike'].sum() 
total_casual = combined_df_encoded['member_casual_casual'].sum()
total_member = combined_df_encoded['member_casual_member'].sum()
print(f"Total classic bikes: {total_classic_bike}")
print(f"Total electric bikes: {total_electric_bike}")
print(f"Total docked bikes: {total_docked_bike}") 
print(f"Total casual users: {total_casual}")
print(f"Total members users: {total_member}")


# In[ ]:


busiest_start_station = combined_df['start_station_name'].value_counts().idxmax()
busiest_start_station_count = combined_df['start_station_name'].value_counts().max()
busiest_end_station = combined_df['end_station_name'].value_counts().idxmax()
busiest_end_station_count = combined_df['end_station_name'].value_counts().max()
top_five_start_stations = combined_df['start_station_name'].value_counts().head(5)
top_five_end_stations = combined_df['end_station_name'].value_counts().head(5)
print(f"Busiest start station: {busiest_start_station} with {busiest_start_station_count} rides")
print(f"Busiest end station: {busiest_end_station} with {busiest_end_station_count} rides")
print("\nTop five start stations:")
print(top_five_start_stations)
print("\nTop five end stations:")
print(top_five_end_stations)


# In[ ]:


combined_df.loc[:, 'started_at'] = pd.to_datetime(combined_df['started_at'])
combined_df.loc[:, 'month'] = combined_df['started_at'].dt.month
rides_per_month = combined_df.groupby('month').size().sort_values(ascending=False)
print("Rides per month (in descending order):")
print(rides_per_month)


# In[ ]:


combined_df['started_at'] = pd.to_datetime(combined_df['started_at'])
combined_df['day_of_week'] = combined_df['started_at'].dt.day_name()
rides_per_day = combined_df['day_of_week'].value_counts().sort_index()
print("Rides for each day of the week:")
print(rides_per_day)


# In[ ]:





# In[ ]:





# In[ ]:


ride_type_durations = combined_df.groupby('rideable_type')['ride_duration'].sum().reset_index()
ride_type_durations.rename(columns={'ride_duration': 'total_ride_duration'}, inplace=True)
print(ride_type_durations)


# In[ ]:


member_casual_durations = combined_df.groupby('member_casual')['ride_duration'].sum().reset_index()
member_casual_durations.rename(columns={'ride_duration': 'total_ride_duration'}, inplace=True)
print(member_casual_durations)


# In[ ]:


average_ride_length = combined_df['ride_duration'].mean()
print(f"Average ride length: {average_ride_length} seconds")


# VISUALISATION

# In[ ]:


correlation_matrix = combined_df_encoded.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', cbar=True, square=True)
plt.title('Correlation Heat Map')
plt.show()


# In[ ]:


rideable_type_counts = combined_df['rideable_type'].value_counts()
member_casual_counts = combined_df['member_casual'].value_counts()
plt.figure(figsize=(14, 7))

plt.subplot(1, 2, 1)
rideable_type_counts.plot.pie(autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff','#99ff99','#ffcc99'])
plt.title('Rideable Type Distribution')
plt.ylabel('') 

plt.subplot(1, 2, 2)
member_casual_counts.plot.pie(autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff'])
plt.title('Member Casual Distribution')
plt.ylabel('') 

plt.tight_layout()
plt.savefig('Rideable Type Distribution and Member Casual Distribution.png')
plt.show()


# In[ ]:


top_20_stations = combined_df['start_station_name'].value_counts().head(20).index
df_top_20 = combined_df[combined_df['start_station_name'].isin(top_20_stations)]
rideable_type_distribution = df_top_20.groupby(['start_station_name', 'rideable_type']).size().unstack(fill_value=0)
member_casual_distribution = df_top_20.groupby(['start_station_name', 'member_casual']).size().unstack(fill_value=0)

# Plotting the rideable_type distribution
plt.figure(figsize=(14, 7))
rideable_type_distribution.plot(kind='bar', stacked=True, ax=plt.gca(), colormap='viridis')
plt.title('Distribution of Rideable Types in Top 20 Stations')
plt.xlabel('Start Station Name')
plt.ylabel('Number of Rides')
plt.xticks(rotation=90)
plt.legend(title='Rideable Type')
plt.tight_layout()
plt.savefig('Distribution of Rideable Types in Top 20 Stations.png')
plt.show()

# Plotting the member_casual distribution
plt.figure(figsize=(14, 7))
member_casual_distribution.plot(kind='bar', stacked=True, ax=plt.gca(), colormap='viridis')
plt.title('Distribution of Member and Casual Rides in Top 20 Stations')
plt.xlabel('Start Station Name')
plt.ylabel('Number of Rides')
plt.xticks(rotation=90)
plt.legend(title='Member/Casual')
plt.tight_layout()
plt.savefig('Distribution of Member and Casual Rides in Top 20 Stations.png')
plt.show()


# In[ ]:


monthly_rides = combined_df.groupby('month').size().reset_index(name='ride_count')
sns.set(style='whitegrid')
plt.figure(figsize=(12, 6))
sns.barplot(data=monthly_rides, x='month', y='ride_count', palette='viridis')
plt.title('Monthly Number of Rides')
plt.xlabel('Month')
plt.ylabel('Number of Rides')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('Monthly Number of Rides')
plt.show()


# In[ ]:


combined_df['started_at'] = pd.to_datetime(combined_df['started_at'])
combined_df['day_of_week'] = combined_df['started_at'].dt.day_name()
day_of_week_counts = combined_df['day_of_week'].value_counts().sort_index()
plt.figure(figsize=(10, 6))
day_of_week_counts.plot(kind='bar', color='skyblue')
plt.title('Number of Rides for Each Day of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Number of Rides')
plt.xticks(rotation=45)
plt.tight_layout()
for i, v in enumerate(day_of_week_counts):
    plt.text(i, v + 10, str(v), ha='center', va='bottom')
plt.savefig('Number of Rides for Each Day of the Week')
plt.show()


# In[ ]:


rideable_type_durations = combined_df.groupby('rideable_type')['ride_duration'].sum()
member_casual_durations = combined_df.groupby('member_casual')['ride_duration'].sum()
plt.figure(figsize=(14, 7))
plt.subplot(1, 2, 1)
rideable_type_durations.plot.pie(autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff','#99ff99','#ffcc99'])
plt.title('Total Ride Duration by Rideable Type')
plt.ylabel('')
plt.subplot(1, 2, 2)
member_casual_durations.plot.pie(autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff'])
plt.title('Total Ride Duration by Member/Casual')
plt.ylabel('')
plt.tight_layout()
plt.savefig('Total Ride Duration by Rideable Type and Total Ride Duration by Member_Casual.png')
plt.show()


# In[ ]:


monthly_distribution = combined_df.groupby(['month', 'rideable_type']).size().reset_index(name='ride_count')
monthly_distribution['month'] = monthly_distribution['month'].dt.to_timestamp()
sns.set(style='whitegrid')
plt.figure(figsize=(12, 6))
sns.lineplot(data=monthly_distribution, x='month', y='ride_count', hue='rideable_type', marker='o')
plt.title('Monthly Distribution of Rideable Types')
plt.xlabel('Month')
plt.ylabel('Number of Rides')
plt.legend(title='Rideable Type')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('Monthly Distribution of Rideable Types')
plt.show()


# In[ ]:


monthly_distribution = combined_df.groupby(['month', 'member_casual']).size().reset_index(name='ride_count')
monthly_distribution['month'] = monthly_distribution['month'].dt.to_timestamp()
sns.set(style='whitegrid')
plt.figure(figsize=(12, 6))
sns.lineplot(data=monthly_distribution, x='month', y='ride_count', hue='member_casual', marker='o')
plt.title('Monthly Distribution of Member vs Casual Rides')
plt.xlabel('Month')
plt.ylabel('Number of Rides')
plt.legend(title='User Type')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('Monthly Distribution of Member vs Casual Rides')
plt.show()


# In[ ]:


casual_users_df = combined_df[combined_df['member_casual'] == 'casual']
top_start_stations = casual_users_df.groupby('start_station_name').size().reset_index(name='ride_count')
top_start_stations = top_start_stations.sort_values(by='ride_count', ascending=False).head(10)
sns.set(style='whitegrid')
plt.figure(figsize=(12, 6))
sns.barplot(data=top_start_stations, x='ride_count', y='start_station_name', palette='viridis')
plt.title('Top 10 Starting Stations for Casual Users')
plt.xlabel('Number of Rides')
plt.ylabel('Starting Station')
plt.tight_layout()
plt.savefig('Top 10 Starting Stations for Casual Users')
plt.show()


# In[ ]:


member_users_df = combined_df[combined_df['member_casual'] == 'member']
top_start_stations = member_users_df.groupby('start_station_name').size().reset_index(name='ride_count')
top_start_stations = top_start_stations.sort_values(by='ride_count', ascending=False).head(10)
sns.set(style='whitegrid')
plt.figure(figsize=(12, 6))
sns.barplot(data=top_start_stations, x='ride_count', y='start_station_name', palette='viridis')
plt.title('Top 10 Starting Stations for Member Users')
plt.xlabel('Number of Rides')
plt.ylabel('Starting Station')
plt.tight_layout()
plt.savefig('Top 10 Starting Stations for Member Users')
plt.show()


# In[ ]:


electric_bike_df = combined_df[combined_df['rideable_type'] == 'electric_bike']
start_station_counts = electric_bike_df['start_station_name'].value_counts()
top_10_stations = start_station_counts.head(10)
plt.figure(figsize=(14, 7))
top_10_stations.plot(kind='bar', color='skyblue')
plt.title('Top 10 Starting Stations for Electric Bikes')
plt.xlabel('Start Station Name')
plt.ylabel('Number of Rides')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('Top 10 Starting Stations for Electric Bikes')
plt.show()


# In[ ]:


classic_bike_df = combined_df[combined_df['rideable_type'] == 'classic_bike']
start_station_counts = classic_bike_df['start_station_name'].value_counts()
top_10_stations = start_station_counts.head(10)
plt.figure(figsize=(14, 7))
top_10_stations.plot(kind='bar', color='skyblue')
plt.title('Top 10 Starting Stations for Classic Bikes')
plt.xlabel('Start Station Name')
plt.ylabel('Number of Rides')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('Top 10 Starting Stations for Classic Bikes')
plt.show()


# In[ ]:


docked_bike_df = combined_df[combined_df['rideable_type'] == 'docked_bike']
start_station_counts = docked_bike_df['start_station_name'].value_counts()
top_10_stations = start_station_counts.head(10)
plt.figure(figsize=(14, 7))
top_10_stations.plot(kind='bar', color='skyblue')
plt.title('Top 10 Starting Stations for Docked Bikes')
plt.xlabel('Start Station Name')
plt.ylabel('Number of Rides')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('Top 10 Starting Stations for Docked Bikes')
plt.show()


# In[ ]:


#Peak hour analysis
combined_df['start_hour'] = combined_df['started_at'].dt.hour
rides_by_hour = combined_df.groupby('start_hour').size().reset_index(name='ride_count')
plt.figure(figsize=(12, 6))
sns.lineplot(data=rides_by_hour, x='start_hour', y='ride_count', marker='o')
plt.title('Distribution of Rides by Hour')
plt.xlabel('Hour of the Day')
plt.ylabel('Number of Rides')
plt.xticks(range(0, 24))
plt.tight_layout()
plt.savefig('Distribution of Rides by Hour')
plt.show()


# In[ ]:


combined_df['hour'] = combined_df['started_at'].dt.hour
hourly_usage = combined_df.groupby(['member_casual', 'hour']).size().unstack(fill_value=0)
plt.figure(figsize=(14, 7))
for member_casual in hourly_usage.index:
    plt.plot(hourly_usage.columns, hourly_usage.loc[member_casual], label=member_casual)
plt.title('Hourly Usage of Bikes According to Member/Casual')
plt.xlabel('Hour of the Day')
plt.ylabel('Number of Rides')
plt.legend(title='User Type')
plt.xticks(range(24))
plt.grid(True)
plt.tight_layout()
plt.savefig('Hourly Usage of Bikes According to Member_Casual')
plt.show()


# In[ ]:


hourly_usage_1 = combined_df.groupby(['rideable_type', 'hour']).size().unstack(fill_value=0)
plt.figure(figsize=(14, 7))
for rideable_type in hourly_usage_1.index:
    plt.plot(hourly_usage_1.columns, hourly_usage_1.loc[rideable_type], label=rideable_type)
plt.title('Hourly Usage of Bikes According to Rideable Type')
plt.xlabel('Hour of the Day')
plt.ylabel('Number of Rides')
plt.legend(title='Rideable Type')
plt.xticks(range(24))
plt.grid(True)
plt.tight_layout()
plt.savefig('Hourly Usage of Bikes According to Rideable Type')
plt.show()


# In[ ]:


ride_counts = combined_df.groupby(['rideable_type', 'member_casual']).size().unstack(fill_value=0)
ax = ride_counts.plot(kind='bar', figsize=(14, 7), color=['skyblue', 'salmon'])
plt.title('Distribution of Casual and Member Users According to Rideable Type')
plt.xlabel('Rideable Type')
plt.ylabel('Number of Rides')
plt.xticks(rotation=0)
plt.legend(title='User Type')
plt.tight_layout()
for container in ax.containers:
    ax.bar_label(container)
plt.savefig('Distribution of Casual and Member Users According to Rideable Type')
plt.show()


# In[ ]:


import folium
from folium.plugins import HeatMap

# Ensure your DataFrame has columns for latitude and longitude
# In your case, it's `start_lat` and `start_lng`

# Create a base map centered around an average latitude and longitude
avg_lat = combined_df['start_lat'].mean()
avg_lng = combined_df['start_lng'].mean()
m = folium.Map(location=[avg_lat, avg_lng], zoom_start=12)

# Create a list of locations for the heatmap
heat_data = [[row['start_lat'], row['start_lng']] for index, row in combined_df.iterrows()]

# Add the heatmap to the map
HeatMap(heat_data).add_to(m)

# Save the map to an HTML file
m.save("heatmap.html")

# Display the map (in Jupyter Notebook or similar environment)
m


# In[ ]:


from scipy.stats import ttest_ind

# Separate ride durations by user type
ride_durations_casual = combined_df[combined_df['member_casual'] == 'casual']['ride_duration']
ride_durations_member = combined_df[combined_df['member_casual'] == 'member']['ride_duration']

# Perform t-test
t_stat, p_value = ttest_ind(ride_durations_casual, ride_durations_member, equal_var=False)
print(f"T-statistic: {t_stat}, P-value: {p_value}")

