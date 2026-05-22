import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 6)

# Load data
db_path = Path("database")
airlines = pd.read_csv(db_path / "airlines.csv")
airports = pd.read_csv(db_path / "airports.csv")
flights = pd.read_csv(db_path / "flights.csv")

print("=" * 80)
print("EXPLORATORY DATA ANALYSIS - FLIGHT DATA")
print("=" * 80)

# ==================== DATA OVERVIEW ====================
print("\n1. DATASET OVERVIEW")
print("-" * 80)
print(f"Airlines: {airlines.shape[0]} records, {airlines.shape[1]} columns")
print(f"Airports: {airports.shape[0]} records, {airports.shape[1]} columns")
print(f"Flights: {flights.shape[0]} records, {flights.shape[1]} columns")

# ==================== MISSING VALUES ====================
print("\n2. MISSING VALUES ANALYSIS")
print("-" * 80)

print("\nAirlines Missing Values:")
print(airlines.isnull().sum())

print("\nAirports Missing Values:")
print(airports.isnull().sum())

print("\nFlights Missing Values:")
flights_missing = flights.isnull().sum()
flights_missing = flights_missing[flights_missing > 0]
print(flights_missing)
print(f"\nTotal missing values in flights: {flights.isnull().sum().sum()}")
print(f"Percentage of missing data: {(flights.isnull().sum().sum() / (flights.shape[0] * flights.shape[1]) * 100):.2f}%")

# ==================== DESCRIPTIVE STATISTICS ====================
print("\n3. DESCRIPTIVE STATISTICS - FLIGHTS DATA")
print("-" * 80)
print("\nBasic Statistics:")
print(flights.describe())

print("\nData Types:")
print(flights.dtypes)

# ==================== AIRLINES ANALYSIS ====================
print("\n4. AIRLINES ANALYSIS")
print("-" * 80)
print(f"Total Airlines: {airlines.shape[0]}")
print("\nAirlines List:")
print(airlines.to_string(index=False))

# ==================== AIRPORTS ANALYSIS ====================
print("\n5. AIRPORTS ANALYSIS")
print("-" * 80)
print(f"Total Airports: {airports.shape[0]}")
print("\nAirport Summary Statistics:")
print(airports[['LATITUDE', 'LONGITUDE']].describe())

# ==================== FLIGHTS DETAILED ANALYSIS ====================
print("\n6. FLIGHTS DETAILED ANALYSIS")
print("-" * 80)

# Delay analysis
print("\nDELAY STATISTICS:")
print(f"  Average Departure Delay: {flights['DEPARTURE_DELAY'].mean():.2f} minutes")
print(f"  Median Departure Delay: {flights['DEPARTURE_DELAY'].median():.2f} minutes")
print(f"  Std Dev Departure Delay: {flights['DEPARTURE_DELAY'].std():.2f} minutes")
print(f"  Max Departure Delay: {flights['DEPARTURE_DELAY'].max():.2f} minutes")
print(f"  Min Departure Delay: {flights['DEPARTURE_DELAY'].min():.2f} minutes")

print(f"\n  Average Arrival Delay: {flights['ARRIVAL_DELAY'].mean():.2f} minutes")
print(f"  Median Arrival Delay: {flights['ARRIVAL_DELAY'].median():.2f} minutes")
print(f"  Std Dev Arrival Delay: {flights['ARRIVAL_DELAY'].std():.2f} minutes")

# Cancellation analysis
cancelled_flights = flights[flights['CANCELLED'] == 1]
print(f"\nCANCELLATION STATISTICS:")
print(f"  Total Flights: {len(flights)}")
print(f"  Cancelled Flights: {len(cancelled_flights)} ({len(cancelled_flights)/len(flights)*100:.2f}%)")
print(f"  Diversion Count: {flights['DIVERTED'].sum()}")

# Distance analysis
print(f"\nDISTANCE STATISTICS:")
print(f"  Average Distance: {flights['DISTANCE'].mean():.2f} miles")
print(f"  Median Distance: {flights['DISTANCE'].median():.2f} miles")
print(f"  Max Distance: {flights['DISTANCE'].max():.2f} miles")
print(f"  Min Distance: {flights['DISTANCE'].min():.2f} miles")

# Time analysis
print(f"\nFLIGHT TIME STATISTICS:")
print(f"  Average Elapsed Time: {flights['ELAPSED_TIME'].mean():.2f} minutes")
print(f"  Average Air Time: {flights['AIR_TIME'].mean():.2f} minutes")
print(f"  Average Taxi Out: {flights['TAXI_OUT'].mean():.2f} minutes")
print(f"  Average Taxi In: {flights['TAXI_IN'].mean():.2f} minutes")

# Day of week analysis
print(f"\nFLIGHTS BY DAY OF WEEK:")
dow_mapping = {1: "Monday", 2: "Tuesday", 3: "Wednesday", 4: "Thursday", 5: "Friday", 6: "Saturday", 7: "Sunday"}
dow_counts = flights['DAY_OF_WEEK'].value_counts().sort_index()
for day, count in dow_counts.items():
    print(f"  {dow_mapping.get(day, f'Day {day}')}: {count} flights")

# Airline analysis
print(f"\nTOP 10 AIRLINES BY FLIGHT COUNT:")
airline_counts = flights['AIRLINE'].value_counts().head(10)
for airline, count in airline_counts.items():
    airline_name = airlines[airlines['IATA_CODE'] == airline]['AIRLINE'].values[0]
    print(f"  {airline} ({airline_name}): {count} flights")

# ==================== VISUALIZATIONS ====================
print("\n7. GENERATING VISUALIZATIONS")
print("-" * 80)

# Create figure with multiple subplots
fig = plt.figure(figsize=(16, 12))

# 1. Departure Delay Distribution
ax1 = plt.subplot(3, 3, 1)
flights['DEPARTURE_DELAY'].hist(bins=50, edgecolor='black', ax=ax1)
ax1.set_title('Distribution of Departure Delays', fontsize=12, fontweight='bold')
ax1.set_xlabel('Delay (minutes)')
ax1.set_ylabel('Frequency')

# 2. Arrival Delay Distribution
ax2 = plt.subplot(3, 3, 2)
flights['ARRIVAL_DELAY'].hist(bins=50, edgecolor='black', ax=ax2)
ax2.set_title('Distribution of Arrival Delays', fontsize=12, fontweight='bold')
ax2.set_xlabel('Delay (minutes)')
ax2.set_ylabel('Frequency')

# 3. Distance vs Elapsed Time
ax3 = plt.subplot(3, 3, 3)
ax3.scatter(flights['DISTANCE'], flights['ELAPSED_TIME'], alpha=0.3, s=10)
ax3.set_title('Distance vs Elapsed Time', fontsize=12, fontweight='bold')
ax3.set_xlabel('Distance (miles)')
ax3.set_ylabel('Elapsed Time (minutes)')

# 4. Flights by Day of Week
ax4 = plt.subplot(3, 3, 4)
dow_data = flights['DAY_OF_WEEK'].value_counts().sort_index()
dow_labels = [dow_mapping.get(i, f'Day {i}') for i in dow_data.index]
ax4.bar(dow_labels, dow_data.values, color='skyblue', edgecolor='black')
ax4.set_title('Flights by Day of Week', fontsize=12, fontweight='bold')
ax4.set_ylabel('Number of Flights')
ax4.tick_params(axis='x', rotation=45)

# 5. Top 10 Airlines
ax5 = plt.subplot(3, 3, 5)
top_airlines = flights['AIRLINE'].value_counts().head(10)
ax5.barh(top_airlines.index, top_airlines.values, color='lightcoral', edgecolor='black')
ax5.set_title('Top 10 Airlines by Flight Count', fontsize=12, fontweight='bold')
ax5.set_xlabel('Number of Flights')
ax5.invert_yaxis()

# 6. Cancellation Rate
ax6 = plt.subplot(3, 3, 6)
cancellation_data = [len(flights[flights['CANCELLED'] == 0]), len(flights[flights['CANCELLED'] == 1])]
ax6.pie(cancellation_data, labels=['Completed', 'Cancelled'], autopct='%1.2f%%',
        colors=['lightgreen', 'lightcoral'], startangle=90)
ax6.set_title('Flight Cancellation Rate', fontsize=12, fontweight='bold')

# 7. Average Delay by Day of Week
ax7 = plt.subplot(3, 3, 7)
delay_by_dow = flights.groupby('DAY_OF_WEEK')['DEPARTURE_DELAY'].mean()
dow_labels = [dow_mapping.get(i, f'Day {i}') for i in delay_by_dow.index]
ax7.plot(dow_labels, delay_by_dow.values, marker='o', linewidth=2, markersize=8, color='darkblue')
ax7.set_title('Average Departure Delay by Day of Week', fontsize=12, fontweight='bold')
ax7.set_ylabel('Delay (minutes)')
ax7.tick_params(axis='x', rotation=45)
ax7.grid(True, alpha=0.3)

# 8. Distance Distribution
ax8 = plt.subplot(3, 3, 8)
flights['DISTANCE'].hist(bins=50, edgecolor='black', color='orange', ax=ax8)
ax8.set_title('Distribution of Flight Distances', fontsize=12, fontweight='bold')
ax8.set_xlabel('Distance (miles)')
ax8.set_ylabel('Frequency')

# 9. Delay Types Distribution (only for cancelled flights)
ax9 = plt.subplot(3, 3, 9)
delay_types = {
    'Air System': flights['AIR_SYSTEM_DELAY'].sum(),
    'Security': flights['SECURITY_DELAY'].sum(),
    'Airline': flights['AIRLINE_DELAY'].sum(),
    'Late Aircraft': flights['LATE_AIRCRAFT_DELAY'].sum(),
    'Weather': flights['WEATHER_DELAY'].sum()
}
delay_types = {k: v for k, v in delay_types.items() if v > 0}
ax9.bar(delay_types.keys(), delay_types.values(), color='mediumpurple', edgecolor='black')
ax9.set_title('Delay Types (by occurrence count)', fontsize=12, fontweight='bold')
ax9.set_ylabel('Count')
ax9.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('visualizations/flight_eda.png', dpi=300, bbox_inches='tight')
print("✓ Saved: visualizations/flight_eda.png")
plt.show()

# ==================== MISSING VALUE HANDLING STRATEGY ====================
print("\n8. MISSING VALUE HANDLING RECOMMENDATIONS")
print("-" * 80)
print("""
Based on the analysis, here are recommended strategies for handling missing values:

1. DELAY COLUMNS (AIR_SYSTEM_DELAY, SECURITY_DELAY, AIRLINE_DELAY, LATE_AIRCRAFT_DELAY, WEATHER_DELAY):
   - These are missing because they only apply to delayed/cancelled flights
   - Strategy: Fill with 0 (no delay of that type) for non-delayed flights
   - Action: Use fillna(0)

2. CANCELLATION_REASON:
   - Only populated for cancelled flights
   - Strategy: Fill with 'N/A' or leave as null (represents non-cancelled flights)
   - Action: Use fillna('N/A')

3. Other numeric columns (e.g., WHEELS_OFF, WHEELS_ON if any):
   - Strategy: Forward fill or interpolation depending on context
   - Action: Investigate actual distribution first
""")

# ==================== SUMMARY STATISTICS ====================
print("\n9. KEY INSIGHTS SUMMARY")
print("-" * 80)
print(f"""
• Dataset contains {len(flights):,} flight records from {flights['YEAR'].nunique()} year(s)
• {len(airports)} unique airports and {len(airlines)} airlines in the dataset
• Average flight delay: {flights['DEPARTURE_DELAY'].mean():.2f} minutes
• Cancellation rate: {len(cancelled_flights)/len(flights)*100:.2f}%
• Most common day of week: {dow_mapping.get(flights['DAY_OF_WEEK'].mode()[0], 'Unknown')}
• Average flight distance: {flights['DISTANCE'].mean():.2f} miles
• Top airline: {flights['AIRLINE'].value_counts().index[0]}
""")

print("\n" + "=" * 80)
print("EDA COMPLETE")
print("=" * 80)
