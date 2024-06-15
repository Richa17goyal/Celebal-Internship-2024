import random

# Sample data generation
months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
temperature_data = [random.uniform(0, 40) for _ in range(12)]  # Generate random temperatures for 12 months
precipitation_data = [random.uniform(0, 200) for _ in range(12)]  # Generate random precipitation for 12 months

# Displaying the generated data
for i, month in enumerate(months):
    print(f"{month}: Temperature - {temperature_data[i]}°C, Precipitation - {precipitation_data[i]} mm")

# Analysis
average_temperature = sum(temperature_data) / len(temperature_data)
average_precipitation = sum(precipitation_data) / len(precipitation_data)

print("\nClimate Analysis:")
print(f"Average Temperature: {average_temperature}°C")
print(f"Average Precipitation: {average_precipitation} mm")
