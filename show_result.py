import numpy as np
import os
import matplotlib.pyplot as plt

# Path to the results directory
results_dir = 'results/single noise'

# List to store experiment names and their corresponding MSE values
experiments = []
mse_values = []

# Iterate over all files in the results directory
for file_name in os.listdir(results_dir):
    if file_name.endswith('.npy'):
        # Extract experiment name from the file name
        exp_name = os.path.splitext(file_name)[0]
        # Load MSE values from the numpy file
        mse = np.load(os.path.join(results_dir, file_name))
        # Append to the lists
        experiments.append(exp_name)
        mse_values.append(mse.sum())  # Assuming you want to sum the MSE values for the bar chart

# Create a bar chart
plt.figure(figsize=(12, 6))
bars = plt.bar(experiments, mse_values, color='skyblue')
plt.xlabel('Experiments')
plt.ylabel('MSE')
plt.title('MSE Results of Each Model')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Add exact value on top of each bar
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.4f}', ha='center', va='bottom')

# Save the plot
output_path = os.path.join(results_dir, 'mse_results.png')
plt.savefig(output_path)

# Show the plot
plt.show()
