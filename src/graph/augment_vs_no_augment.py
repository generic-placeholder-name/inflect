import matplotlib.pyplot as plt
import os

# Data: No synthesized data
normal_data = {
    200: 0.2078,
    300: 0.5045,
    400: 0.5925,
    500: 0.7698,
    600: 0.8429,
    700: 0.8270,
    800: 0.8782,
    900: 0.8857,
    1000: 0.9109
}

# Data: With synthesized data (updated)
augmented_data = {
    0.2: {
        200: 0.7651,
        300: 0.8648,
        400: 0.7836,
        500: 0.8790,
        600: 0.8795,
        700: 0.9019,
        800: 0.8949,
        900: 0.8655,
        1000: 0.9108
    },
    0.5: {
        200: 0.8609,
        300: 0.8722,
        400: 0.8987,
        500: 0.8969,
        600: 0.9008,
        700: 0.6269,
        800: 0.9077,
        900: 0.9150,
        1000: 0.9221
    },
    1.0: {
        200: 0.8465,
        300: 0.8927,
        400: 0.9073,
        500: 0.8973,
        600: 0.9020,
        700: 0.9012,
        800: 0.9217,
        900: 0.8661,
        1000: 0.8904
    }
}

# Create img directory if it doesn't exist
os.makedirs('img', exist_ok=True)

# Plotting
plt.figure(figsize=(10, 6))

# Plot normal training data
x_normal = list(normal_data.keys())
y_normal = list(normal_data.values())
plt.plot(x_normal, y_normal, label="No synthesized data", marker='o', linestyle='-', color='black')

# Plot augmented training data
colors = {0.2: 'blue', 0.5: 'green', 1.0: 'red'}
for synth_ratio, data in augmented_data.items():
    x_aug = list(data.keys())
    y_aug = list(data.values())
    plt.plot(x_aug, y_aug, label=f"+{synth_ratio} synthesized", marker='o', linestyle='--', color=colors[synth_ratio])

# Style the plot
plt.xlabel('Number of paradigms in training data')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Number of Paradigms (with/without Synthesized Data)')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the figure
filename = 'graph1.png'
plt.savefig(f'img/{filename}')
plt.close()

print(f"Plot saved to img/{filename}")
