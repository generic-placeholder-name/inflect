import matplotlib.pyplot as plt
import os

# Old augmented accuracies (before final phase)
old_augmented = {
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

# New accuracies after final phase (aligned: paradigms -> [0.2, 0.5, 1.0])
refresh = {
    200: [0.8531, 0.8946, 0.8413],
    300: [0.8464, 0.8941, 0.8908],
    400: [0.9047, 0.9082, 0.9090],
    500: [0.7227, 0.9110, 0.9074],
    600: [0.9089, 0.9057, 0.8661],
    700: [0.9069, 0.9094, 0.9248],
    800: [0.5737, 0.9213, 0.9132],
    900: [0.9188, 0.9074, 0.9210],
    1000: [0.9268, 0.9033, 0.9292]
}

# Create img directory if needed
os.makedirs('img', exist_ok=True)

# Plotting
plt.figure(figsize=(12, 7))

colors = {0.2: 'blue', 0.5: 'green', 1.0: 'red'}
linestyles = {'before': '--', 'after': '-'}

for idx, synth_ratio in enumerate([0.2, 0.5, 1.0]):
    # Extract old and new data
    x = list(old_augmented[synth_ratio].keys())
    y_old = list(old_augmented[synth_ratio].values())
    y_new = [refresh[n][idx] for n in x]

    # Plot with no refresh
    plt.plot(x, y_old, label=f"+{synth_ratio} synth (with no refresh)", 
             color=colors[synth_ratio], linestyle=linestyles['before'], marker='o')
    
    # Plot with refresh
    plt.plot(x, y_new, label=f"+{synth_ratio} synth (with refresh)", 
             color=colors[synth_ratio], linestyle=linestyles['after'], marker='s')

# Style
plt.xlabel('Number of paradigms in training data')
plt.ylabel('Accuracy')
plt.title('Effect of Refreshing Synthesized Data on Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save
plt.savefig('img/refresh_comparison.png')
plt.close()

print("Comparison plot saved to img/refresh_comparison.png")
