import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def load_results(file_path):
    """Load results from a CSV file."""
    results = pd.read_csv(file_path)
    results['iteracoes_tempo'] = results['iteracoes_tempo'].str.extract(r'(\d+\.\d+)').astype(float)
    results['iteracoes_acuracia'] = results['iteracoes_acuracia'].str.extract(r'(\d+\.\d+)').astype(float)
    return results

def calculate_statistics(results, environment):
    """Calculate mean and confidence interval for a given environment."""
    mean_time = results[results['ambiente'] == environment].groupby('algoritmo')['iteracoes_tempo'].mean()
    ci_time = results[results['ambiente'] == environment].groupby('algoritmo')['iteracoes_tempo'].std()
    mean_accuracy = results[results['ambiente'] == environment].groupby('algoritmo')['iteracoes_acuracia'].mean()
    ci_accuracy = results[results['ambiente'] == environment].groupby('algoritmo')['iteracoes_acuracia'].std()
    return mean_time, ci_time, mean_accuracy, ci_accuracy

def map_algorithm_names(algorithms):
    """Map algorithm names with line breaks."""
    return algorithms.map({
        'nb': 'Naive\nBayes',
        'dt': 'Decision\nTree',
        'lr': 'Logistic\nRegression',
        'rf': 'Random\nForest'
    })

def plot_bar_graph(data, ax, bar_width, palette):
    """Plot a bar graph."""
    bar1 = ax.bar(data.index - bar_width/2, data['Mean Time'], bar_width, yerr=data['CI Time'], capsize=5, label='Time', color=palette[0])
    bar2 = ax.bar([i + bar_width/2 for i in data.index], data['Mean Accuracy'], bar_width, yerr=data['CI Accuracy'], capsize=5, label='Accuracy', color=palette[2])
    return bar1, bar2

# Load results
results_file = '/content/drive/MyDrive/Mestrado/dataset/attack-ddos-layer7/resultado.csv'
results_data = load_results(results_file)

# Calculate statistics for ML and BD environments
ml_mean_time, ml_ci_time, ml_mean_accuracy, ml_ci_accuracy = calculate_statistics(results_data, 'ml')
bd_mean_time, bd_ci_time, bd_mean_accuracy, bd_ci_accuracy = calculate_statistics(results_data, 'bd')

# Map algorithm names for line breaks
ml_mean_time.index = map_algorithm_names(ml_mean_time.index)
bd_mean_time.index = map_algorithm_names(bd_mean_time.index)

# Create subplots
fig, (ax1_ml, ax1_bd) = plt.subplots(2, 1, figsize=(14, 14))

# Set palette
palette = sns.color_palette("tab10")
palette = [(r, g, b, 0.2) for r, g, b in palette]

# Plot bar graph for ML environment
plot_bar_graph(data_combined_ml, ax1_ml, 0.4, palette)
ax1_ml.set_xlabel('Algorithm', fontsize=18)
ax1_ml.set_ylabel('Time (s)', fontsize=18, color='darkblue')
ax1_ml.set_xticks(data_combined_ml.index)
ax1_ml.set_xticklabels(data_combined_ml['Algorithm'], rotation=0, ha='center', fontsize=14)
ax1_ml.tick_params(axis='y', labelcolor='darkblue', labelsize=18)
ax1_ml.tick_params(axis='x', labelsize=18)
ax1_ml.set_ylim(0, 350)
ax1_ml.grid(True, axis='y', linestyle=':', linewidth=0.5, color='gray')

# Plot bar graph for BD environment
plot_bar_graph(data_combined_bd, ax1_bd, 0.4, palette)
ax1_bd.set_xlabel('Algorithm', fontsize=18)
ax1_bd.set_ylabel('Time (s)', fontsize=18, color='darkblue')
ax1_bd.set_xticks(data_combined_bd.index)
ax1_bd.set_xticklabels(data_combined_bd['Algorithm'], rotation=0, ha='center', fontsize=14)
ax1_bd.tick_params(axis='y', labelcolor='darkblue', labelsize=18)
ax1_bd.tick_params(axis='x', labelsize=18)
ax1_bd.set_ylim(0, 350)
ax1_bd.grid(True, axis='y', linestyle=':', linewidth=0.5, color='gray')

# Add legend
fig.legend(loc='upper center', fontsize=16, bbox_to_anchor=(0.5, 1.06), ncol=2)

# Adjust layout
plt.tight_layout()
plt.savefig("graph_time_accuracy_by_algorithm_abd.png", bbox_inches='tight')
plt.show()