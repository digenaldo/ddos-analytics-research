import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def load_data(file):
    """Load data from the CSV file."""
    return pd.read_csv(file)

def calculate_statistics(results, environment):
    """Calculate the mean and confidence interval of the results for a specific environment."""
    mean = results.groupby(['environment', 'algorithm'])['iterations_time', 'iterations_accuracy'].mean().loc[environment]
    ci = results.groupby(['environment', 'algorithm'])['iterations_time', 'iterations_accuracy'].std().loc[environment]
    return mean, ci

def map_algorithms(algorithms):
    """Map algorithm names for line breaks."""
    return algorithms.map({
        'nb': 'Naive\nBayes',
        'dt': 'Decision\nTree',
        'lr': 'Logistic\nRegression',
        'rf': 'Random\nForest'
    })

def plot_graph(data, bar_width=0.4):
    """Plot the bar graph."""
    fig, ax = plt.subplots(figsize=(14, 7))
    palette = sns.color_palette("tab10")
    palette = [(r, g, b, 0.2) for r, g, b in palette]

    bar_time = ax.bar(data.index - bar_width/2, data['Mean Time'], bar_width, yerr=data['CI Time'], capsize=5, label='Time', color=palette[0])

    for i in data.index:
        ax.annotate(f'{data.at[i, "Mean Time"]:.2f}',
                    xy=(i - bar_width/2, data.at[i, 'Mean Time']),
                    xytext=(0, 30),
                    textcoords='offset points',
                    ha='center', va='bottom',
                    fontsize=16, color='blue', fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        ax.annotate(f'±{data.at[i, "CI Time"]:.2f}',
                    xy=(i - bar_width/2, data.at[i, 'Mean Time']),
                    xytext=(0, 10),
                    textcoords='offset points',
                    ha='center', va='bottom',
                    fontsize=18, color='blue', fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    ax.set_xlabel('Algorithm', fontsize=18)
    ax.set_ylabel('Time (s)', fontsize=18, color='darkblue')
    ax.set_xticks(data.index)
    ax.set_xticklabels(data['Algorithm'], rotation=0, ha='center', fontsize=14)
    ax.tick_params(axis='y', labelcolor='darkblue', labelsize=18)
    ax.tick_params(axis='x', labelsize=18)
    ax.set_ylim(0, 350)
    ax.grid(True, axis='y', linestyle=':', linewidth=0.5, color='gray')

    return fig

# Load data
results_file = '/content/drive/MyDrive/Mestrado/dataset/attack-ddos-layer7/resultado.csv'
results_data = load_data(results_file)

# Calculate statistics for ML and BD environments
mean_ml, ci_ml = calculate_statistics(results_data, 'ml')
mean_bd, ci_bd = calculate_statistics(results_data, 'bd')

# Map algorithm names for line breaks
mean_ml['Algorithm'] = map_algorithms(mean_ml.index)
mean_bd['Algorithm'] = map_algorithms(mean_bd.index)

# Plot the graph for the ML environment
fig_ml = plot_graph(mean_ml)

# Add legend
fig_ml.legend(loc='upper center', fontsize=16, bbox_to_anchor=(0.5, 1.05), ncol=2)

# Adjust overall appearance of the graph
plt.tight_layout()
plt.savefig("graph_time_accuracy_by_algorithm_apg.png", bbox_inches='tight')
plt.show()