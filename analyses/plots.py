import pandas as pd
import os
import matplotlib
matplotlib.use('MacOSX')

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Directory path
dir_path = 'results/BEA/ci/'

condition_name_map = {
    'Pretext Only': 'Baseline',
    'Pretext + Words': 'Words',
    'Pretext + Sentences': 'Sentences',
    'Control': 'Control'
}
conditions = ['Control', 'Baseline', 'Words', 'Sentences']
# Model file names
model_files = {
    'gpt-3.5': 'gpt-3.5_ci_stats.csv',
    'gpt-4': 'gpt-4_ci_stats.csv',
    'llama3': 'llama3_ci_stats.csv',
    'mistral7b': 'mistral_ci_stats.csv'
}

score_types = {
    "semantic": "Semantic Score",
    "f1": "F1 Score",
    "jaccard": "Jaccard Index"
}

def extract_score_group(df, prefix, conditions):
    prefix_map = {
        "semantic": "Semantic",
        "f1": "F1",
        "jaccard": "Jaccard"
    }
    measure_name = prefix_map[prefix]

    means = []
    lowers = []
    uppers = []

    for cond in conditions:
        filtered = df[(df['condition'] == cond) & (df['measure'] == measure_name)]
        if not filtered.empty:
            means.append(filtered['mean'].values[0])
            lowers.append(filtered['95% CI Lower'].values[0])
            uppers.append(filtered['95% CI Upper'].values[0])
        else:
            means.append(float('nan'))
            lowers.append(float('nan'))
            uppers.append(float('nan'))

    return {'mean': means, 'lower': lowers, 'upper': uppers}

def plot_scores_with_ci_all_models(ci_dfs, conditions):
    for prefix, label in score_types.items():
        plt.figure(figsize=(7, 6))
        x = range(len(conditions))
        ax = plt.gca()

        for model_name, ci_df in ci_dfs.items():
            stats = extract_score_group(ci_df, prefix, conditions)
            line = plt.plot(
                x, stats['mean'],
                marker='o',
                markersize=10,
                linewidth=3,
                label=model_name
            )
            color = line[0].get_color()
            plt.fill_between(x, stats['lower'], stats['upper'], alpha=0.3, color=color)
            plt.plot(x, stats['lower'], linestyle='--', color=color, alpha=0.7, linewidth=1)
            plt.plot(x, stats['upper'], linestyle='--', color=color, alpha=0.7, linewidth=1)

        # Round y-axis ticks to 3 decimals
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))
        plt.yticks(fontsize=14)
        # Custom ylim padding
        ymin, ymax = ax.get_ylim()
        if prefix == "semantic":
            ax.set_ylim(ymin, ymax * 1.001)
        else:
            ax.set_ylim(ymin, ymax * 1.008)

        plt.xticks(x, conditions, fontsize=14)
        plt.xlabel('Prompting Condition', fontsize=15)
        plt.ylabel(label, fontsize=15)
        plt.grid(False)

        # Legend inside plot, close to top
        plt.legend(
            fontsize=12,
            ncol=4,
            loc='upper center',
            bbox_to_anchor=(0.5, 0.99),
            borderaxespad=0
        )
        plt.tight_layout()
        plt.savefig(f"plots/{label.replace(' ', '_').lower()}.png", format='png')
        plt.show()

ci_dfs = {}
for model_name, filename in model_files.items():
    filepath = os.path.join(dir_path, filename)
    df = pd.read_csv(filepath)
    df['condition'] = df['condition'].map(condition_name_map)
    ci_dfs[model_name] = df

plot_scores_with_ci_all_models(ci_dfs, conditions)
