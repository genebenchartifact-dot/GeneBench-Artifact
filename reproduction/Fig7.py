import pandas as pd
import matplotlib.pyplot as plt

Names = {
    "after/avatar_complexity.csv": "(a) Avatar",
    "after/classeval_complexity.csv": "(b) ClassEval",
    "after/cruxeval_complexity.csv": "(c) CRUXEval",
    "after/humaneval_complexity.csv": "(d) HumanEval",
    "after/swe-bench_complexity.csv": "(e) SWE-Sub",
}

def plot_box_charts(csv_paths):
    original_columns = [
        "Base complexity", 
        "Predicates with operators", 
        "Nested levels", 
        "Complex code structures", 
        "Third-Party calls", 
        "Inter_dependencies", 
        "Intra_dependencies"
    ]
    
    new_columns = ['C' + str(i + 1) for i in range(len(original_columns))]
    
    rows = 1  
    cols = 5 

    fig, axs = plt.subplots(rows, cols, figsize=(15, 3))
    axs = axs.flatten()

    for idx, (ax, csv_path) in enumerate(zip(axs, csv_paths)):
        df = pd.read_csv(csv_path)
        
        rename_dict = dict(zip(original_columns, new_columns))
        df.rename(columns=rename_dict, inplace=True)

        boxplot = df[new_columns].boxplot(patch_artist=True, return_type='dict', ax=ax)
        colors = ['#a0c4ff', '#b5ea8c', '#d6e6ff', '#ffd6a5', '#fbe0e0', '#bdb2ff', 'pink']
        ax.grid(False)
        
        for i, color in enumerate(colors):
            plt.setp(boxplot['boxes'][i], color=color, facecolor=color)
            plt.setp(boxplot['medians'][i], color="#525e75", linewidth=2)
            plt.setp(boxplot['fliers'][i], marker='o', color="grey", markersize=3)
            plt.setp(boxplot['whiskers'][2*i:2*i+2], color="black")
            plt.setp(boxplot['caps'][2*i:2*i+2], color="black")

        ax.set_xlabel(Names[csv_path], fontsize=14)
        ax.tick_params(axis='x', labelsize=12)

        if idx == 0:
            ax.set_ylabel('Metric Value', fontsize=14)
            ax.tick_params(axis='y', labelsize=12)
        elif "swe-bench_complexity.csv" in csv_path:
            ax.tick_params(axis='y', labelsize=12)
        else:
            ax.set_ylabel('')
            ax.tick_params(axis='y', labelleft=False)

        if "swe-bench_complexity.csv" in csv_path:
            ax.set_ylim(0, 70)
        else:
            ax.set_ylim(0, 25)

    plt.tight_layout(w_pad=0.5)
    plt.show()

plot_box_charts([
    "after/avatar_complexity.csv", 
    "after/classeval_complexity.csv",
    "after/cruxeval_complexity.csv",
    "after/humaneval_complexity.csv",  
    "after/swe-bench_complexity.csv"
])
