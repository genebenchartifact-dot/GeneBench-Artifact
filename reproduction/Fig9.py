import matplotlib.pyplot as plt
import pandas as pd
import json

def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

data = load_json('track_results/cruxeval_track.json')
translation = load_json('track_results/translation_track.json')
humaneval = load_json('track_results/humaneval_track.json')
classeval = load_json('track_results/classeval_track.json')

input_prediction = {}
output_prediction = {}

for key in data:
    if "input" in key:
        input_prediction[key.replace("_input", "")] = data[key]
    if "output" in key:
        output_prediction[key.replace("_output", "")] = data[key]
        
models_order = [
    "CodeLlama-13-B",
    "CodeLlama-13-I",
    "CodeLlama-34-I",
    "DeepSeekCoder-6.7-B",
    "DeepSeekCoder-6.7-I",
    "DeepSeekCoder-33-I",
    "SemCoder",
    "StarCoder2",
    "WizardCoder-15",
    "WizardCoder-33",
    "GPT4o",   
]

df_input_prediction = pd.DataFrame(input_prediction).T.reindex(models_order)
df_output_prediction = pd.DataFrame(output_prediction).T.reindex(models_order)
df_translation = pd.DataFrame(translation).T.reindex(models_order)
df_humaneval = pd.DataFrame(humaneval).T.reindex(models_order)
df_classeval = pd.DataFrame(classeval).T.reindex(models_order)

dataframes = [df_input_prediction, df_output_prediction, df_translation, df_humaneval, df_classeval]

for df in dataframes:
    df['Total'] = df.sum(axis=1)
    df['True_True'] = (df['True_True'] / df['Total']) * 100
    df['True_False'] = (df['True_False'] / df['Total']) * 100
    df.drop(columns='Total', inplace=True)
    
avg_true_false = {}
for model in models_order:
    avg = sum(df.loc[model, 'True_False'] for df in dataframes) / len(dataframes)
    avg_true_false[model] = avg

# Plotting setup
fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(25, 3), sharey=True)
plt.tight_layout(pad=1.0)
soft_green = (150/255, 200/255, 150/255)
colors = ['#c0e2ad', '#fbe0e0'] 

for ax, df in zip(axes, dataframes):
    bars = df.plot(kind='barh', stacked=True, color=colors, ax=ax, legend=False)
    # ax.set_xlabel('Percentage (%)', fontsize=20, fontweight='bold')
    ax.invert_yaxis()  
    for bar_container in bars.containers:
        for bar in bar_container:
            width = bar.get_width()
            label_x = bar.get_x() + width/2
            ax.text(label_x+0.6, bar.get_y() + bar.get_height()/2, f'{width:.0f}', 
                    va='center', ha='center', color='black', fontsize=12, ) #fontweight='bold'

axes[-1].set_xlabel('Percentage (%)', fontsize=15, ) #fontweight='bold'


for ax in axes:
    ax.set_xlim(0, 100)
    ax.tick_params(axis='y', labelsize=12)
    # for label in ax.get_xticklabels():
    #     label.set_fontweight('bold')
    # for label in ax.get_yticklabels():
    #     label.set_fontweight('bold') 
    for patch in ax.patches:
        current_height = patch.get_height()
        new_height = 0.95 
        new_y = patch.get_y() + (current_height - new_height) / 2
        patch.set_y(new_y)
        patch.set_height(new_height)

titles = ['CRUXEval-I', 'CRUXEval-O', 
          'Avatar', 'HumanEval', 'ClassEval']
for ax, title in zip(axes, titles):
    ax.set_title(title, fontsize=15, )
    

for ax in axes:
    ax.set_xticks([])
plt.tight_layout(pad=0)
plt.show()
