import matplotlib.pyplot as plt
import numpy as np
import json

def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)


complexity = load_json('inter_nums/complexity_count_results.json')
CRUXEval = load_json('inter_nums/cruxeval_count_results.json')
HumanEval = load_json('inter_nums/humaneval_count_results.json')
ClassEval = load_json('inter_nums/classeval_count_results.json')
Avatar = load_json('inter_nums/avatar_count_results.json')


operator_mapping = {
    'AddNestedFor': 'S1',
    'AddNestedIf': 'S2',
    'AddNestedWhile': 'S3',
    'AddThread': 'S4',
    'AddTryExcept': 'S5',
    'CreateFunction': 'S6',
    'CreateModuleDependencies': 'S7',
    'IntroduceDecorator': 'S8',
    'ReplaceNumpy': 'S9',
    'TransformAugAssignment': 'S10',
    'TransformLoopToRecursion': 'S11',
    'TransformPrimToCompound': 'S12',
    'AddBase64': 'A1',
    'AddCrypto': 'A2',
    'AddDatetime': 'A3',
    'AddDateUtil': 'A4',
    'AddHttp': 'A5',
    'AddScipy': 'A6',
    'AddSklearn': 'A7',
    'AddTime': 'A8',
    'RenameVariable': 'N1',
    'RenameFunction': 'N2'
}

operators = list(operator_mapping.keys())
labels = [operator_mapping[op] for op in operators]
labels = [r"$" + label[0] + "_{" + label[1:] + "}$" for label in labels]


CRUX_values   = [CRUXEval.get(op, 0) / 200 for op in operators]
Human_values  = [HumanEval.get(op, 0) / 164 for op in operators]
Class_values  = [ClassEval.get(op, 0) / 100 for op in operators]
Avatar_values = [Avatar.get(op, 0) / 250 for op in operators]

avg_freq = [(c + h + cl + a) / 4 for c, h, cl, a in zip(CRUX_values, Human_values, Class_values, Avatar_values)]
avg_complexity_pct = [complexity.get(op, 0) * 100 for op in operators]

x = np.arange(len(operators))
width = 0.2

fig, ax = plt.subplots(figsize=(14, 3))

ax.bar(x - 1.5 * width, CRUX_values, width, label='CRUXEval', color='#ffc8dd',edgecolor='grey')
ax.bar(x - 0.5 * width, Human_values, width, label='HumanEval', color='#f8cc1b' ,edgecolor='grey')
ax.bar(x + 0.5 * width, Class_values, width, label='ClassEval', color='#a0c4ff', edgecolor='grey')
ax.bar(x + 1.5 * width, Avatar_values, width, label='Avatar', color='#b6e2d3', edgecolor='grey')

line1, = ax.plot(x, avg_freq, marker='o', linestyle='-', color='black', linewidth=2, label='Avg Frequency')
ax.set_ylabel('Frequency', fontsize=14, ) #fontweight='bold'
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=13)
ax.grid(False)

ax2 = ax.twinx()
line2, = ax2.plot(x, avg_complexity_pct, marker='o', linestyle='--', color='magenta', linewidth=2, label='Avg ΔComplexity')

ax2.set_ylabel('Δ Complexity (%)', fontsize=14, color='magenta',) # fontweight='bold'
ax2.tick_params(axis='y', labelcolor='magenta', labelsize=12)
ax.tick_params(axis='y', labelsize=12)

ax2.set_ylim(0, 100) 
ax2.spines['right'].set_color('magenta')  

lines = [line1, line2]
handles, labels_leg = ax.get_legend_handles_labels()
handles2, labels_leg2 = ax2.get_legend_handles_labels()
ax.legend(handles + handles2, labels_leg + labels_leg2, loc='upper left')

fig.tight_layout()
plt.show()