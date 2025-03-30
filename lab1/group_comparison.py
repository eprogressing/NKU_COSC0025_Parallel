import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'axes.linewidth': 0.8,
    'grid.color': '#D3D3D3',
    'grid.linestyle': ':',
    'grid.linewidth': 0.5,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight'
})


data = {
    "Group2": {
        "N": [100, 200, 300, 400, 500, 600, 700, 800, 900],
        "common": [0.05064, 0.199102, 0.457049, 0.821042, 1.36919, 1.9177, 2.43425, 3.6368, 4.75876],
        "cache_improve": [0.048206, 0.190763, 0.436359, 0.764997, 1.19199, 1.72783, 2.21932, 3.09371, 3.92989],
        "unroll":[0.035159,0.139565,0.334223, 0.602684,1.02809,1.42339,1.80553,2.54792,3.66033]
    }
}

fig, ax = plt.subplots(figsize=(7, 5.5)) 


colors = {
    'common': '#006BA4',      
    'cache_improve': '#FF800E', 
    'unroll': '#2CA02C'       
}


markers = {
    'common': 'o',       
    'cache_improve': '^',  
    'unroll': 's'         
}


group_name = "Group2"
group_data = data[group_name]
df = pd.DataFrame(group_data)


for col in ['common', 'cache_improve', 'unroll']:
    ax.plot(df['N'], df[col], 
            marker=markers[col],
            markersize=5,
            linewidth=1.5,
            color=colors[col],
            markeredgecolor='k',
            markeredgewidth=0.3,
            linestyle='-',
            zorder=3)  


ax.set_xlabel('Matrix Size (N)', labelpad=5)
ax.set_ylabel('Execution Time (ms)', labelpad=5)
ax.set_title('Performance Comparison with Loop Unrolling', pad=12, fontweight='bold')


ax.set_xticks(df['N'][::2])  
ax.tick_params(axis='x', rotation=35)
ax.grid(True, alpha=0.3)

ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
ax.yaxis.get_offset_text().set_fontsize(9)


ax.spines[['top', 'right']].set_visible(False)


legend_elements = [
    plt.Line2D([0], [0], marker=markers['common'], color=colors['common'],
               label='Baseline', markersize=7, linewidth=0),
    plt.Line2D([0], [0], marker=markers['cache_improve'], color=colors['cache_improve'],
               label='Cache Optimized', markersize=7, linewidth=0),
    plt.Line2D([0], [0], marker=markers['unroll'], color=colors['unroll'],
               label='Unrolled Loop', markersize=7, linewidth=0)
]

ax.legend(handles=legend_elements,
          loc='upper left',
          frameon=True,
          framealpha=0.95,
          edgecolor='#FFFFFF',
          bbox_to_anchor=(0.02, 0.98))


ax.text(750, 4.2, 'Performance Gain: 17.3%', rotation=18, 
        fontsize=9, color=colors['unroll'], ha='center',
        bbox=dict(facecolor='white', edgecolor='#D3D3D3', boxstyle='round'))


plt.tight_layout(pad=2)


plt.savefig('Group2_Full_Comparison.png', format='png', dpi=600)
plt.show()