import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

n_values = [2**i for i in range(10, 27)]  
methods = {
    'common':          [0.0007, 0.0014, 0.0028, 0.0055, 0.0105, 0.0208, 0.0415, 
                      0.0833, 0.1708, 0.3448, 0.7161, 1.4774, 3.466, 8.3573, 
                      18.1312, 35.8264, 71.453],
    'cache_improve2':  [0.0005, 0.001, 0.0019, 0.0036, 0.0075, 0.015, 0.0317,
                      0.0701, 0.1543, 0.2901, 0.6191, 1.2294, 3.2754, 7.2743,
                      14.549, 29.2599, 58.4473],
    'cache_improve4':  [0.0003, 0.0006, 0.0012, 0.0024, 0.005, 0.0101, 0.0217,
                      0.0641, 0.1344, 0.2578, 0.5513, 1.1685, 3.003, 6.8182,
                      13.7774, 30.0607, 61.3131],
    'circle':          [0.0016, 0.0032, 0.0063, 0.0126, 0.0252, 0.0506, 0.1035,
                      0.2995, 0.6506, 1.3547, 2.7545, 5.6227, 12.43, 28.5927,
                      60.6917, 124.598, 253.781]
}


plt.figure(figsize=(14, 8), dpi=100)


for method, times in methods.items():
    plt.plot(n_values, times, marker='o', markersize=6, linewidth=2, label=method)


plt.xscale('log', base=2)
plt.yscale('log')
plt.xticks(n_values, [r'$2^{%d}$' % np.log2(n) for n in n_values], rotation=45)
plt.yticks([0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000], 
          ['0.0001', '0.001', '0.01', '0.1', '1', '10', '100', '1000'])

plt.xlabel('数组规模 (n)', fontsize=12)
plt.ylabel('运行时间 (ms)', fontsize=12)
plt.title('不同规模下各算法的性能对比', fontsize=14, pad=20)

plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.legend(fontsize=10, loc='upper left')

plt.tight_layout()

plt.savefig('problem2_comparison.png', bbox_inches='tight')
plt.show()