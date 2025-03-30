import matplotlib.pyplot as plt

plt.style.use('ggplot')

# 数据
n_values = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
cache_improve = [0.048206, 0.190763, 0.436359, 0.764997, 1.19199, 1.72783, 2.21932, 3.09371, 3.92989, 4.88232]
vector_cache_improve = [0.097732, 0.387542, 0.875487, 1.55422, 2.42588, 3.51426, 4.77454, 6.23643, 7.90622, 9.74322]

# 创建图形并绘制折线图
plt.figure(figsize=(10, 6))
plt.plot(n_values, cache_improve, marker='o', linestyle='-', linewidth=2, label='cache_improve')
plt.plot(n_values, vector_cache_improve, marker='s', linestyle='-', linewidth=2, label='vector_cache_improve')

# 添加标题和坐标轴标签
plt.title('Comparison of Performance', fontsize=16)
plt.xlabel('n', fontsize=14)
plt.ylabel('Time (ms)', fontsize=14)

# 添加图例
plt.legend(fontsize=12)

# 调整布局、保存图像并展示
plt.tight_layout()
plt.savefig('vector_comparison.png', dpi=300)
plt.show()
