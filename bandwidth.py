import numpy as np
import matplotlib.pyplot as plt

# =====================================================================
# 1. 物理参数与硬件边界设定 (根据你的实验配置)
# =====================================================================
V = 5750.0            # 石英晶体声速 (m/s)
w0 = 1.35e-3          # 激光束腰半径 (m)
fc = 75.0             # 中心频率 (MHz)
f_min = 60.0          # 硬件带宽下限 (MHz)
f_max = 90.0          # 硬件带宽上限 (MHz)

# 全局像差幅度调节因子 (你可以调节这个值，看曲线何时报警变红)
A_scale = 1.2         

# 建立归一化物理坐标系 (在束腰范围内 -1 到 +1)
N = 1000
xn = np.linspace(-1.0, 1.0, N)

# =====================================================================
# 2. 读取你的编译器输出公式，计算频率梯度 dPhi/dx
# f_local = fc + (V * A_scale / (2 * pi * w0 * 1e6)) * dPhi/dx
# =====================================================================
scale_factor = (V * A_scale) / (2 * np.pi * w0 * 1e6)

# --- AOD_1 (0°, X轴): Φ(X) = 1.6667*X^4 + 0.6667*X^3
# dPhi/dx = 6.6668*X^3 + 2.0001*X^2
df_AOD1 = scale_factor * (6.6668 * xn**3 + 2.0001 * xn**2)
f_AOD1 = fc + df_AOD1

# --- AOD_2 (90°, Y轴): Φ(Y) = 1.6667*Y^4
# dPhi/dy = 6.6668*Y^3
df_AOD2 = scale_factor * (6.6668 * xn**3)
f_AOD2 = fc + df_AOD2

# --- AOD_3 / AOD_4 (±45°, U/V轴): Φ(U) = 0.6667*U^4 + 1.8856*U^3
# dPhi/du = 2.6668*U^3 + 5.6568*U^2
df_AOD3 = scale_factor * (2.6668 * xn**3 + 5.6568 * xn**2)
f_AOD3 = fc + df_AOD3
f_AOD4 = f_AOD3  # AOD_4 与 AOD_3 结构对称，曲线一致

# 合并数据便于遍历画图
aod_data = [
    {"name": "AOD_1 (0°, X 轴)", "freq": f_AOD1, "color": "#1f77b4"},
    {"name": "AOD_2 (90°, Y 轴)", "freq": f_AOD2, "color": "#2ca02c"},
    {"name": "AOD_3 (+45°, U 轴)", "freq": f_AOD3, "color": "#ff7f0e"},
    {"name": "AOD_4 (-45°, V 轴)", "freq": f_AOD4, "color": "#bcbd22"}
]

# =====================================================================
# 3. 绘制高颜值学术级图表
# =====================================================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'SimSun'] # 双语兼容
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 11

fig, axes = plt.subplots(2, 2, figsize=(12, 9), facecolor='#fcfcfc')
axes = axes.flatten()

print("--- 级联 AOD 射频工作频偏极限分析 ---")

for idx, aod in enumerate(aod_data):
    ax = axes[idx]
    freq = aod["freq"]
    
    # 检测局部工作频率是否超出 [60, 90] MHz 物理阈值
    exceed_upper = freq > f_max
    exceed_lower = freq < f_min
    is_safe = not (np.any(exceed_upper) or np.any(exceed_lower))
    
    # 打印终端状态，供调试和报告引用
    print(f"{aod['name']}:")
    print(f"  -> 局部工作频率区间: [{np.min(freq):.2f} MHz, {np.max(freq):.2f} MHz]")
    if is_safe:
        print("  -> ✅ 状态: 安全运行\n")
    else:
        print("  -> ❌ 状态: 超出硬件带宽限制 (发生削顶畸变!)\n")
        
    # 分段绘制：安全区域绘制原色/绿色，超限区域强闪红色
    ax.plot(xn, freq, color=aod["color"], linewidth=2.5, label='工作射频频率')
    
    # 绘制超限高亮
    unsafe_mask = (freq > f_max) | (freq < f_min)
    if np.any(unsafe_mask):
        # 找出超限的坐标点并高亮为红色
        ax.scatter(xn[unsafe_mask], freq[unsafe_mask], color='red', s=5, zorder=5, label='超限区域 (带宽警告!)')
        
    # 绘制硬件带宽红线
    ax.axhline(f_max, color='red', linestyle='--', linewidth=1.2, alpha=0.8, label='硬件带宽上限 (90 MHz)')
    ax.axhline(f_min, color='red', linestyle='--', linewidth=1.2, alpha=0.8, label='硬件带宽下限 (60 MHz)')
    ax.axhline(fc, color='gray', linestyle=':', linewidth=1.0, alpha=0.6, label='中心载波 (75 MHz)')
    
    # 装饰轴线
    ax.set_title(aod["name"], fontsize=12, fontweight='bold')
    ax.set_xlabel("归一化空间坐标 $z_n$ ($z/w_0$)", fontsize=10)
    ax.set_ylabel("射频局部工作频率 (MHz)", fontsize=10)
    ax.set_xlim([-1.05, 1.05])
    ax.set_ylim([45.0, 105.0])
    ax.grid(True, linestyle=':', alpha=0.5)
    
    # 状态水印
    if is_safe:
        ax.text(-0.9, 50, "Safe Operation (安全)", color='green', fontsize=12, 
                fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='green'))
    else:
        ax.text(-0.9, 50, "LIMIT EXCEEDED (超限!)", color='red', fontsize=12, 
                fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='red'))
        
    ax.legend(loc='upper left', fontsize=8)

plt.suptitle(f"级联多轴 AOD 工作射频带宽安全评估 (像差缩放因子 A_scale = {A_scale})", 
             fontsize=15, fontweight='bold', y=0.98, fontproperties='SimHei')
plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.show()