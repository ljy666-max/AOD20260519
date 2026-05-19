import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import itertools
import os

# ==========================================
# 全局字体与绘图格式设置
# ==========================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 12

# ==========================================
# 1. 物理参数与环境设置
# ==========================================
wl = 355e-9           # 激光波长 (m)
w0 = 1.35e-3          # 1/e^2 束腰半径 (m)
V = 5750.0            # 石英晶体声速 (m/s)
z = 1.7               # 传播距离 (m)

# 【核心修改】文献 Fig 3(e) 的二维扫描频率参数
fx_list = [110e6, 140e6]  # X方向的跳变频率
fy_list = [110e6, 140e6]  # Y方向的跳变频率
f_center = 125e6          # 理论中心频率

# 获取所有 2x2 = 4 种频率组合
freq_pairs = list(itertools.product(fx_list, fy_list))

# 计算视窗的理论中心坐标
center_pos_mm = z * np.tan((wl * f_center) / V) * 1e3

# ==========================================
# 2. 空间网格初始化
# ==========================================
L = 60e-3             # 保持 60mm 大网格，防止边界截断
N = 4096              
dx = L / N            
dy = dx

x = np.linspace(-L/2, L/2, N)
y = np.linspace(-L/2, L/2, N)
X, Y = np.meshgrid(x, y)

k = 2 * np.pi / wl
E_in = np.exp(-(X**2 + Y**2) / (w0**2))
I_in = np.abs(E_in)**2

# ==========================================
# 3. 角谱传播法 (ASM) 函数
# ==========================================
def angular_spectrum_propagate(E0, wl, z, dx, dy):
    Ny, Nx = E0.shape
    fx = np.fft.fftfreq(Nx, d=dx)
    fy = np.fft.fftfreq(Ny, d=dy)
    FX, FY = np.meshgrid(fx, fy)
    
    term = 1.0 - (wl * FX)**2 - (wl * FY)**2
    term = np.maximum(term, 0)
    H = np.exp(1j * k * z * np.sqrt(term))
    
    E0_fft = np.fft.fft2(E0)
    return np.fft.ifft2(E0_fft * H)

# ==========================================
# 4. 模拟 2D 阵列扫描 (非相干叠加)
# ==========================================
I_total_out = np.zeros((N, N))
spot_positions_x = []
spot_positions_y = []

# 遍历 4 种频率组合
for fx, fy in freq_pairs:
    # 同时计算 X 和 Y 方向的偏转相位 (无二次啁啾项)
    phi_x = k * (wl * fx / V) * X
    phi_y = k * (wl * fy / V) * Y
    
    E_mod = E_in * np.exp(1j * (phi_x + phi_y))
    E_out = angular_spectrum_propagate(E_mod, wl, z, dx, dy)
    
    I_current = np.abs(E_out)**2
    I_total_out += I_current
    
    # 记录理论靶点位置
    spot_positions_x.append(z * np.tan((wl * fx) / V) * 1e3)
    spot_positions_y.append(z * np.tan((wl * fy) / V) * 1e3)

# 提取唯一的靶点坐标用于绘图标线
unique_targets_x = sorted(list(set(spot_positions_x)))
unique_targets_y = sorted(list(set(spot_positions_y)))
spacing_mm = unique_targets_x[1] - unique_targets_x[0]

# 光强归一化
I_in /= np.max(I_in)
I_total_out /= np.max(I_total_out)

# ==========================================
# 5. 可视化输出与保存
# ==========================================
fig = plt.figure(figsize=(14, 9))
gs = GridSpec(2, 2, figure=fig, width_ratios=[1, 1], height_ratios=[1, 0.6], wspace=0.3, hspace=0.3)

lim_span = 4 # 视窗半宽 4mm (足以包容 3.15mm 间距的阵列)

# --- 图1: 输入光斑 ---
ax1 = fig.add_subplot(gs[0, 0])
ax1.imshow(I_in, extent=[-L/2*1e3, L/2*1e3, -L/2*1e3, L/2*1e3], cmap='jet', origin='lower')
ax1.set_xlim([-lim_span, lim_span])
ax1.set_ylim([-lim_span, lim_span])
ax1.set_aspect('equal')
ax1.set_title("Input Gaussian Beam ($z = 0$ m)")
ax1.set_xlabel("$x$ (mm)")
ax1.set_ylabel("$y$ (mm)")

# --- 图2: 远场 2x2 扫描阵列 ---
ax2 = fig.add_subplot(gs[0, 1])
ax2.imshow(I_total_out, extent=[-L/2*1e3, L/2*1e3, -L/2*1e3, L/2*1e3], cmap='jet', origin='lower')
# 将视窗精准对准阵列几何中心
ax2.set_xlim([center_pos_mm - lim_span, center_pos_mm + lim_span])
ax2.set_ylim([center_pos_mm - lim_span, center_pos_mm + lim_span])
ax2.set_aspect('equal')
ax2.set_title(f"2D Multi-Spot Scan via Frequency Jumps ($z = {z}$ m)\n4 Positions (Paper Fig 3e)")
ax2.set_xlabel("$x$ (mm)")
ax2.set_ylabel("$y$ (mm)")

# --- 侧边栏数据 ---
# 使用三引号 """ 来定义多行字符串，这样就不需要写 \n 了
stats_text = rf"""--- 2D Scan Parameters ---
Freq $X$: 110, 140 MHz
Freq $Y$: 110, 140 MHz
Total Spots: 4
Grid Spacing: {spacing_mm:.2f} mm
Cylinder-Lensing: None
Beam Quality: Preserved"""

fig.text(0.8, 0.75, stats_text, fontsize=13, va='center', ha='left',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#f4f8ff', edgecolor='blue', alpha=0.9))

# --- 图3: 输入X方向剖面 ---
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(x*1e3, I_in[N//2, :], 'k-', linewidth=1.5)
ax3.axhline(np.exp(-2), color='r', linestyle=':', label='$1/e^2$ threshold')
ax3.set_xlim([-lim_span, lim_span])
ax3.set_ylim([-0.05, 1.05])
ax3.set_title("Input Intensity Profile ($x$-axis)")
ax3.set_xlabel("$x$ Position (mm)")
ax3.legend()
ax3.grid(True, alpha=0.5)

# --- 图4: 远场穿过上方两个光斑的X方向剖面 ---
ax4 = fig.add_subplot(gs[1, 1])
# 动态寻找上方一排光斑的 Y 坐标索引
slice_y_mm = unique_targets_y[1]
slice_y_idx = np.argmin(np.abs(y*1e3 - slice_y_mm))

ax4.plot(x*1e3, I_total_out[slice_y_idx, :], 'b-', linewidth=1.5, label=f'Slice at $y = {slice_y_mm:.1f}$ mm')

# 画出X轴的理论靶线
for idx, target_x in enumerate(unique_targets_x):
    label = 'Target Centers' if idx == 0 else ""
    ax4.axvline(target_x, color='g', linestyle='--', alpha=0.6, label=label)

ax4.axhline(np.exp(-2), color='r', linestyle=':')
ax4.set_xlim([center_pos_mm - lim_span, center_pos_mm + lim_span])
ax4.set_ylim([-0.05, 1.05])
ax4.set_title("Output X-Profile (Through Top Row)")
ax4.set_xlabel("$x$ Position (mm)")
ax4.legend(loc='upper center')
ax4.grid(True, alpha=0.5)

plt.subplots_adjust(left=0.08, right=0.88, top=0.92, bottom=0.08)

# 保存图像
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
save_path = os.path.join(current_dir, "AOD_Scanning_Fig3e.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"2D 阵列扫描图像已成功保存至:\n{save_path}")

plt.show()