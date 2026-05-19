import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os

# ==========================================
# 全局字体与绘图格式设置 (新罗马字体)
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

# 频率跳变参数 (对应文献 Fig 3b)
f_center = 125e6            # 理论中心频率 (用于对齐相机视窗)
frequencies = [110e6, 140e6] # 两个交替跳变的频率

# 计算预期的中心坐标，以方便我们将相机视区对准目标
y_center = z * np.tan((wl * f_center) / V)

# ==========================================
# 2. 空间网格初始化
# ==========================================
L = 40e-3             # 网格扩大至 40mm
N = 2048              
dx = L / N            
dy = dx

x = np.linspace(-L/2, L/2, N)
y = np.linspace(-L/2, L/2, N)
X, Y = np.meshgrid(x, y)

k = 2 * np.pi / wl
E_in = np.exp(-(X**2 + Y**2) / (w0**2))

# ==========================================
# 3. 角谱传播法 (ASM) 函数定义
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
# 4. 模拟跳频扫描与光强叠加
# ==========================================
I_total_out = np.zeros((N, N))
peak_positions = []

for f in frequencies:
    # 静态跳频：仅有线性相位，没有任何啁啾二次项！
    theta = (wl * f) / V
    phi_linear = k * theta * Y
    
    E_mod = E_in * np.exp(1j * phi_linear)
    E_out = angular_spectrum_propagate(E_mod, wl, z, dx, dy)
    
    # 获取当前频率的光强并累加 (模拟相机对多个脉冲的曝光积分)
    I_current = np.abs(E_out)**2
    I_total_out += I_current
    
    # 记录该频率导致的光斑中心物理位置
    peak_y_idx = np.unravel_index(np.argmax(I_current), I_current.shape)[0]
    peak_positions.append(y[peak_y_idx])

# 光强归一化
I_in = np.abs(E_in)**2
I_in /= np.max(I_in)
I_total_out /= np.max(I_total_out)

# 计算两光斑间距
spot_distance = abs(peak_positions[1] - peak_positions[0])

# ==========================================
# 5. 提取剖面与分析 (取中心轴 x=0)
# ==========================================
profile_y_in = I_in[:, N//2]  
profile_y_out = I_total_out[:, N//2] # 提取包含双峰的 Y 轴剖面

# ==========================================
# 6. 可视化输出与保存
# ==========================================
fig = plt.figure(figsize=(14, 9))
gs = GridSpec(2, 2, figure=fig, width_ratios=[1, 1], height_ratios=[1, 0.6], wspace=0.3, hspace=0.3)

lim_span = 4 # 视窗半宽 4mm (画框 8x8 mm)

# --- 图1: 输入光斑 ---
ax1 = fig.add_subplot(gs[0, 0])
ax1.imshow(I_in, extent=[-L/2*1e3, L/2*1e3, -L/2*1e3, L/2*1e3], cmap='jet', origin='lower')
ax1.set_xlim([-lim_span, lim_span])
ax1.set_ylim([-lim_span, lim_span])
ax1.set_aspect('equal')
ax1.set_title("Input Gaussian Beam ($z = 0$ m)")
ax1.set_xlabel("$x$ (mm)")
ax1.set_ylabel("$y$ (mm)")

# --- 图2: 远场双频扫描光斑 ---
ax2 = fig.add_subplot(gs[0, 1])
ax2.imshow(I_total_out, extent=[-L/2*1e3, L/2*1e3, -L/2*1e3, L/2*1e3], cmap='jet', origin='lower')
ax2.set_xlim([-lim_span, lim_span])
# 将视窗精准对焦在 125MHz 对应的中心位置，上下延伸 4mm
y_center_mm = y_center * 1e3
ax2.set_ylim([y_center_mm - lim_span, y_center_mm + lim_span])
ax2.set_aspect('equal')
ax2.set_title(f"Fast Scanning via Frequency Jumps ($z = {z}$ m)\nNo Cylinder-Lensing Effect")
ax2.set_xlabel("$x$ (mm)")
ax2.set_ylabel("$y$ (mm)")

# --- 侧边栏数据 ---
stats_text = (
    f"--- Scan Parameters ---\n"
    f"Frequencies: 110 & 140 MHz\n"
    f"Spot 1 Pos: {peak_positions[0]*1e3:.2f} mm\n"
    f"Spot 2 Pos: {peak_positions[1]*1e3:.2f} mm\n"
    f"Spot Distance: {spot_distance*1e3:.2f} mm\n"
    f"Beam Quality: Preserved (Circular)"
)
fig.text(0.85, 0.75, stats_text, fontsize=13, va='center', ha='left',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#f9f9f9', edgecolor='gray', alpha=0.9))

# --- 图3: 输入Y方向剖面 ---
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(y*1e3, profile_y_in, 'k-', linewidth=1.5)
ax3.axhline(np.exp(-2), color='r', linestyle=':', label='$1/e^2$ threshold')
ax3.set_xlim([-lim_span, lim_span])
ax3.set_ylim([-0.05, 1.05])
ax3.set_title("Input Intensity Profile")
ax3.set_xlabel("$y$ Position (mm)")
ax3.legend()
ax3.grid(True, alpha=0.5)

# --- 图4: 远场Y方向双峰剖面 ---
ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(y*1e3, profile_y_out, 'b-', linewidth=1.5, label='Accumulated Profile')
ax4.axvline(peak_positions[0]*1e3, color='g', linestyle='--', alpha=0.7)
ax4.axvline(peak_positions[1]*1e3, color='g', linestyle='--', alpha=0.7, label='Beam Centers')
ax4.axhline(np.exp(-2), color='r', linestyle=':')
ax4.set_xlim([y_center_mm - lim_span, y_center_mm + lim_span])
ax4.set_ylim([-0.05, 1.05])
ax4.set_title("Output Intensity Profile (Dual Spots)")
ax4.set_xlabel("$y$ Position (mm)")
ax4.legend(loc='upper center')
ax4.grid(True, alpha=0.5)

plt.subplots_adjust(left=0.08, right=0.88, top=0.92, bottom=0.08)

# 保存图像
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
save_path = os.path.join(current_dir, "AOD_Scanning_Fig3b.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"1D 频率跳变扫描图像已成功保存至:\n{save_path}")

plt.show()