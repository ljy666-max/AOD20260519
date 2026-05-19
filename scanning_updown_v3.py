import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
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

# AOD 基础参数
f_center = 125e6      # 几何中心频率
fy_constant = 125e6   # Y轴保持恒定偏转，使其在屏幕上有个基础高度


# 【核心修改】X轴步进扫描序列参数 (增大频率差以分开光斑)
fx_start = 95e6       # 起始频率 95 MHz
delta_fx = 30e6       # 步进频率 30 MHz (与文献保持同等间距)
num_steps = 3         # 扫描点数 (共3个点：95, 125, 155 MHz)

# 生成步进频率数组
fx_array = fx_start + np.arange(num_steps) * delta_fx

# 计算视窗的理论中心(使得相机视角一直盯着偏转后的光斑群)
y_center_mm = z * np.tan((wl * fy_constant) / V) * 1e3
x_center_mm = z * np.tan((wl * f_center) / V) * 1e3

# ==========================================
# 2. 空间网格初始化
# ==========================================
L = 60e-3             # 网格扩大至 40mm
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
# 4. 模拟 X 轴步进扫描 (非相干叠加)
# ==========================================
I_total_out = np.zeros((N, N))
spot_x_positions = []

# 预先计算 Y 轴的固定线性相位
phi_y = k * (wl * fy_constant / V) * Y

for fx in fx_array:
    # 计算当前 X 轴频率带来的线性相位
    phi_x = k * (wl * fx / V) * X
    
    # 施加 2D 偏转相位 (无二次啁啾项)
    E_mod = E_in * np.exp(1j * (phi_x + phi_y))
    E_out = angular_spectrum_propagate(E_mod, wl, z, dx, dy)
    
    # 获取当前脉冲光强并累加
    I_current = np.abs(E_out)**2
    I_total_out += I_current
    
    # 记录该点的理论 X 偏转位置
    spot_x_positions.append(z * np.tan((wl * fx) / V) * 1e3)

# 光强归一化
I_in /= np.max(I_in)
I_total_out /= np.max(I_total_out)

# ==========================================
# 5. 可视化输出与保存
# ==========================================
fig = plt.figure(figsize=(14, 9))
gs = GridSpec(2, 2, figure=fig, width_ratios=[1, 1], height_ratios=[1, 0.6], wspace=0.3, hspace=0.3)

lim_span = 8 # 视窗半宽 5mm，稍微开大一点以容纳一整排光斑

# --- 图1: 输入光斑 ---
ax1 = fig.add_subplot(gs[0, 0])
ax1.imshow(I_in, extent=[-L/2*1e3, L/2*1e3, -L/2*1e3, L/2*1e3], cmap='jet', origin='lower')
ax1.set_xlim([-lim_span, lim_span])
ax1.set_ylim([-lim_span, lim_span])
ax1.set_aspect('equal')
ax1.set_title("Input Gaussian Beam ($z = 0$ m)")
ax1.set_xlabel("$x$ (mm)")
ax1.set_ylabel("$y$ (mm)")

# --- 图2: X轴线性扫描光斑群 ---
ax2 = fig.add_subplot(gs[0, 1])
ax2.imshow(I_total_out, extent=[-L/2*1e3, L/2*1e3, -L/2*1e3, L/2*1e3], cmap='jet', origin='lower')
# 视窗平移至 X 和 Y 的基础偏转中心
ax2.set_xlim([x_center_mm - lim_span, x_center_mm + lim_span])
ax2.set_ylim([y_center_mm - lim_span, y_center_mm + lim_span])
ax2.set_aspect('equal')
ax2.set_title(f"1D Linear Step Scan along X-axis ($z = {z}$ m)\nDiscrete Pulse Accumulation")
ax2.set_xlabel("$x$ (mm)")
ax2.set_ylabel("$y$ (mm)")

# --- 侧边栏数据 ---
stats_text = (
    f"--- X-Scan Parameters ---\n"
    f"Start Freq: {fx_start/1e6:.1f} MHz\n"
    rf"Step $\Delta f_x$: {delta_fx/1e6:.1f} MHz\n"
    f"Number of Spots: {num_steps}\n"
    f"Y-Freq (Const): {fy_constant/1e6:.1f} MHz\n"
    rf"Spacing $\Delta x$: {abs(spot_x_positions[1]-spot_x_positions[0]):.2f} mm\n"
    f"Beam Quality: Preserved"
)
fig.text(0.85, 0.75, stats_text, fontsize=13, va='center', ha='left',
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

# --- 图4: 远场X方向多峰剖面 ---
ax4 = fig.add_subplot(gs[1, 1])
# 提取穿过靶点群中心的 X 剖面
peak_y_idx = np.argmin(np.abs(y*1e3 - y_center_mm))
ax4.plot(x*1e3, I_total_out[peak_y_idx, :], 'b-', linewidth=1.5, label='Accumulated Profile')

# 画出理论靶点位置的虚线
for idx, pos in enumerate(spot_x_positions):
    label = 'Target Centers' if idx == 0 else ""
    ax4.axvline(pos, color='g', linestyle='--', alpha=0.6, label=label)

ax4.axhline(np.exp(-2), color='r', linestyle=':')
ax4.set_xlim([x_center_mm - lim_span, x_center_mm + lim_span])
ax4.set_ylim([-0.05, 1.05])
ax4.set_title("Output Intensity Profile (Array of Spots)")
ax4.set_xlabel("$x$ Position (mm)")
ax4.legend(loc='upper right')
ax4.grid(True, alpha=0.5)

plt.subplots_adjust(left=0.08, right=0.88, top=0.92, bottom=0.08)

# 保存图像
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
save_path = os.path.join(current_dir, "AOD_Linear_Scan_X.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"X方向线性步进扫描图像已成功保存至:\n{save_path}")

plt.show()