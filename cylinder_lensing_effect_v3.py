import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os

# ==========================================
# 全局字体与绘图格式设置 (新罗马字体)
# ==========================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'  # 公式字体匹配新罗马
plt.rcParams['font.size'] = 12

# ==========================================
# 1. 物理参数与环境设置
# ==========================================
wl = 355e-9           # 激光波长 (m)
w0 = 1.35e-3          # 1/e^2 束腰半径 (m)
V = 5750.0            # 石英晶体声速 (m/s)
dfdt = 3e13           # 频率啁啾率 (Hz/s)
z = 1.7               # 传播距离 (m)

# 中心偏转设置
f0 = 125e6            # AOD中心工作频率 125 MHz
theta0 = (wl * f0) / V  # 基础偏转角 (rad)
expected_shift = z * np.tan(theta0) # 预计在屏幕上的纵向偏移量

# ==========================================
# 2. 空间网格初始化
# ==========================================
L = 40e-3             # 网格扩大至 40mm，容纳偏转光束
N = 2048              
dx = L / N            
dy = dx

x = np.linspace(-L/2, L/2, N)
y = np.linspace(-L/2, L/2, N)
X, Y = np.meshgrid(x, y)

# ==========================================
# 3. 生成输入光场与相位调制
# ==========================================
E_in = np.exp(-(X**2 + Y**2) / (w0**2))
k = 2 * np.pi / wl
phi_linear = k * theta0 * Y                     # 线性相位 (宏观偏转)
phi_quad = (np.pi / V**2) * dfdt * (Y**2)       # 二次相位 (柱面镜效应/像散)
E_mod = E_in * np.exp(1j * (phi_linear + phi_quad))

# ==========================================
# 4. 角谱传播法 (ASM)
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
    E_out = np.fft.ifft2(E0_fft * H)
    return E_out

E_out = angular_spectrum_propagate(E_mod, wl, z, dx, dy)

# ==========================================
# 5. 光强计算与参数提取
# ==========================================
I_in = np.abs(E_in)**2
I_out = np.abs(E_out)**2
I_in /= np.max(I_in)
I_out /= np.max(I_out)

# 动态寻找远场光斑的能量峰值中心
peak_y_idx, peak_x_idx = np.unravel_index(np.argmax(I_out), I_out.shape)
simulated_shift = y[peak_y_idx]                # 仿真计算的中心偏移量
simulated_theta = np.arctan(simulated_shift/z) # 仿真计算的偏转角

# 提取剖面
profile_y_in = I_in[:, N//2]  
profile_x_out = I_out[peak_y_idx, :] 
profile_y_out = I_out[:, peak_x_idx] 

# 计算 1/e^2 宽度和椭圆度
threshold = np.exp(-2)
wx_out = np.sum(profile_x_out > threshold) * dx / 2
wy_out = np.sum(profile_y_out > threshold) * dy / 2
ellipticity = min(wx_out, wy_out) / max(wx_out, wy_out)

# --- 终端输出打印 ---
print("\n" + "="*45)
print("AOD 高速偏转与像散仿真结果 (z = 1.7m)")
print("="*45)
print(f"理论基础偏转角: {theta0*1e3:.2f} mrad")
print(f"仿真实际偏转角: {simulated_theta*1e3:.2f} mrad")
print("-" * 45)
print(f"理论中心偏移量: {expected_shift*1e3:.2f} mm")
print(f"仿真中心偏移量: {simulated_shift*1e3:.2f} mm")
print("-" * 45)
print(f"远场光斑 X轴半径 (wx): {wx_out*1e3:.3f} mm")
print(f"远场光斑 Y轴半径 (wy): {wy_out*1e3:.3f} mm")
print(f"远场光斑 椭 圆 度 : {ellipticity:.3f}")
print("="*45 + "\n")

# ==========================================
# 6. 可视化输出与保存
# ==========================================
fig = plt.figure(figsize=(14, 9))
# 调整布局比例，留出底部空间
gs = GridSpec(2, 2, figure=fig, width_ratios=[1, 1], height_ratios=[1, 0.6], wspace=0.3, hspace=0.3)

lim_span = 4 # 视窗半宽 4mm (即画框物理大小为 8x8 mm)

# --- 图1: 输入光斑 ---
ax1 = fig.add_subplot(gs[0, 0])
ax1.imshow(I_in, extent=[-L/2*1e3, L/2*1e3, -L/2*1e3, L/2*1e3], cmap='jet', origin='lower')
ax1.set_xlim([-lim_span, lim_span])
ax1.set_ylim([-lim_span, lim_span])
ax1.set_aspect('equal') # 强制物理比例 1:1
ax1.set_title("Input Gaussian Beam ($z = 0$ m)")
ax1.set_xlabel("$x$ (mm)")
ax1.set_ylabel("$y$ (mm)")

# --- 图2: 远场畸变偏转光斑 ---
ax2 = fig.add_subplot(gs[0, 1])
ax2.imshow(I_out, extent=[-L/2*1e3, L/2*1e3, -L/2*1e3, L/2*1e3], cmap='jet', origin='lower')
ax2.set_xlim([-lim_span, lim_span])
# 视窗“镜头”跟随光束移动，依然保持 8x8 mm 的视区
y_center_mm = simulated_shift * 1e3
ax2.set_ylim([y_center_mm - lim_span, y_center_mm + lim_span])
ax2.set_aspect('equal') # 强制物理比例 1:1
ax2.set_title(f"Deflected & Distorted Beam ($z = 1.7$ m)")
ax2.set_xlabel("$x$ (mm)")
ax2.set_ylabel("$y$ (mm)")

# --- 将数据文本框嵌入图2右上角外部 ---
stats_text = (
    f"--- Simulation Data ---\n"
    f"Input $w_0$: {w0*1e3:.2f} mm\n"
    f"Deflect Angle: {simulated_theta*1e3:.2f} mrad\n"
    f"Beam Shift: {simulated_shift*1e3:.2f} mm\n"
    f"Output $w_x$: {wx_out*1e3:.2f} mm\n"
    f"Output $w_y$: {wy_out*1e3:.2f} mm\n"
    f"Ellipticity: {ellipticity:.3f}"
)
# 将文本框放置在两幅图像之间的右上角或右侧空白处
fig.text(0.85, 0.75, stats_text, fontsize=13, va='center', ha='left',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#f9f9f9', edgecolor='gray', alpha=0.9))

# --- 图3: 输入Y方向剖面 ---
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(y*1e3, profile_y_in, 'k-', linewidth=1.5, label='Input Profile ($y$)')
ax3.axhline(threshold, color='r', linestyle=':', label='$1/e^2$ threshold')
ax3.set_xlim([-lim_span, lim_span])
ax3.set_ylim([-0.05, 1.05])
ax3.set_title("Input Intensity Profile")
ax3.set_xlabel("$y$ Position (mm)")
ax3.set_ylabel("Normalized Intensity")
ax3.legend()
ax3.grid(True, alpha=0.5)

# --- 图4: 远场Y方向剖面 ---
ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(y*1e3, profile_y_out, 'b-', linewidth=1.5, label='Deflected Profile ($y$)')
ax4.axvline(expected_shift*1e3, color='g', linestyle='--', label='Theoretical Center')
ax4.axhline(threshold, color='r', linestyle=':', label='$1/e^2$ threshold')
# 纵坐标范围设置得宽一点，以展现完整的偏移
ax4.set_xlim([-2, y_center_mm + 5]) 
ax4.set_ylim([-0.05, 1.05])
ax4.set_title("Output Intensity Profile (Shifted & Broadened)")
ax4.set_xlabel("$y$ Position (mm)")
ax4.legend(loc='upper left')
ax4.grid(True, alpha=0.5)

# 自动调整边距以适应文本框
plt.subplots_adjust(left=0.08, right=0.88, top=0.92, bottom=0.08)

# ==========================================
# 自动保存到当前脚本所在文件夹
# ==========================================
# 获取当前脚本所在绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
save_path = os.path.join(current_dir, "AOD_Simulation_clinder_lensing_Result.png")

plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"图像已成功保存至:\n{save_path}")

plt.show()