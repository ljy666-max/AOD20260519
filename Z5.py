import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ==========================================
# 1. 全局学术审美与物理参数配置
# ==========================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 11

# 物理常量
wl = 355e-9           # 激光波长 (m)
w0 = 1.35e-3          # 激光 1/e^2 束腰半径 (m)
V = 5750.0            # 石英晶体声速 (m/s)
z_prop = 1.7          # 远场传播距离 (m)
k = 2 * np.pi / wl    # 光学波数

# 核心像差强度设计 (设定束腰边缘相位深度为 3.0 rad)
B = 3.0 / (w0**2)     

# 反算驱动级真实的射频扫频斜率 alpha (Hz/s)
alpha_45 = (B * V**2) / np.pi
print(f"--- 射频驱动时域控制参数 ---")
print(f"45度 AOD 所需正扫频斜率:  {alpha_45*1e-12:.3f} MHz/us")
print(f"0度/90度 AOD 所需负扫频斜率: {-alpha_45*1e-12/2:.3f} MHz/us\n")

# ==========================================
# 2. 空间空间坐标与多轴投影初始化
# ==========================================
L = 30e-3             # 计算网格宽度 30mm
N = 1024              # 采样点数
dx = L / N
x = np.linspace(-L/2, L/2, N)
y = np.linspace(-L/2, L/2, N)
X, Y = np.meshgrid(x, y)

# 【核心物理】构建 45° 旋转 AOD 晶体轴的空间投影坐标 U
U = (np.sqrt(2) / 2) * (X + Y)

# 输入高斯光场
E_in = np.exp(-(X**2 + Y**2) / (w0**2))
I_in = np.abs(E_in)**2

# ==========================================
# 3. 三轴 AOD 相位独立雕刻与级联叠加
# ==========================================
# AOD 1 (0度, X轴): 产生负二次项，消去后续产生的 X^2 
phi_AOD1 = -0.5 * B * X**2

# AOD 2 (90度, Y轴): 产生负二次项，消去后续产生的 Y^2
phi_AOD2 = -0.5 * B * Y**2

# AOD 3 (45度倾斜): 产生关于 U 轴的一维二次项
phi_AOD3 = B * U**2

# 三轴连续级联，相位在物理空间上非相干加和
phi_total = phi_AOD1 + phi_AOD2 + phi_AOD3

# ==========================================
# 4. 波动光学传播引擎 (ASM)
# ==========================================
def angular_spectrum_propagate(E0, wl, z, dx):
    Ny, Nx = E0.shape
    fx = np.fft.fftfreq(Nx, d=dx)
    fy = np.fft.fftfreq(Ny, d=dx)
    FX, FY = np.meshgrid(fx, fy)
    
    # 传递函数计算
    term = np.maximum(1.0 - (wl * FX)**2 - (wl * FY)**2, 0)
    H = np.exp(1j * k * z * np.sqrt(term))
    
    # 傅里叶变换到频域，滤波后再转回空域
    return np.fft.ifft2(np.fft.fft2(E0) * H)

# 调制并执行传播
E_mod = E_in * np.exp(1j * phi_total)
E_out = angular_spectrum_propagate(E_mod, wl, z_prop, dx)

I_out = np.abs(E_out)**2
I_out /= np.max(I_out) # 归一化

# ==========================================
# 5. 高颜值学术级图表可视化展现实录
# ==========================================
fig = plt.figure(figsize=(15, 8), facecolor='#fdfdfd')
gs = GridSpec(2, 3, wspace=0.3, hspace=0.3)
lim_span = 4 # 视窗画框范围 4mm

# 限制显示掩码，只看激光核心能量区域的相位，画面更干净
mask = (X**2 + Y**2) < (2*w0)**2
extent_mm = [-L/2*1e3, L/2*1e3, -L/2*1e3, L/2*1e3]

# --- 子图 1: AOD 1 相位 (0°) ---
ax1 = fig.add_subplot(gs[0, 0])
im1 = ax1.imshow(np.where(mask, phi_AOD1, np.nan), extent=extent_mm, cmap='coolwarm', origin='lower')
ax1.set_xlim([-lim_span, lim_span]); ax1.set_ylim([-lim_span, lim_span])
ax1.set_title("AOD 1 Phase (0°, $-x^2$ Lens Effect)")
fig.colorbar(im1, ax=ax1, label="Phase (rad)")

# --- 子图 2: AOD 2 相位 (90°) ---
ax2 = fig.add_subplot(gs[0, 1])
im2 = ax2.imshow(np.where(mask, phi_AOD2, np.nan), extent=extent_mm, cmap='coolwarm', origin='lower')
ax2.set_xlim([-lim_span, lim_span]); ax2.set_ylim([-lim_span, lim_span])
ax2.set_title("AOD 2 Phase (90°, $-y^2$ Lens Effect)")
fig.colorbar(im2, ax=ax2, label="Phase (rad)")

# --- 子图 3: AOD 3 相位 (45°) ---
ax3 = fig.add_subplot(gs[0, 2])
im3 = ax3.imshow(np.where(mask, phi_AOD3, np.nan), extent=extent_mm, cmap='coolwarm', origin='lower')
ax3.set_xlim([-lim_span, lim_span]); ax3.set_ylim([-lim_span, lim_span])
ax3.set_title("AOD 3 Phase (45° Rotated, $+u^2$)")
fig.colorbar(im3, ax=ax3, label="Phase (rad)")

# --- 子图 4: 叠加后的终极合成相位 (完美的 xy 项) ---
ax4 = fig.add_subplot(gs[1, 0])
im4 = ax4.imshow(np.where(mask, phi_total, np.nan), extent=extent_mm, cmap='twilight_shifted', origin='lower')
ax4.set_xlim([-lim_span, lim_span]); ax4.set_ylim([-lim_span, lim_span])
ax4.set_title(r"Synthesized Total Phase ($\Phi = B \cdot xy$)")
ax4.set_xlabel("x (mm)"); ax4.set_ylabel("y (mm)")
fig.colorbar(im4, ax=ax4, label="Phase (rad)")

# --- 子图 5: 最终远场相差光斑 (物理证据) ---
ax5 = fig.add_subplot(gs[1, 1])
im5 = ax5.imshow(I_out, extent=extent_mm, cmap='jet', origin='lower')
ax5.set_xlim([-lim_span, lim_span]); ax5.set_ylim([-lim_span, lim_span])
ax5.set_title(f"Resulting Far-Field Beam (z={z_prop}m)")
ax5.set_xlabel("x (mm)")
fig.colorbar(im5, ax=ax5, label="Normalized Intensity")

# --- 子图 6: 45° 对角线切片分析 (验证像散拉伸) ---
ax6 = fig.add_subplot(gs[1, 2])
# 提取对角线和反对角线剖面
diag_prof1 = np.diag(I_out)
diag_prof2 = np.diag(np.fliplr(I_out))
diag_axis = np.linspace(-L/2*np.sqrt(2)*1e3, L/2*np.sqrt(2)*1e3, N)

ax6.plot(diag_axis, diag_prof1, 'r-', linewidth=1.5, label='45° Diagonal Profile')
ax6.plot(diag_axis, diag_prof2, 'b--', linewidth=1.5, label='135° Diagonal Profile')
ax6.set_xlim([-lim_span*1.4, lim_span*1.4])
ax6.set_title("Diagonal Profile Intensity Analysis")
ax6.set_xlabel("Diagonal Distance (mm)")
ax6.grid(True, linestyle=':', alpha=0.6)
ax6.legend(loc='upper right')

plt.suptitle(r"Cascade 3-AOD Wavefront Shaping: Successful Generation of Pure $xy$ Cross-Term", fontsize=15, fontweight='bold', y=0.98)
plt.show()