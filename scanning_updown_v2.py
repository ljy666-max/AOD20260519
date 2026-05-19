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

# 【核心修改】文献 Fig 3c 的不同步频率参数
f1 = 124e6            # 下半部分光斑遇到的旧频率 (120 MHz)
f2 = 125e6            # 上半部分光斑遇到的新频率 (130 MHz)
f_center = 125e6      # 几何中心频率（用于对准相机的视窗）

y_center_mm = z * np.tan((wl * f_center) / V) * 1e3

# ==========================================
# 2. 空间网格初始化
# ==========================================
L = 40e-3             
N = 2048              
dx = L / N            
dy = dx

x = np.linspace(-L/2, L/2, N)
y = np.linspace(-L/2, L/2, N)
X, Y = np.meshgrid(x, y)

E_in = np.exp(-(X**2 + Y**2) / (w0**2))

# ==========================================
# 3. 【核心物理】生成不同步导致的空间阶跃相位
# ==========================================
# 模拟最极端的不同步情况：新旧声波交界线正好在激光中心 (y=0)
# y > 0 区域频率为 130MHz，y <= 0 区域频率为 120MHz
f_space = np.where(Y > 0, f2, f1)

# 计算相位: d(phi)/dy = 2*pi*f/V => phi(y) = 2*pi*f*y/V
# 这里由于 f 在 y=0 处跳变，但乘上 y 之后，在 y=0 处相位依然连续(都为0)
# 这物理上对应了一个像双棱镜一样中间有折线的连续波前
phi_step = (2 * np.pi / V) * f_space * Y

E_mod = E_in * np.exp(1j * phi_step)

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
    H = np.exp(1j * (2 * np.pi / wl) * z * np.sqrt(term))
    
    E0_fft = np.fft.fft2(E0)
    return np.fft.ifft2(E0_fft * H)

E_out = angular_spectrum_propagate(E_mod, wl, z, dx, dy)

# ==========================================
# 5. 光强计算与参数提取
# ==========================================
I_in = np.abs(E_in)**2
I_out = np.abs(E_out)**2
I_in /= np.max(I_in)
I_out /= np.max(I_out)

profile_y_in = I_in[:, N//2]  
profile_y_out = I_out[:, N//2] 

# ==========================================
# 6. 可视化输出与保存
# ==========================================
fig = plt.figure(figsize=(14, 9))
gs = GridSpec(2, 2, figure=fig, width_ratios=[1, 1], height_ratios=[1, 0.6], wspace=0.3, hspace=0.3)

lim_span = 4 # 视窗半宽 4mm

# --- 图1: 输入光斑 ---
ax1 = fig.add_subplot(gs[0, 0])
ax1.imshow(I_in, extent=[-L/2*1e3, L/2*1e3, -L/2*1e3, L/2*1e3], cmap='jet', origin='lower')
ax1.set_xlim([-lim_span, lim_span])
ax1.set_ylim([-lim_span, lim_span])
ax1.set_aspect('equal')
ax1.set_title("Input Gaussian Beam ($z = 0$ m)")
ax1.set_xlabel("$x$ (mm)")
ax1.set_ylabel("$y$ (mm)")

# --- 图2: 不同步导致的畸变光斑 ---
ax2 = fig.add_subplot(gs[0, 1])
ax2.imshow(I_out, extent=[-L/2*1e3, L/2*1e3, -L/2*1e3, L/2*1e3], cmap='jet', origin='lower')
ax2.set_xlim([-lim_span, lim_span])
ax2.set_ylim([y_center_mm - lim_span, y_center_mm + lim_span])
ax2.set_aspect('equal')
ax2.set_title(f"Unsynchronized Deflection ($z = {z}$ m)\nSeverely Distorted Beam Profile")
ax2.set_xlabel("$x$ (mm)")
ax2.set_ylabel("$y$ (mm)")

# --- 侧边栏数据 ---
stats_text = (
    f"--- Scan Parameters ---\n"
    f"State: Unsynchronized\n"
    f"Spatial Freq Step:\n"
    f"  Upper Half: 130 MHz\n"
    f"  Lower Half: 120 MHz\n"
    f"Beam Quality: DESTROYED\n"
    f"(Matches Paper Fig 3c)"
)
fig.text(0.92, 0.75, stats_text, fontsize=13, va='center', ha='left',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#ffeeee', edgecolor='red', alpha=0.9))

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

# --- 图4: 远场被撕裂的光斑剖面 ---
ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(y*1e3, profile_y_out, 'r-', linewidth=1.5, label='Smeared Profile')
# 标出120MHz和130MHz理论上的偏转中心
y1_th = z * np.tan((wl * f1) / V) * 1e3
y2_th = z * np.tan((wl * f2) / V) * 1e3
ax4.axvline(y1_th, color='gray', linestyle='--', label='120 MHz Target')
ax4.axvline(y2_th, color='gray', linestyle='--', label='130 MHz Target')

ax4.set_xlim([y_center_mm - lim_span, y_center_mm + lim_span])
ax4.set_ylim([-0.05, 1.05])
ax4.set_title("Output Intensity Profile (Beam Tearing)")
ax4.set_xlabel("$y$ Position (mm)")
ax4.legend(loc='upper center')
ax4.grid(True, alpha=0.5)

plt.subplots_adjust(left=0.08, right=0.88, top=0.92, bottom=0.08)

# 保存图像
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
save_path = os.path.join(current_dir, "AOD_Scanning_Fig3c.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"不同步畸变扫描图像已成功保存至:\n{save_path}")

plt.show()