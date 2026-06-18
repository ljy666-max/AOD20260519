import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ==========================================
# 1. 全局设置与物理网格
# ==========================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 10

wl = 355e-9           
w0 = 1.35e-3          
z_prop = 1.7          
k = 2 * np.pi / wl    

L = 12e-3             # 视窗大小
N = 1024*1              
dx = L / N
x = np.linspace(-L/2, L/2, N)
y = np.linspace(-L/2, L/2, N)
X, Y = np.meshgrid(x, y)

Xn = X / w0
Yn = Y / w0
U = (np.sqrt(2) / 2) * (Xn + Yn) 
V = (np.sqrt(2) / 2) * (Xn - Yn) 

# 控制像差深度 A (Z14是四阶像差，A过大会导致散斑混叠，0.5是个不错的展示值)
A = 0.3 
# 这种波的干涉，在中心区域形成了一个非常规则的网格状驻波图案。
# 这在物理学上被称为光学焦散（Optics Caustics）。
# 可以降低像差深度，收缩光束
# A=0.5，硬边缘的图——强畸变下，波动光学的“焦散干涉”主导，形成网格斑点。
# A=0.15四阶像差中心平坦，强切趾效应会导致光束“屏蔽”高阶畸变，退化为圆形。
# A=0.3，标准高斯的图：展示几何光学与波动光学的完美平衡，呈现理想四叶草形貌。

# 恢复平滑的高斯光束包络 (切趾效应)
E_in = np.exp(-(Xn**2 + Yn**2)) 
# 依然保留 pupil_mask 用于画图时抠出相位的圆形区域
pupil_mask = (Xn**2 + Yn**2) <= 1.0

def angular_spectrum_propagate(E0, wl, z, dx):
    Ny, Nx = E0.shape
    fx = np.fft.fftfreq(Nx, d=dx)
    fy = np.fft.fftfreq(Ny, d=dx)
    FX, FY = np.meshgrid(fx, fy)
    term = np.maximum(1.0 - (wl * FX)**2 - (wl * FY)**2, 0)
    H = np.exp(1j * k * z * np.sqrt(term))
    return np.fft.ifft2(np.fft.fft2(E0) * H)

# ==========================================
# 2. 生成 Z14 (Quadrafoil 0°) 的四轴级联相位
# ==========================================
c = np.sqrt(10)

# 按照推导的 4-AOD 配方
phi_AOD1 = -A * 2 * c * U**4
phi_AOD2 = -A * 2 * c * V**4
phi_AOD3 = A * 2 * c * Xn**4
phi_AOD4 = A * 2 * c * Yn**4

# 终极合成相位
phi_total = phi_AOD1 + phi_AOD2 + phi_AOD3 + phi_AOD4

# 理论标准的 Z14 相位验证
phi_ideal = A * (c * Xn**4 + c * Yn**4 - 6 * c * (Xn**2) * (Yn**2))

# 衍射传播
E_out = angular_spectrum_propagate(E_in * np.exp(1j * phi_total), wl, z_prop, dx)
I_out = np.abs(E_out)**2
I_out /= np.max(I_out)

# ==========================================
# 3. 可视化绘图
# ==========================================
fig = plt.figure(figsize=(16, 8), facecolor='#fdfdfd')
gs = GridSpec(2, 4, wspace=0.3, hspace=0.3)
lim_span = 2
extent_mm = [-L/2*1e3, L/2*1e3, -L/2*1e3, L/2*1e3]

def plot_phase(ax, phi, title):
    phi_plot = np.where(pupil_mask, phi, np.nan)
    im = ax.imshow(phi_plot, extent=extent_mm, cmap='coolwarm', origin='lower')
    ax.set_xlim([-lim_span, lim_span]); ax.set_ylim([-lim_span, lim_span])
    ax.set_title(title, fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    circle = plt.Circle((0, 0), w0*1e3, color='black', fill=False, linestyle=':', alpha=0.5)
    ax.add_patch(circle)
    return im

def plot_intensity(ax, I_out, title):
    im = ax.imshow(I_out, extent=extent_mm, cmap='jet', origin='lower')
    ax.set_xlim([-lim_span, lim_span]); ax.set_ylim([-lim_span, lim_span])
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    return im

# --- 第一排：四轴独立 AOD 相位 ---
plot_phase(fig.add_subplot(gs[0, 0]), phi_AOD1, r"$\text{AOD}_1 (+45^\circ): -2\sqrt{10}u^4$")
plot_phase(fig.add_subplot(gs[0, 1]), phi_AOD2, r"$\text{AOD}_2 (-45^\circ): -2\sqrt{10}v^4$")
plot_phase(fig.add_subplot(gs[0, 2]), phi_AOD3, r"$\text{AOD}_3 (0^\circ, X): 2\sqrt{10}x^4$")
plot_phase(fig.add_subplot(gs[0, 3]), phi_AOD4, r"$\text{AOD}_4 (90^\circ, Y): 2\sqrt{10}y^4$")

# --- 第二排：合成结果与远场 ---
plot_phase(fig.add_subplot(gs[1, 0]), phi_AOD1 + phi_AOD2, r"Diagonal Sum (Cross-term Engine)")
plot_phase(fig.add_subplot(gs[1, 1]), phi_total, r"Synthesized Total Phase ($\Phi_{Total}$)")
plot_phase(fig.add_subplot(gs[1, 2]), phi_ideal, r"Ideal $Z_{14}$ Math Formula")
plot_intensity(fig.add_subplot(gs[1, 3]), I_out, r"Far-Field Intensity ($Z_{14}$ Quadrafoil)")

plt.suptitle(r"Conquering 4th-Order Quadrafoil ($Z_{14}$) via 4-Axis AOD Cascade", fontsize=16, fontweight='bold', y=0.98)
plt.show()