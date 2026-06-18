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

L = 10e-3             # 计算网格 10mm (放大中心细节)
N = 1024              
dx = L / N
x = np.linspace(-L/2, L/2, N)
y = np.linspace(-L/2, L/2, N)
X, Y = np.meshgrid(x, y)

Xn = X / w0
Yn = Y / w0
U = (np.sqrt(2) / 2) * (Xn + Yn) 
V = (np.sqrt(2) / 2) * (Xn - Yn) 

# 控制像差深度 A 
A = 0.5 

# 定义圆形光阑 (Pupil)
pupil_mask = (Xn**2 + Yn**2) <= 1.0
E_in = np.zeros_like(X)
E_in[pupil_mask] = 1.0 # 均匀平顶光穿过圆形光阑

def angular_spectrum_propagate(E0, wl, z, dx):
    Ny, Nx = E0.shape
    fx = np.fft.fftfreq(Nx, d=dx)
    fy = np.fft.fftfreq(Ny, d=dx)
    FX, FY = np.meshgrid(fx, fy)
    term = np.maximum(1.0 - (wl * FX)**2 - (wl * FY)**2, 0)
    H = np.exp(1j * k * z * np.sqrt(term))
    return np.fft.ifft2(np.fft.fft2(E0) * H)

# ==========================================
# 2. 生成 Z11 (Spherical Aberration) 的四轴级联相位
# ==========================================
c = np.sqrt(5)

# 按照推导的 4-AOD 配方
phi_AOD1 = A * 4 * c * U**4
phi_AOD2 = A * 4 * c * V**4
phi_AOD3 = A * (4 * c * Xn**4 - 6 * c * Xn**2 + c/2)
phi_AOD4 = A * (4 * c * Yn**4 - 6 * c * Yn**2 + c/2)

# 对角线 AOD 叠加结果 (用于中间过程展示)
phi_diag_sum = phi_AOD1 + phi_AOD2

# 终极叠加总相位
phi_total = phi_AOD1 + phi_AOD2 + phi_AOD3 + phi_AOD4

# 理论标准的 Z11 相位 (用于对比)
phi_ideal = A * (12*c*(Xn**2)*(Yn**2) + 6*c*Xn**4 + 6*c*Yn**4 - 6*c*Xn**2 - 6*c*Yn**2 + c)

# 衍射传播
E_out = angular_spectrum_propagate(E_in * np.exp(1j * phi_total), wl, z_prop, dx)
I_out = np.abs(E_out)**2
I_out /= np.max(I_out)

# ==========================================
# 3. 绘图：四轴拼图与球差重生
# ==========================================
fig = plt.figure(figsize=(16, 8), facecolor='#fdfdfd')
gs = GridSpec(2, 4, wspace=0.3, hspace=0.3)
lim_span = 2
extent_mm = [-L/2*1e3, L/2*1e3, -L/2*1e3, L/2*1e3]

def plot_phase(ax, phi, title):
    # 只在光阑内显示相位，外围透明
    phi_plot = np.where(pupil_mask, phi, np.nan)
    im = ax.imshow(phi_plot, extent=extent_mm, cmap='coolwarm', origin='lower')
    ax.set_xlim([-lim_span, lim_span]); ax.set_ylim([-lim_span, lim_span])
    ax.set_title(title, fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    # 画一个光阑边界线
    circle = plt.Circle((0, 0), w0*1e3, color='black', fill=False, linestyle=':', alpha=0.5)
    ax.add_patch(circle)
    return im

def plot_intensity(ax, I_out, title):
    im = ax.imshow(I_out, extent=extent_mm, cmap='jet', origin='lower')
    ax.set_xlim([-lim_span, lim_span]); ax.set_ylim([-lim_span, lim_span])
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    return im

# --- 第一排：四个独立的 AOD 相位 ---
plot_phase(fig.add_subplot(gs[0, 0]), phi_AOD1, r"$\text{AOD}_1 (+45^\circ): 4\sqrt{5}u^4$")
plot_phase(fig.add_subplot(gs[0, 1]), phi_AOD2, r"$\text{AOD}_2 (-45^\circ): 4\sqrt{5}v^4$")
plot_phase(fig.add_subplot(gs[0, 2]), phi_AOD3, r"$\text{AOD}_3 (0^\circ, X): 4\sqrt{5}x^4 - 6\sqrt{5}x^2$")
plot_phase(fig.add_subplot(gs[0, 3]), phi_AOD4, r"$\text{AOD}_4 (90^\circ, Y): 4\sqrt{5}y^4 - 6\sqrt{5}y^2$")

# --- 第二排：合成过程与远场结果 ---
plot_phase(fig.add_subplot(gs[1, 0]), phi_diag_sum, r"Diagonal Sum (Cross-term Engine)")
plot_phase(fig.add_subplot(gs[1, 1]), phi_total, r"Synthesized Total Phase ($\Phi_{Total}$)")
plot_phase(fig.add_subplot(gs[1, 2]), phi_ideal, r"Ideal $Z_{11}$ Math Formula (Verification)")
plot_intensity(fig.add_subplot(gs[1, 3]), I_out, r"Far-Field Intensity ($Z_{11}$ Sphere)")

plt.suptitle(r"Conquering 4th-Order Spherical Aberration ($Z_{11}$) via 4-Axis AOD Cascade", fontsize=16, fontweight='bold', y=0.98)
plt.show()