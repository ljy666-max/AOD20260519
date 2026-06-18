import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ==========================================
# 1. 全局设置与物理网格
# ==========================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 11

wl = 355e-9           
w0 = 1.35e-3          
z_prop = 1.7          
k = 2 * np.pi / wl    

L = 10e-3             # 视窗 10mm
N = 1024              
dx = L / N
x = np.linspace(-L/2, L/2, N)
y = np.linspace(-L/2, L/2, N)
X, Y = np.meshgrid(x, y)

Xn = X / w0
Yn = Y / w0
U = (np.sqrt(2) / 2) * (Xn + Yn) 
V = (np.sqrt(2) / 2) * (Xn - Yn) 

# 控制像差深度 A (Z13系数很大，A设小一点防极度混叠)
A = 0.4

# 圆形光阑
pupil_mask = (Xn**2 + Yn**2) <= 1.0
E_in = np.zeros_like(X)
E_in[pupil_mask] = 1.0

def angular_spectrum_propagate(E0, wl, z, dx):
    Ny, Nx = E0.shape
    fx = np.fft.fftfreq(Nx, d=dx)
    fy = np.fft.fftfreq(Ny, d=dx)
    FX, FY = np.meshgrid(fx, fy)
    term = np.maximum(1.0 - (wl * FX)**2 - (wl * FY)**2, 0)
    H = np.exp(1j * k * z * np.sqrt(term))
    return np.fft.ifft2(np.fft.fft2(E0) * H)

# ==========================================
# 2. 生成 Z13 (Secondary Astigmatism 45°) 
# ==========================================
c = np.sqrt(10)

# 按照推导的 2-AOD 极简配方
phi_AOD1 = A * (4 * c * U**4 - 3 * c * U**2)
phi_AOD2 = A * (-4 * c * V**4 + 3 * c * V**2)

# 合成总相位
phi_total = phi_AOD1 + phi_AOD2

# 理论标准 Z13 公式验证
phi_ideal = A * (8*c*(Xn**3)*Yn + 8*c*Xn*(Yn**3) - 6*c*Xn*Yn)

# 衍射传播
E_out = angular_spectrum_propagate(E_in * np.exp(1j * phi_total), wl, z_prop, dx)
I_out = np.abs(E_out)**2
I_out /= np.max(I_out)

# ==========================================
# 3. 可视化绘图
# ==========================================
fig = plt.figure(figsize=(15, 8), facecolor='#fdfdfd')
gs = GridSpec(2, 3, wspace=0.3, hspace=0.3)
lim_span = 2
extent_mm = [-L/2*1e3, L/2*1e3, -L/2*1e3, L/2*1e3]

def plot_phase(ax, phi, title):
    phi_plot = np.where(pupil_mask, phi, np.nan)
    im = ax.imshow(phi_plot, extent=extent_mm, cmap='twilight_shifted', origin='lower')
    ax.set_xlim([-lim_span, lim_span]); ax.set_ylim([-lim_span, lim_span])
    ax.set_title(title, fontsize=12)
    ax.set_xticks([]); ax.set_yticks([])
    circle = plt.Circle((0, 0), w0*1e3, color='black', fill=False, linestyle=':', alpha=0.5)
    ax.add_patch(circle)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return im

def plot_intensity(ax, I_out, title):
    im = ax.imshow(I_out, extent=extent_mm, cmap='jet', origin='lower')
    ax.set_xlim([-lim_span, lim_span]); ax.set_ylim([-lim_span, lim_span])
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return im

# --- 上排：AOD 相位 ---
plot_phase(fig.add_subplot(gs[0, 0]), phi_AOD1, r"$\text{AOD}_1 (+45^\circ): 4\sqrt{10}u^4 - 3\sqrt{10}u^2$")
plot_phase(fig.add_subplot(gs[0, 1]), phi_AOD2, r"$\text{AOD}_2 (-45^\circ): -4\sqrt{10}v^4 + 3\sqrt{10}v^2$")

# --- 右侧大图：远场结果 ---
ax_farfield = fig.add_subplot(gs[:, 2]) # 占据右侧整列
plot_intensity(ax_farfield, I_out, r"Far-Field Intensity ($Z_{13}$ Sec. Astigmatism)")

# --- 下排：理论验证 ---
plot_phase(fig.add_subplot(gs[1, 0]), phi_total, r"Synthesized Total Phase ($\Phi_{1} + \Phi_{2}$)")
plot_phase(fig.add_subplot(gs[1, 1]), phi_ideal, r"Ideal $Z_{13}$ Math Formula")

plt.suptitle(r"Elegant 2-AOD Wavefront Shaping: $4^{th}$-Order Secondary Astigmatism ($Z_{13}$)", 
             fontsize=16, fontweight='bold', y=0.98)
plt.show()