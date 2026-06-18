import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ==========================================
# 1. 全局设置与归一化物理网格
# ==========================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 11

wl = 355e-9           
w0 = 1.35e-3          
z_prop = 1.7          
k = 2 * np.pi / wl    

L = 20e-3             
N = 1024              
dx = L / N
x = np.linspace(-L/2, L/2, N)
y = np.linspace(-L/2, L/2, N)
X, Y = np.meshgrid(x, y)

Xn = X / w0
Yn = Y / w0
U = (np.sqrt(2) / 2) * (Xn + Yn) 
V = (np.sqrt(2) / 2) * (Xn - Yn) 

# 控制像差深度 A (Trefoil的远场极易发散，这里调小一点保证图案清晰)
A = 0.8 
E_in = np.exp(-(Xn**2 + Yn**2))

def angular_spectrum_propagate(E0, wl, z, dx):
    Ny, Nx = E0.shape
    fx = np.fft.fftfreq(Nx, d=dx)
    fy = np.fft.fftfreq(Ny, d=dx)
    FX, FY = np.meshgrid(fx, fy)
    term = np.maximum(1.0 - (wl * FX)**2 - (wl * FY)**2, 0)
    H = np.exp(1j * k * z * np.sqrt(term))
    return np.fft.ifft2(np.fft.fft2(E0) * H)

# ==========================================
# 2. 生成 Z10 (Trefoil 30°) 的三轴级联相位
# ==========================================
# 配方: -4u^3 - 4v^3 + 4*sqrt(2)x^3
phi_Z10_AOD1 = -A * 4 * U**3
phi_Z10_AOD2 = -A * 4 * V**3
phi_Z10_AOD3 = A * 4 * np.sqrt(2) * Xn**3

phi_Z10_total = phi_Z10_AOD1 + phi_Z10_AOD2 + phi_Z10_AOD3
E_out_Z10 = angular_spectrum_propagate(E_in * np.exp(1j * phi_Z10_total), wl, z_prop, dx)
I_out_Z10 = np.abs(E_out_Z10)**2
I_out_Z10 /= np.max(I_out_Z10)

# ==========================================
# 3. 生成 Z9 (Trefoil 0°) 的三轴级联相位
# ==========================================
# 配方: 4u^3 - 4v^3 - 4*sqrt(2)y^3
phi_Z9_AOD1 = A * 4 * U**3
phi_Z9_AOD2 = -A * 4 * V**3  
phi_Z9_AOD3 = -A * 4 * np.sqrt(2) * Yn**3

phi_Z9_total = phi_Z9_AOD1 + phi_Z9_AOD2 + phi_Z9_AOD3
E_out_Z9 = angular_spectrum_propagate(E_in * np.exp(1j * phi_Z9_total), wl, z_prop, dx)
I_out_Z9 = np.abs(E_out_Z9)**2
I_out_Z9 /= np.max(I_out_Z9)

# ==========================================
# 4. 图表可视化 
# ==========================================
fig = plt.figure(figsize=(16, 8), facecolor='#fdfdfd')
gs = GridSpec(2, 4, wspace=0.3, hspace=0.3)
lim_span = 4
extent_mm = [-L/2*1e3, L/2*1e3, -L/2*1e3, L/2*1e3]
mask = (Xn**2 + Yn**2) < 4  

def plot_phase(ax, phi, title):
    im = ax.imshow(np.where(mask, phi, np.nan), extent=extent_mm, cmap='coolwarm', origin='lower', vmin=-10, vmax=10)
    ax.set_xlim([-lim_span, lim_span]); ax.set_ylim([-lim_span, lim_span])
    ax.set_title(title, fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    return im

def plot_intensity(ax, I_out, title):
    im = ax.imshow(I_out, extent=extent_mm, cmap='jet', origin='lower')
    ax.set_xlim([-lim_span, lim_span]); ax.set_ylim([-lim_span, lim_span])
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    return im

# --- Z10 (Trefoil 30°) ---
plot_phase(fig.add_subplot(gs[0, 0]), phi_Z10_AOD1, r"$\text{AOD}_1(+45^\circ): -4u^3$")
plot_phase(fig.add_subplot(gs[0, 1]), phi_Z10_AOD2, r"$\text{AOD}_2(-45^\circ): -4v^3$")
plot_phase(fig.add_subplot(gs[0, 2]), phi_Z10_AOD3, r"$\text{AOD}_3(0^\circ): 4\sqrt{2}x^3$")
plot_intensity(fig.add_subplot(gs[0, 3]), I_out_Z10, r"Resulting $Z_{10}$ (Trefoil 30°)")

# --- Z9 (Trefoil 0°) ---
plot_phase(fig.add_subplot(gs[1, 0]), phi_Z9_AOD1, r"$\text{AOD}_1(+45^\circ): 4u^3$")
plot_phase(fig.add_subplot(gs[1, 1]), phi_Z9_AOD2, r"$\text{AOD}_2(-45^\circ): -4v^3$")
plot_phase(fig.add_subplot(gs[1, 2]), phi_Z9_AOD3, r"$\text{AOD}_3(90^\circ): -4\sqrt{2}y^3$")
plot_intensity(fig.add_subplot(gs[1, 3]), I_out_Z9, r"Resulting $Z_9$ (Trefoil 0°)")

plt.suptitle(r"Cascade 3-AOD Wavefront Shaping: Trefoil Aberrations ($Z_9$ & $Z_{10}$)", fontsize=16, fontweight='bold', y=0.98)
plt.show()