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

# 物理常量
wl = 355e-9           
w0 = 1.35e-3          
z_prop = 1.7          
k = 2 * np.pi / wl    

L = 20e-3             # 计算网格 20mm
N = 1024              
dx = L / N
x = np.linspace(-L/2, L/2, N)
y = np.linspace(-L/2, L/2, N)
X, Y = np.meshgrid(x, y)

# 【核心】像差拟合必须使用基于束腰的归一化坐标！
Xn = X / w0
Yn = Y / w0

# 建立 45° 和 -45° 的投影坐标
U = (np.sqrt(2) / 2) * (Xn + Yn)  # +45° 轴
V = (np.sqrt(2) / 2) * (Xn - Yn)  # -45° 轴

# 设定一个全局控制强度 A，用于控制像差严重程度 (避免严重混叠)
A = 1.5 

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
# 2. 生成 Z8 (Coma X) 的三轴级联相位
# ==========================================
# 公式: 4u^3 + 4v^3 + (4*sqrt(2)x^3 - 4*sqrt(2)x)
phi_Z8_AOD1 = A * 4 * U**3
phi_Z8_AOD2 = A * 4 * V**3
phi_Z8_AOD3 = A * (4 * np.sqrt(2) * Xn**3 - 4 * np.sqrt(2) * Xn)

phi_Z8_total = phi_Z8_AOD1 + phi_Z8_AOD2 + phi_Z8_AOD3

E_out_Z8 = angular_spectrum_propagate(E_in * np.exp(1j * phi_Z8_total), wl, z_prop, dx)
I_out_Z8 = np.abs(E_out_Z8)**2
I_out_Z8 /= np.max(I_out_Z8)

# ==========================================
# 3. 生成 Z7 (Coma Y) 的三轴级联相位
# ==========================================
# 公式: 4u^3 - 4v^3 + (4*sqrt(2)y^3 - 4*sqrt(2)y)
phi_Z7_AOD1 = A * 4 * U**3
phi_Z7_AOD2 = -A * 4 * V**3   # 注意这里的负号！
phi_Z7_AOD3 = A * (4 * np.sqrt(2) * Yn**3 - 4 * np.sqrt(2) * Yn)

phi_Z7_total = phi_Z7_AOD1 + phi_Z7_AOD2 + phi_Z7_AOD3

E_out_Z7 = angular_spectrum_propagate(E_in * np.exp(1j * phi_Z7_total), wl, z_prop, dx)
I_out_Z7 = np.abs(E_out_Z7)**2
I_out_Z7 /= np.max(I_out_Z7)

# ==========================================
# 4. 图表可视化 (全景拼图)
# ==========================================
fig = plt.figure(figsize=(16, 8), facecolor='#fdfdfd')
gs = GridSpec(2, 4, wspace=0.3, hspace=0.3)
lim_span = 4
extent_mm = [-L/2*1e3, L/2*1e3, -L/2*1e3, L/2*1e3]
mask = (Xn**2 + Yn**2) < 4  # 限制在2倍束腰内显示相位

def plot_phase(ax, phi, title):
    im = ax.imshow(np.where(mask, phi, np.nan), extent=extent_mm, cmap='coolwarm', origin='lower', vmin=-15, vmax=15)
    ax.set_xlim([-lim_span, lim_span]); ax.set_ylim([-lim_span, lim_span])
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    return im

def plot_intensity(ax, I_out, title):
    im = ax.imshow(I_out, extent=extent_mm, cmap='jet', origin='lower')
    ax.set_xlim([-lim_span, lim_span]); ax.set_ylim([-lim_span, lim_span])
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    return im

# --- Z8 (Coma X) 绘图 ---
plot_phase(fig.add_subplot(gs[0, 0]), phi_Z8_AOD1, r"$\text{AOD}_1(+45^\circ): 4u^3$")
plot_phase(fig.add_subplot(gs[0, 1]), phi_Z8_AOD2, r"$\text{AOD}_2(-45^\circ): 4v^3$")
plot_phase(fig.add_subplot(gs[0, 2]), phi_Z8_AOD3, r"$\text{AOD}_3(0^\circ): 4\sqrt{2}x^3 - 4\sqrt{2}x$")
im_z8 = plot_intensity(fig.add_subplot(gs[0, 3]), I_out_Z8, r"Resulting $Z_8$ (Coma X)")

# --- Z7 (Coma Y) 绘图 ---
plot_phase(fig.add_subplot(gs[1, 0]), phi_Z7_AOD1, r"$\text{AOD}_1(+45^\circ): 4u^3$")
plot_phase(fig.add_subplot(gs[1, 1]), phi_Z7_AOD2, r"$\text{AOD}_2(-45^\circ): -4v^3$")
plot_phase(fig.add_subplot(gs[1, 2]), phi_Z7_AOD3, r"$\text{AOD}_3(90^\circ): 4\sqrt{2}y^3 - 4\sqrt{2}y$")
im_z7 = plot_intensity(fig.add_subplot(gs[1, 3]), I_out_Z7, r"Resulting $Z_7$ (Coma Y)")

plt.suptitle(r"Cascade 3-AOD Wavefront Shaping: 3rd-Order Coma ($Z_7$ & $Z_8$)", fontsize=16, fontweight='bold', y=0.98)
plt.show()