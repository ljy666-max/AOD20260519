import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ==========================================
# 全局设置与物理参数
# ==========================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 11

wl = 355e-9           # 波长 (m)
w0 = 1.35e-3          # 束腰半径 (m)
V = 5750.0            # 声速 (m/s)
z_prop = 1.7          # 传播距离 (m)

# 网格参数
L = 30e-3             
N = 4096             
dx = L / N            
dy = dx
x = np.linspace(-L/2, L/2, N)
y = np.linspace(-L/2, L/2, N)
X, Y = np.meshgrid(x, y)

Xn = X / w0
Yn = Y / w0
Rn_sq = Xn**2 + Yn**2

# 【修复 2】：超高斯软光阑 (Super-Gaussian Window)
# 它在中心完全透明，但在 r > 1.8 w_0 时平滑归零，切掉边缘会导致高频混叠的相位
super_gaussian = np.exp(-(Rn_sq / 1.8)**6)
E_in = np.exp(-Rn_sq) * super_gaussian 
k = 2 * np.pi / wl

def angular_spectrum_propagate(E0, wl, z, dx, dy):
    Ny, Nx = E0.shape
    fx = np.fft.fftfreq(Nx, d=dx)
    fy = np.fft.fftfreq(Ny, d=dy)
    FX, FY = np.meshgrid(fx, fy)
    term = np.maximum(1.0 - (wl * FX)**2 - (wl * FY)**2, 0)
    H = np.exp(1j * k * z * np.sqrt(term))
    return np.fft.ifft2(np.fft.fft2(E0) * H)

# ==========================================
# 定义不能完美拟合的像差 (根据文献 Table 2 分解)
# ==========================================
# ideal: 完整的 Zernike 多项式 (用于对比和计算误差)
# aod: AOD 实际能施加的相位 (仅保留 x^n 和 y^n)

# 定义不能完美拟合的像差 (独立设定振幅 A_val 防止高阶混叠)
# ==========================================
modes = {
    'Z7_Coma_Y': {
        'A': -3.0, # 三阶像差，斜率可控
        'ideal': lambda a: a * (6*np.sqrt(2)*Yn*Xn**2 + 6*np.sqrt(2)*Yn**3 - 4*np.sqrt(2)*Yn),
        'aod':   lambda a: a * (6*np.sqrt(2)*Yn**3 - 4*np.sqrt(2)*Yn),
        'desc': "Coma Y"
    },
    'Z8_Coma_X': {
        'A': -3.0,
        'ideal': lambda a: a * (6*np.sqrt(2)*Xn*Yn**2 + 6*np.sqrt(2)*Xn**3 - 4*np.sqrt(2)*Xn),
        'aod':   lambda a: a * (6*np.sqrt(2)*Xn**3 - 4*np.sqrt(2)*Xn),
        'desc': "Coma X"
    },
    'Z9_Trefoil_0': {
        'A': -3.0,
        'ideal': lambda a: a * (6*np.sqrt(2)*Yn*Xn**2 - 2*np.sqrt(2)*Yn**3),
        'aod':   lambda a: a * (-2*np.sqrt(2)*Yn**3),
        'desc': "Trefoil 0°"
    },
    'Z10_Trefoil_30': {
        'A': -3.0,
        'ideal': lambda a: a * (-6*np.sqrt(2)*Xn*Yn**2 + 2*np.sqrt(2)*Xn**3),
        'aod':   lambda a: a * (2*np.sqrt(2)*Xn**3),
        'desc': "Trefoil 30°"
    },
    'Z11_Spherical': {
        'A': -0.25, # 【修复 2】：四阶像差，降低振幅防止相伴斜率突破奈奎斯特极限
        'ideal': lambda a: a * (12*np.sqrt(5)*Xn**2*Yn**2 + 6*np.sqrt(5)*Xn**4 + 6*np.sqrt(5)*Yn**4 - 6*np.sqrt(5)*Xn**2 - 6*np.sqrt(5)*Yn**2 + np.sqrt(5)),
        'aod':   lambda a: a * (6*np.sqrt(5)*Xn**4 + 6*np.sqrt(5)*Yn**4 - 6*np.sqrt(5)*Xn**2 - 6*np.sqrt(5)*Yn**2 + np.sqrt(5)),
        'desc': "Spherical Aberration"
    },
    'Z14_Quadrafoil_0': {
        'A': -0.25, # 【修复 2】：四阶像差
        'ideal': lambda a: a * (np.sqrt(10)*Xn**4 + np.sqrt(10)*Yn**4 - 6*np.sqrt(10)*Xn**2*Yn**2),
        'aod':   lambda a: a * (np.sqrt(10)*Xn**4 + np.sqrt(10)*Yn**4),
        'desc': "Quadrafoil 0°"
    }
}

# ==========================================
# 循环仿真并绘图
# ==========================================
num_modes = len(modes)
fig = plt.figure(figsize=(18, 3.5 * num_modes))
gs = GridSpec(num_modes, 4, wspace=0.3, hspace=0.3)

lim_span = 3      # 光斑显示范围
phase_mask_r = 1.2 # 画相位图时的圆形遮罩半径，避免边缘干扰视觉

# 定义两个范围的掩码
# err_mask_1 对应 r <= 1 (文献中的 [0...1])
err_mask_1 = Rn_sq <= 1.0
# err_mask_2 对应 r <= 1/sqrt(2) (文献中的 [0...2^(-1/2)])，平方后即 0.5
err_mask_2 = Rn_sq <= 0.5

print(f"{'Mode':<20} | {'Err (r<=0.707)':<15} | {'Err (r<=1)':<15}")
print("-" * 55)

for i, (name, params) in enumerate(modes.items()):
    
    # 1. 计算相位与残差
    phi_ideal = params['ideal'](params['A'])
    phi_aod = params['aod'](params['A'])
    phi_res = phi_ideal - phi_aod # 残差 OPD
    
    # 计算 r <= 1 范围内的误差
    max_ideal_1 = np.max(np.abs(phi_ideal[err_mask_1]))
    max_res_1 = np.max(np.abs(phi_res[err_mask_1]))
    rel_error_1 = max_res_1 / max_ideal_1 if max_ideal_1 != 0 else 0
    
    # 计算 r <= 1/sqrt(2) 范围内的误差
    max_ideal_2 = np.max(np.abs(phi_ideal[err_mask_2]))
    max_res_2 = np.max(np.abs(phi_res[err_mask_2]))
    rel_error_2 = max_res_2 / max_ideal_2 if max_ideal_2 != 0 else 0
    
    # 打印双范围误差
    print(f"{name:<20} | {rel_error_2:<15.3f} | {rel_error_1:<15.3f}")
    
    # 2. AOD 光场传播 (不变)
    E_mod = E_in * np.exp(1j * phi_aod)
    E_out = angular_spectrum_propagate(E_mod, wl, z_prop, dx, dy)
    I_out = np.abs(E_out)**2
    I_out /= np.max(I_out)
    
    # 提取画图掩码 (不变)
    plot_mask = Rn_sq <= phase_mask_r**2
    phi_ideal_plot = np.where(plot_mask, phi_ideal, np.nan)
    phi_aod_plot = np.where(plot_mask, phi_aod, np.nan)
    phi_res_plot = np.where(plot_mask, phi_res, np.nan)
    
    # ================= 绘图 =================
    # 图 1: 理想相位
    ax1 = fig.add_subplot(gs[i, 0])
    im1 = ax1.imshow(phi_ideal_plot, extent=[-L/2*1e3, L/2*1e3, -L/2*1e3, L/2*1e3], cmap='RdBu_r', origin='lower')
    ax1.set_xlim([-lim_span, lim_span])
    ax1.set_ylim([-lim_span, lim_span])
    ax1.set_ylabel(f"{name}\ny (mm)", fontweight='bold')
    ax1.set_title("Ideal Phase (Target)")
    
    # 图 2: AOD实际相位
    ax2 = fig.add_subplot(gs[i, 1])
    im2 = ax2.imshow(phi_aod_plot, extent=[-L/2*1e3, L/2*1e3, -L/2*1e3, L/2*1e3], cmap='RdBu_r', origin='lower')
    ax2.set_xlim([-lim_span, lim_span])
    ax2.set_ylim([-lim_span, lim_span])
    ax2.set_title("AOD Actual Phase (No Cross-terms)")
    
   # 图 3: 残差 OPD (在标题中同时显示两个误差)
    ax3 = fig.add_subplot(gs[i, 2])
    im3 = ax3.imshow(phi_res_plot, extent=[-L/2*1e3, L/2*1e3, -L/2*1e3, L/2*1e3], cmap='coolwarm', origin='lower')
    ax3.set_xlim([-lim_span, lim_span])
    ax3.set_ylim([-lim_span, lim_span])
    # 将两个误差值都写进标题，用逗号分隔，格式对标文献
    ax3.set_title(rf"Residual $\Delta$OPD (Err: {rel_error_2:.2f}, {rel_error_1:.2f})")
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    
    # 图 4: 远场畸变光斑
    ax4 = fig.add_subplot(gs[i, 3])
    im4 = ax4.imshow(I_out, extent=[-L/2*1e3, L/2*1e3, -L/2*1e3, L/2*1e3], cmap='jet', origin='lower')
    ax4.set_xlim([-lim_span, lim_span])
    ax4.set_ylim([-lim_span, lim_span])
    ax4.set_title("Resulting Far-field Beam")

    if i == num_modes - 1:
        ax1.set_xlabel("x (mm)")
        ax2.set_xlabel("x (mm)")
        ax3.set_xlabel("x (mm)")
        ax4.set_xlabel("x (mm)")

plt.suptitle("AOD Partial Fitting for Complex Aberrations (Residual Analysis)", fontsize=18, y=0.92)
plt.subplots_adjust(top=0.88, bottom=0.05, left=0.05, right=0.95)

plt.savefig("AOD_Zernike_Partial_Fitting.png", dpi=300, bbox_inches='tight')
print("Simulation complete! Saved as AOD_Partial_Fitting.png")
plt.show()