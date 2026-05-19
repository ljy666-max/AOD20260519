import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ==========================================
# 全局设置
# ==========================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 11

# 物理参数
wl = 355e-9           # 波长 (m)
w0 = 1.35e-3          # 束腰半径 (m)
V = 5750.0            # 声速 (m/s)
z_prop = 1.7          # 传播距离 (m)

# 网格参数
L = 40e-3             
N = 2048              
dx = L / N            
dy = dx
x = np.linspace(-L/2, L/2, N)
y = np.linspace(-L/2, L/2, N)
X, Y = np.meshgrid(x, y)

# 归一化坐标 (基于束腰)
Xn = X / w0
Yn = Y / w0

# 输入光场
E_in = np.exp(-(Xn**2 + Yn**2))
I_in = np.abs(E_in)**2
I_in /= np.max(I_in)

# 波数
k = 2 * np.pi / wl

# ==========================================
# 角谱传播法 (ASM)
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
# 定义需要仿真的 Zernike 像差模式
# ==========================================
# 【核心修改】将振幅系数降低到符合 1.7m 传播距离的物理真实量级，彻底消除相位混叠
modes = {
    'Z2_Tip': {
        'A': 40,  # 适度偏转，产生约几毫米的偏移
        'phase_func': lambda a: a * 2 * Xn,
        'desc': "X-Shift (Linear Phase)"
    },
    'Z3_Tilt': {
        'A': 40,
        'phase_func': lambda a: a * 2 * Yn,
        'desc': "Y-Shift (Linear Phase)"
    },
    'Z4_Defocus': {
        'A': -2.7, # 根据等效透镜焦距 1.7m 严格推导出的系数
        'phase_func': lambda a: a * 2 * np.sqrt(3) * (Xn**2 + Yn**2),
        'desc': "Focusing (Quadratic Phase)"
    },
    'Z6_Astigmatism': {
        'A': -2.7, # 匹配散焦的强度，形成清晰的椭圆
        'phase_func': lambda a: a * np.sqrt(6) * (Xn**2 - Yn**2),
        'desc': "Astigmatism (Elliptical Focus)"
    },
    'Z12_2nd_Astig': {
        'A': -2.0, # 高阶像差，形成复杂的能量分裂
        'phase_func': lambda a: a * 4 * np.sqrt(10) * (Xn**4 - Yn**4),
        'desc': "4th Order Phase (Cross Pattern)"
    }
}

# ==========================================
# 循环仿真并绘图
# ==========================================
num_modes = len(modes)
fig = plt.figure(figsize=(15, 3 * num_modes))
gs = GridSpec(num_modes, 3, width_ratios=[1, 1, 1.2], wspace=0.3, hspace=0.4)

lim_span = 4 # mm，视窗显示范围

for i, (name, params) in enumerate(modes.items()):
    print(f"Simulating {name}...")
    
    # 1. 计算相位
    A = params['A']
    phi = params['phase_func'](A)
    
    # 2. 调制与传播
    E_mod = E_in * np.exp(1j * phi)
    E_out = angular_spectrum_propagate(E_mod, wl, z_prop, dx, dy)
    
    I_out = np.abs(E_out)**2
    I_out /= np.max(I_out)
    
    # 获取最高强度点的截面
    peak_y_idx, peak_x_idx = np.unravel_index(np.argmax(I_out), I_out.shape)
    prof_x = I_out[peak_y_idx, :]
    prof_y = I_out[:, peak_x_idx]
    
    # ================= 绘图 =================
    
    # 【核心修改】图 A: 替换为输入光强 (Input Intensity)
    # 在 ax1 处增加大标签
    ax1 = fig.add_subplot(gs[i, 0])
    ax1.imshow(I_in, extent=[-L/2*1e3, L/2*1e3, -L/2*1e3, L/2*1e3], cmap='jet', origin='lower')
    ax1.set_xlim([-lim_span, lim_span])
    ax1.set_ylim([-lim_span, lim_span])
    ax1.set_title("Input Intensity (z=0m)")
    ax1.set_ylabel("y (mm)")
    if i == num_modes - 1:
        ax1.set_xlabel("x (mm)")
    
    # 图 B: 远场光斑
    ax2 = fig.add_subplot(gs[i, 1])
    ax2.imshow(I_out, extent=[-L/2*1e3, L/2*1e3, -L/2*1e3, L/2*1e3], cmap='jet', origin='lower')
    ax2.set_xlim([-lim_span, lim_span])
    ax2.set_ylim([-lim_span, lim_span])
    # 修改为显示 Zernike 模式名称
    
    # 若发生严重偏转 (如Z2, Z3)，让视窗跟随光斑中心
    if name in ['Z2_Tip', 'Z3_Tilt']:
        cy, cx = y[peak_y_idx]*1e3, x[peak_x_idx]*1e3
        ax2.set_xlim([cx - lim_span/2, cx + lim_span/2])
        ax2.set_ylim([cy - lim_span/2, cy + lim_span/2])
        
    ax2.set_title(f"{name}\nFar-field Intensity (z={z_prop}m)")
    if i == num_modes - 1:
        ax2.set_xlabel("x (mm)")
    
    # 图 C: 1D 强度剖面
    ax3 = fig.add_subplot(gs[i, 2])
    ax3.plot(x*1e3, prof_x, 'r-', linewidth=1.5, label='X Profile')
    ax3.plot(y*1e3, prof_y, 'b--', linewidth=1.5, label='Y Profile')
    ax3.set_xlim([-lim_span, lim_span])
    
    if name in ['Z2_Tip', 'Z3_Tilt']:
        cy, cx = y[peak_y_idx]*1e3, x[peak_x_idx]*1e3
        ax3.set_xlim([min(cx, cy) - lim_span/2, max(cx, cy) + lim_span/2])
        
    ax3.set_title(params['desc'])
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.5)
    if i == num_modes - 1:
        ax3.set_xlabel("Position (mm)")

plt.suptitle("Acousto-Optical Wavefront Shaping (Zernike Polynomials)", fontsize=16, y=0.92)
plt.subplots_adjust(top=0.88, bottom=0.05, left=0.05, right=0.95)

# 保存大图
plt.savefig("AOD_Zernike_Shaping_Fitted_Perfectly.png", dpi=300, bbox_inches='tight')
print("Simulation complete! Saved as AOD_Zernike_Shaping_Fixed.png")
plt.show()