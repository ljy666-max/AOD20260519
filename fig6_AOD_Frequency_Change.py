import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os

# ==========================================
# 全局设置与物理参数
# ==========================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 12

wl = 355e-9           # 波长 (m)
V = 5750.0            # 声速 (m/s)
z_prop = 1.7          # 传播距离 (m)
w0 = 1.35e-3          # 束腰半径 (m)
f_center = 125e6      # 载波中心频率

# ==========================================
# 网格初始化 (剥离载波频率，防混叠且加速)
# ==========================================
L = 15e-3             # 15mm 的视窗足以容纳 Fig 6 的所有光斑
N = 1024              
dx = L / N            
x = np.linspace(-L/2, L/2, N)
y = np.linspace(-L/2, L/2, N)
X, Y = np.meshgrid(x, y)
k = 2 * np.pi / wl

# 预计算 ASM 传递函数
fx_freq = np.fft.fftfreq(N, d=dx)
fy_freq = np.fft.fftfreq(N, d=dx)
FX, FY = np.meshgrid(fx_freq, fy_freq)
term = np.maximum(1.0 - (wl * FX)**2 - (wl * FY)**2, 0)
H_asm = np.exp(1j * k * z_prop * np.sqrt(term))

E_in = np.exp(-(X**2 + Y**2) / (w0**2))

def simulate_fig6(fy1, fx1, fy2, fx2):
    """根据输入的四个象限频率，生成远场光斑"""
    # 计算相对于中心频率的偏移量
    dfy1 = (fy1 - 125) * 1e6
    dfx1 = (fx1 - 125) * 1e6
    dfy2 = (fy2 - 125) * 1e6
    dfx2 = (fx2 - 125) * 1e6
    
    # 构造空间频率分布 (以 x=0, y=0 为界)
    dfx_space = np.where(X < 0, dfx1, dfx2)
    dfy_space = np.where(Y < 0, dfy1, dfy2)
    
    # 计算相位 (原点处天然连续，无需常数 C)
    phi_x = (2 * np.pi / V) * dfx_space * X
    phi_y = (2 * np.pi / V) * dfy_space * Y
    phi = phi_x + phi_y
    
    # 角谱传播
    E_mod = E_in * np.exp(1j * phi)
    E_out = np.fft.ifft2(np.fft.fft2(E_mod) * H_asm)
    I_out = np.abs(E_out)**2
    I_out /= np.max(I_out)
    
    return phi, I_out

# ==========================================
# 终端交互主循环
# ==========================================
print("="*60)
print("  AOD 2D Wavefront Shaping Simulator (Paper Fig 6)  ")
print("="*60)
print("Reference Presets from the Paper:")
print("  - Fig 6(a) No shaping : 125 125 125 125")
print("  - Fig 6(b) Top-Hat    : 124 124 126 126")
print("  - Fig 6(c) Convergence: 127 127 123 123")
print("  - Fig 6(d) 4 Beams    : 118 118 132 132")
print("="*60)

while True:
    try:
        user_input = input("\nEnter fy1, fx1, fy2, fx2 separated by spaces (or 'q' to quit): ")
        if user_input.lower() == 'q':
            break
            
        freqs = [float(f) for f in user_input.split()]
        if len(freqs) != 4:
            print("Error: Please enter exactly 4 numbers.")
            continue
            
        fy1, fx1, fy2, fx2 = freqs
        print(f"\nSimulating: fy1={fy1}, fx1={fx1}, fy2={fy2}, fx2={fx2} MHz...")
        
        phi_map, I_out = simulate_fig6(fy1, fx1, fy2, fx2)
        
        # 绘图展示
        fig = plt.figure(figsize=(12, 5), facecolor='#f8f9fa')
        gs = GridSpec(1, 2, width_ratios=[1, 1.2])
        
        # 左图：施加的相位形貌
        ax1 = fig.add_subplot(gs[0, 0])
        # 掩码去掉边缘无能量的区域，增强对比度
        mask = (X**2 + Y**2) < (2*w0)**2
        phi_plot = np.where(mask, phi_map, np.nan)
        im1 = ax1.imshow(phi_plot, extent=[-L/2*1e3, L/2*1e3, -L/2*1e3, L/2*1e3], cmap='twilight_shifted', origin='lower')
        ax1.set_xlim([-3, 3])
        ax1.set_ylim([-3, 3])
        ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax1.axvline(0, color='gray', linestyle='--', alpha=0.5)
        ax1.set_title("Generated Phase Map $\phi(x,y)$")
        ax1.set_xlabel("$x$ (mm)")
        ax1.set_ylabel("$y$ (mm)")
        
        # 右图：远场光斑
        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(I_out, extent=[-L/2*1e3, L/2*1e3, -L/2*1e3, L/2*1e3], cmap='jet', origin='lower')
        ax2.set_xlim([-3, 3])
        ax2.set_ylim([-3, 3])
        ax2.set_title(f"Far-Field Spot\nInput: {fy1} / {fx1} / {fy2} / {fx2} MHz")
        ax2.set_xlabel("Relative $x$ (mm)")
        ax2.set_ylabel("Relative $y$ (mm)")
        
        plt.tight_layout()
        plt.show()
        
    except ValueError:
        print("Invalid input. Please enter numbers like '124 124 126 126'.")