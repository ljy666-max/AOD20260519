import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os

# ==========================================
# 全局设置
# ==========================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 11

wl = 355e-9           # 波长 (m)
w0 = 1.35e-3          # 1/e^2 束腰半径 (m)
V = 5750.0            # 声速 (m/s)
z_prop = 1.7          # 传播距离 (m)

# 跳变频率参数
f1 = 110e6            # 旧频率 (上方)
f2 = 140e6            # 新频率 (下方进场)

# 网格参数
L = 40e-3             
N = 2048              
dx = L / N            
dy = dx
x = np.linspace(-L/2, L/2, N)
y = np.linspace(-L/2, L/2, N)
X, Y = np.meshgrid(x, y)

# 输入光场
E_in = np.exp(-(X**2 + Y**2) / (w0**2))
I_in = np.abs(E_in)**2
k = 2 * np.pi / wl

# ==========================================
# 角谱传播法 (ASM)
# ==========================================
def angular_spectrum_propagate(E0, wl, z, dx, dy):
    Ny, Nx = E0.shape
    fx = np.fft.fftfreq(Nx, d=dx)
    fy = np.fft.fftfreq(Ny, d=dy)
    FX, FY = np.meshgrid(fx, fy)
    term = np.maximum(1.0 - (wl * FX)**2 - (wl * FY)**2, 0)
    H = np.exp(1j * k * z * np.sqrt(term))
    return np.fft.ifft2(np.fft.fft2(E0) * H)

# ==========================================
# 定义 5 个关键的脉冲到达时间 (微秒 us)
# t=0 为频率边界到达光束中心
# ==========================================
t_delays_us = [-0.4, -0.15, 0.0, 0.15, 0.4] 

fig = plt.figure(figsize=(18, 7))
gs = GridSpec(2, 5, hspace=0.3, wspace=0.3)
lim_span = 4 # 视窗半宽 4mm

print("Simulating Time-to-Space Mapping...")

for i, t_us in enumerate(t_delays_us):
    t = t_us * 1e-6
    # 1. 计算交界线物理位置
    y_bound = V * t
    
    # 2. 生成频率空间分布矩阵 (仅用于可视化上半图)
    # y < y_bound 是新频率 f2，y >= y_bound 是旧频率 f1
    f_space = np.where(Y < y_bound, f2, f1)
    
    # 3. 严密的连续相位计算
    # 确保在 y_bound 处： (2*pi/V)*f1*y_bound = (2*pi/V)*f2*y_bound + C
    # 解得 C = (2*pi/V) * (f1 - f2) * y_bound
    phi = np.zeros_like(Y)
    mask = Y < y_bound
    
    # 上半部 (旧频率)
    phi[~mask] = (2 * np.pi / V) * f1 * Y[~mask]
    
    # 下半部 (新频率) + 相位连续补偿常数
    C = (2 * np.pi / V) * (f1 - f2) * y_bound
    phi[mask] = (2 * np.pi / V) * f2 * Y[mask] + C
    
    # 4. 调制与传播
    E_mod = E_in * np.exp(1j * phi)
    E_out = angular_spectrum_propagate(E_mod, wl, z_prop, dx, dy)
    I_out = np.abs(E_out)**2
    I_out /= np.max(I_out) # 单独归一化以看清形貌
    
    # ================= 绘图 =================
    
    # 图 A (上排): AOD内部频率分布与激光光斑
    ax1 = fig.add_subplot(gs[0, i])
    # 绘制频率背景 (蓝色代表低频f1, 红色代表高频f2)
    im1 = ax1.imshow(f_space, extent=[-L/2*1e3, L/2*1e3, -L/2*1e3, L/2*1e3], cmap='coolwarm', origin='lower', vmin=100e6, vmax=150e6)
    # 叠加上高斯光束的等高线，表示激光打在这里
    ax1.contour(x*1e3, y*1e3, I_in, levels=[np.exp(-2)], colors='lime', linewidths=2, linestyles='--')
    
    ax1.set_xlim([-lim_span, lim_span])
    ax1.set_ylim([-lim_span, lim_span])
    
    # 状态判定与标题
    if t_us <= -0.3:
        state = "Too Early"
    elif t_us >= 0.3:
        state = "Synchronized"
    else:
        state = "Transitioning"
        
    ax1.set_title(f"Time: {t_us} $\mu$s\n({state})")
    if i == 0: ax1.set_ylabel("AOD $y$ (mm)")
    
    # 图 B (下排): 远场光斑
    ax2 = fig.add_subplot(gs[1, i])
    ax2.imshow(I_out, extent=[-L/2*1e3, L/2*1e3, -L/2*1e3, L/2*1e3], cmap='jet', origin='lower')
    
    # 视窗对准 125MHz 的中心区域
    y_center_mm = z_prop * np.tan((wl * 125e6) / V) * 1e3
    ax2.set_xlim([-lim_span, lim_span])
    ax2.set_ylim([y_center_mm - lim_span*1.2, y_center_mm + lim_span*1.2])
    
    # 画出两个理论靶点的参考线
    y1_th = z_prop * np.tan((wl * f1) / V) * 1e3
    y2_th = z_prop * np.tan((wl * f2) / V) * 1e3
    ax2.axhline(y1_th, color='white', linestyle=':', alpha=0.5)
    ax2.axhline(y2_th, color='white', linestyle=':', alpha=0.5)
    
    ax2.set_title("Far-field Intensity")
    ax2.set_xlabel("x (mm)")
    if i == 0: ax2.set_ylabel("y (mm)")

plt.suptitle("Time-to-Space Mapping: Effect of Laser Pulse Timing on Beam Deflection (Fig. 5)", fontsize=18, y=0.98)
plt.subplots_adjust(top=0.85, bottom=0.1, left=0.05, right=0.95)

# 保存
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
save_path = os.path.join(current_dir, "AOD_Time_to_Space_Fig5.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"时空转换仿真图像已成功保存至:\n{save_path}")

plt.show()