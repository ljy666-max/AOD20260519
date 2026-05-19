import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ==========================================
# 1. 物理参数与环境设置
# ==========================================
wl = 355e-9           # 激光波长 (m)
w0 = 1.35e-3          # 1/e^2 束腰半径 (m)
V = 5750.0            # 石英晶体声速 (m/s)
dfdt = 3e13           # 频率啁啾率 (Hz/s)
z = 1.7               # 传播距离 (m)

# 【新增参数】中心偏转设置
f0 = 125e6            # AOD中心工作频率 125 MHz
theta0 = (wl * f0) / V  # 基础偏转角 (rad)
expected_shift = z * np.tan(theta0) # 预计在屏幕上的纵向偏移量

print(f"理论基础偏转角: {theta0*1e3:.2f} mrad")
print(f"理论光斑中心偏移量: {expected_shift*1e3:.2f} mm")

# ==========================================
# 2. 空间网格初始化
# ==========================================
# 【修改】网格扩大至 40mm 以容纳偏转出去的光束，防止FFT混叠
L = 40e-3             
N = 2048              
dx = L / N            
dy = dx

x = np.linspace(-L/2, L/2, N)
y = np.linspace(-L/2, L/2, N)
X, Y = np.meshgrid(x, y)

# ==========================================
# 3. 生成输入光场与AOD完整相位调制
# ==========================================
E_in = np.exp(-(X**2 + Y**2) / (w0**2))

k = 2 * np.pi / wl
# 【新增】线性相位：对应基础偏转角 theta0
phi_linear = k * theta0 * Y
# 二次相位：对应柱面镜效应 (像散)
phi_quad = (np.pi / V**2) * dfdt * (Y**2)

# 总相位调制
E_mod = E_in * np.exp(1j * (phi_linear + phi_quad))

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
    E_out = np.fft.ifft2(E0_fft * H)
    return E_out

E_out = angular_spectrum_propagate(E_mod, wl, z, dx, dy)

# ==========================================
# 5. 光强计算与【动态】截面提取
# ==========================================
I_in = np.abs(E_in)**2
I_out = np.abs(E_out)**2
I_in /= np.max(I_in)
I_out /= np.max(I_out)

# 动态寻找远场光斑的能量最高点(峰值中心)
peak_y_idx, peak_x_idx = np.unravel_index(np.argmax(I_out), I_out.shape)

profile_x_in = I_in[N//2, :]  
profile_y_in = I_in[:, N//2]  
profile_x_out = I_out[peak_y_idx, :] # 穿过偏转后光斑中心的X剖面
profile_y_out = I_out[:, peak_x_idx] # 穿过偏转后光斑中心的Y剖面

# 计算宽度
threshold = np.exp(-2)
wx_out = np.sum(profile_x_out > threshold) * dx / 2
wy_out = np.sum(profile_y_out > threshold) * dy / 2
ellipticity = wy_out / wx_out if wy_out < wx_out else wx_out / wy_out

# ==========================================
# 6. 可视化输出
# ==========================================
fig = plt.figure(figsize=(14, 10))
gs = GridSpec(2, 3, figure=fig, width_ratios=[1, 1, 1], height_ratios=[1, 0.4])

# 注意：这里调整了显示窗口，让偏移量 (约13mm) 能完美展示
lim_x = 4    # x方向保持对称
lim_y_min = -2
lim_y_max = 18 # y方向扩展到18mm

ax1 = fig.add_subplot(gs[0, 0])
ax1.imshow(I_in, extent=[-L/2*1e3, L/2*1e3, -L/2*1e3, L/2*1e3], cmap='jet', origin='lower')
ax1.set_xlim([-lim_x, lim_x])
ax1.set_ylim([-lim_x, lim_x]) # 输入光斑仍在中心，用小窗口
ax1.set_title("Input Gaussian Beam (z = 0 m)")
ax1.set_xlabel("x (mm)")
ax1.set_ylabel("y (mm)")

ax2 = fig.add_subplot(gs[0, 1])
ax2.imshow(I_out, extent=[-L/2*1e3, L/2*1e3, -L/2*1e3, L/2*1e3], cmap='jet', origin='lower')
ax2.set_xlim([-lim_x, lim_x])
ax2.set_ylim([lim_y_min, lim_y_max]) # 重点：调整视窗追踪偏转光束
ax2.set_title(f"Distorted & Deflected Beam at z=1.7m\nDeflection: {expected_shift*1e3:.1f} mm")
ax2.set_xlabel("x (mm)")
ax2.set_ylabel("y (mm)")

ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(y*1e3, profile_y_in, 'k-', label='y-profile (Input)')
ax3.axhline(threshold, color='r', linestyle=':', label='1/e² threshold')
ax3.set_xlim([-lim_x, lim_x])
ax3.set_title("Input Y Profile")
ax3.legend()
ax3.grid(True)

ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(y*1e3, profile_y_out, 'b-', label='y-profile (Deflected)')
ax4.axvline(expected_shift*1e3, color='g', linestyle='--', label='Theoretical Peak')
ax4.axhline(threshold, color='r', linestyle=':')
ax4.set_xlim([lim_y_min, lim_y_max])
ax4.set_title(f"Output Y Profile (Shift & Broaden)")
ax4.set_xlabel("y Position (mm)")
ax4.legend()
ax4.grid(True)

plt.tight_layout()
plt.show()