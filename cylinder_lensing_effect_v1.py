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

# 计算等效柱透镜焦距以作理论验证
# 根据公式：f_cyl = V^2 / (wl * dfdt)
f_cyl = (V**2) / (wl * dfdt)
print(f"理论验证: AOD等效柱透镜焦距 f_cyl = {f_cyl:.3f} m")

# ==========================================
# 2. 空间网格初始化
# ==========================================
# 建议网格大小为20mm，远大于束腰的4倍，防止角谱法在边界发生周期性混叠(Wrap-around)
L = 20e-3             # 网格边长 20 mm
N = 2048              # 采样点数 (2048保证了高分辨率且符合FFT的最优性能)
dx = L / N            # 空间采样间隔 (约 9.7 um)
dy = dx

x = np.linspace(-L/2, L/2, N)
y = np.linspace(-L/2, L/2, N)
X, Y = np.meshgrid(x, y)

# ==========================================
# 3. 生成输入光场与AOD相位调制
# ==========================================
# 3.1 生成无相差的高斯输入光场 E_in
E_in = np.exp(-(X**2 + Y**2) / (w0**2))

# 3.2 计算柱透镜效应引入的相位 phi(y)
# phi(y) = (pi / V^2) * (df/dt) * y^2
phi_y = (np.pi / V**2) * dfdt * (Y**2)

# 3.3 对输入光束施加相位调制 (仅在y方向)
# 这里忽略整体线性偏转相位，只关注导致像散的二次相位
E_mod = E_in * np.exp(-1j * phi_y)

# ==========================================
# 4. 角谱传播法 (Angular Spectrum Method) 定义与计算
# ==========================================
def angular_spectrum_propagate(E0, wl, z, dx, dy):
    """
    使用角谱法计算光波在自由空间中的传播
    """
    Ny, Nx = E0.shape
    # 生成频域坐标
    fx = np.fft.fftfreq(Nx, d=dx)
    fy = np.fft.fftfreq(Ny, d=dy)
    FX, FY = np.meshgrid(fx, fy)
    
    # 传递函数 H(fx, fy) = exp(1j * k * z * sqrt(1 - (wl*fx)^2 - (wl*fy)^2))
    # 为防止倏逝波(Evanescent waves)导致复数开根号报错，将负数置0
    term = 1.0 - (wl * FX)**2 - (wl * FY)**2
    term = np.maximum(term, 0)
    
    # 计算空间频率传递函数 H
    k = 2 * np.pi / wl
    H = np.exp(1j * k * z * np.sqrt(term))
    
    # 将源光场变换到频域，乘以传递函数，再逆变换回空域
    E0_fft = np.fft.fft2(E0)
    E_out_fft = E0_fft * H
    E_out = np.fft.ifft2(E_out_fft)
    
    return E_out

# 执行传播计算，得到 z=1.7m 处的远场复振幅
E_out = angular_spectrum_propagate(E_mod, wl, z, dx, dy)

# ==========================================
# 5. 光强计算与截面分析
# ==========================================
I_in = np.abs(E_in)**2
I_out = np.abs(E_out)**2

# 归一化光强
I_in /= np.max(I_in)
I_out /= np.max(I_out)

# 提取中心剖面数据用于画图和计算宽度
center_idx = N // 2
profile_x_in = I_in[center_idx, :]  # X方向未调制剖面
profile_y_in = I_in[:, center_idx]  # Y方向未调制剖面
profile_x_out = I_out[center_idx, :] # 远场X方向剖面
profile_y_out = I_out[:, center_idx] # 远场Y方向剖面（受柱面镜效应影响）

# 计算远场 1/e^2 (约0.135) 阈值处的宽度
threshold = np.exp(-2)
wx_out = np.sum(profile_x_out > threshold) * dx / 2
wy_out = np.sum(profile_y_out > threshold) * dy / 2

ellipticity = wy_out / wx_out if wy_out < wx_out else wx_out / wy_out
print(f"远场光斑 x 方向 1/e^2 半径: {wx_out*1e3:.3f} mm")
print(f"远场光斑 y 方向 1/e^2 半径: {wy_out*1e3:.3f} mm")
print(f"计算所得椭圆度 (短轴/长轴): {ellipticity:.3f}")

# ==========================================
# 6. 可视化输出
# ==========================================
fig = plt.figure(figsize=(14, 10))
gs = GridSpec(2, 3, figure=fig, width_ratios=[1, 1, 1], height_ratios=[1, 0.4])

# 视场显示限制 (mm)
lim_mm = 4

# 图1: 输入光斑
ax1 = fig.add_subplot(gs[0, 0])
im1 = ax1.imshow(I_in, extent=[-L/2*1e3, L/2*1e3, -L/2*1e3, L/2*1e3], cmap='jet')
ax1.set_xlim([-lim_mm, lim_mm])
ax1.set_ylim([-lim_mm, lim_mm])
ax1.set_title("Input Gaussian Beam (z = 0 m)")
ax1.set_xlabel("x (mm)")
ax1.set_ylabel("y (mm)")

# 图2: 远场光斑 (z=1.7m)
ax2 = fig.add_subplot(gs[0, 1])
im2 = ax2.imshow(I_out, extent=[-L/2*1e3, L/2*1e3, -L/2*1e3, L/2*1e3], cmap='jet')
ax2.set_xlim([-lim_mm, lim_mm])
ax2.set_ylim([-lim_mm, lim_mm])
ax2.set_title(f"Distorted Beam at z = {z}m\nEllipticity = {ellipticity:.2f}")
ax2.set_xlabel("x (mm)")
ax2.set_ylabel("y (mm)")

# 图3: 输入光斑剖面
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(x*1e3, profile_x_in, 'r-', label='x-profile')
ax3.plot(y*1e3, profile_y_in, 'b--', label='y-profile')
ax3.axhline(threshold, color='k', linestyle=':', label='1/e² threshold')
ax3.set_xlim([-lim_mm, lim_mm])
ax3.set_title("Input Beam Profile")
ax3.set_xlabel("Position (mm)")
ax3.set_ylabel("Normalized Intensity")
ax3.legend()
ax3.grid(True)

# 图4: 远场光斑剖面
ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(x*1e3, profile_x_out, 'r-', label='x-profile (Unmodulated)')
ax4.plot(y*1e3, profile_y_out, 'b-', label='y-profile (Focused by AOD)')
ax4.axhline(threshold, color='k', linestyle=':', label='1/e² threshold')
ax4.set_xlim([-lim_mm, lim_mm])
ax4.set_title("Output Beam Profile (Astigmatism)")
ax4.set_xlabel("Position (mm)")
ax4.legend()
ax4.grid(True)

plt.tight_layout()
plt.show()