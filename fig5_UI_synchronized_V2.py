import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Polygon, Rectangle

# ==========================================
# 全局审美与字体设置
# ==========================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 11

# ==========================================
# 1. 物理参数与网格
# ==========================================
wl = 355e-9           # 波长 (m)
V = 5750.0            # 声速 (m/s)
z_prop = 1.7          # 传播距离 (m)
w0 = 1.35e-3          # 束腰半径 (m)

f1 = 110e6            # 旧频率 (110 MHz)
f2 = 140e6            # 新频率 (140 MHz)
f_center = 125e6

df1 = f1 - f_center
df2 = f2 - f_center

# 波动光学网格 (用于右图计算)
L_asm = 20e-3             
N = 512               
dx = L_asm / N            
x = np.linspace(-L_asm/2, L_asm/2, N)
y = np.linspace(-L_asm/2, L_asm/2, N)
X, Y = np.meshgrid(x, y)
k = 2 * np.pi / wl

term = np.maximum(1.0 - (wl * np.fft.fftfreq(N, d=dx))**2 - (wl * np.fft.fftfreq(N, d=dx)[:, None])**2, 0)
H_asm = np.exp(1j * k * z_prop * np.sqrt(term))

# ==========================================
# 2. 核心计算：ASM与光路几何
# ==========================================
def compute_physics(t_us):
    t = t_us * 1e-6
    y_bound = V * t
    
    # --- 1. 远场 ASM 计算 ---
    E_in = np.exp(-(X**2 + Y**2) / (w0**2))
    phi = np.zeros_like(Y)
    mask = Y < y_bound
    phi[~mask] = (2 * np.pi / V) * df1 * Y[~mask]
    C = (2 * np.pi / V) * (df1 - df2) * y_bound
    phi[mask] = (2 * np.pi / V) * df2 * Y[mask] + C
    
    E_out = np.fft.ifft2(np.fft.fft2(E_in * np.exp(1j * phi)) * H_asm)
    I_out = np.abs(E_out)**2
    I_out /= np.max(I_out)
    
    # --- 2. 光栅可视化计算 (放大频率以肉眼可见) ---
    # 模拟声波疏密，保证在 y_bound 处相位连续
    vis_k1 = 2 * np.pi * 3.0  # f1 对应的稀疏波数
    vis_k2 = 2 * np.pi * 6.0  # f2 对应的密集波数
    
    y_line = np.linspace(-4, 4, 800)
    phi_vis = np.zeros_like(y_line)
    m_vis = y_line < (y_bound * 1e3)
    phi_vis[~m_vis] = vis_k1 * y_line[~m_vis]
    C_vis = (vis_k1 - vis_k2) * (y_bound * 1e3)
    phi_vis[m_vis] = vis_k2 * y_line[m_vis] + C_vis
    grating_pattern = np.sin(phi_vis).reshape(-1, 1)
    
    return y_bound, I_out, grating_pattern

# ==========================================
# 3. 构建 UI 界面
# ==========================================
fig = plt.figure(figsize=(14, 7), facecolor='#f8f9fa')
plt.subplots_adjust(left=0.05, bottom=0.25, right=0.97, top=0.9, wspace=0.25)
gs = GridSpec(1, 2, width_ratios=[1.2, 1])

init_t_us = 0.0
y_bound_init, I_out_init, grating_init = compute_physics(init_t_us)
w0_mm = w0 * 1e3

# ---------------- 左图：宏观光路与光栅 ----------------
ax1 = fig.add_subplot(gs[0, 0], facecolor='#2c3e50')
ax1.set_xlim(-6, 8)
ax1.set_ylim(-5, 5)

# 画 AOD 晶体外框
crystal_width = 2
crystal_left = -1
ax1.add_patch(Rectangle((crystal_left, -4.5), crystal_width, 9, fill=False, edgecolor='white', lw=2, zorder=5))

# 初始化光栅图像
im_grating = ax1.imshow(grating_init, extent=[crystal_left, crystal_left+crystal_width, -4.5, 4.5], 
                        cmap='gray', origin='lower', alpha=0.8, aspect='auto', zorder=2)

# 入射光 (左侧直线传播)
ax1.add_patch(Polygon([(-6, w0_mm), (crystal_left, w0_mm), (crystal_left, -w0_mm), (-6, -w0_mm)], 
                      color='#3b00ff', alpha=0.4, zorder=3))

# 初始化偏转光束 (出射部分)
beam_top = ax1.add_patch(Polygon([[0,0],[0,0],[0,0]], color='#0073ff', alpha=0.6, zorder=3))
beam_bot = ax1.add_patch(Polygon([[0,0],[0,0],[0,0]], color='#00d0ff', alpha=0.6, zorder=3))

# 交界线
line_bound_left = ax1.axhline(y_bound_init*1e3, color='red', linewidth=2.5, linestyle='--', zorder=6)

ax1.set_title("AOD Acoustic Grating & Ray Tracing", fontweight='bold', pad=10)
ax1.set_xlabel("$z$ axis (Propagation Direction)")
ax1.set_ylabel("$y$ axis (Deflection Axis)")
ax1.text(-5.5, w0_mm+0.3, 'Input Laser', color='cyan', fontweight='bold')
ax1.text(3, 4, 'Diffracted Beam (110 MHz)', color='cyan', fontweight='bold')
ax1.text(3, -4, 'Diffracted Beam (140 MHz)', color='cyan', fontweight='bold')

# ---------------- 右图：微观远场光斑 ----------------
y_center_mm = z_prop * np.tan((wl * f_center) / V) * 1e3
extent_right = [-L_asm/2*1e3, L_asm/2*1e3, y_center_mm - L_asm/2*1e3, y_center_mm + L_asm/2*1e3]

ax2 = fig.add_subplot(gs[0, 1])
im_out = ax2.imshow(I_out_init, extent=extent_right, cmap='jet', origin='lower')
ax2.set_xlim([-4, 4])
ax2.set_ylim([y_center_mm - 4.5, y_center_mm + 4.5])

y1_th = z_prop * np.tan((wl * f1) / V) * 1e3
y2_th = z_prop * np.tan((wl * f2) / V) * 1e3
ax2.axhline(y1_th, color='white', linestyle=':', linewidth=1.5, alpha=0.7)
ax2.axhline(y2_th, color='white', linestyle=':', linewidth=1.5, alpha=0.7)
ax2.text(3.8, y1_th+0.2, '110 MHz Target', color='white', ha='right', fontsize=10)
ax2.text(3.8, y2_th+0.2, '140 MHz Target', color='white', ha='right', fontsize=10)

ax2.set_title("Far-Field Interference Spot", fontweight='bold', pad=10)
ax2.set_xlabel("$x$ (mm)")
ax2.set_ylabel("$y$ (mm)")

# ==========================================
# 4. 更新逻辑与交互
# ==========================================
# 夸张的偏转角 (仅用于左侧可视化)
theta_1_vis = 0.2  # 向上偏 
theta_2_vis = 0.6 

def update_rays(y_bound_mm):
    right_edge = crystal_left + crystal_width
    z_end = 8
    dist = z_end - right_edge
    
    # 上半部分光束 (f1, 稀疏, 偏角小 -> 浅蓝色)
    y_top1 = w0_mm
    y_bot1 = max(-w0_mm, y_bound_mm)
    if y_bot1 < y_top1:
        pts = [(right_edge, y_top1), (z_end, y_top1 + dist*theta_1_vis),
               (z_end, y_bot1 + dist*theta_1_vis), (right_edge, y_bot1)]
        beam_top.set_xy(pts)
        beam_top.set_visible(True)
    else:
        beam_top.set_visible(False)
        
    # 下半部分光束 (f2, 密集, 偏角大 -> 橙色)
    y_top2 = min(w0_mm, y_bound_mm)
    y_bot2 = -w0_mm
    if y_bot2 < y_top2:
        pts = [(right_edge, y_top2), (z_end, y_top2 + dist*theta_2_vis),
               (z_end, y_bot2 + dist*theta_2_vis), (right_edge, y_bot2)]
        beam_bot.set_xy(pts)
        beam_bot.set_visible(True)
    else:
        beam_bot.set_visible(False)

update_rays(y_bound_init * 1e3)

# 配置滑块
ax_t = plt.axes([0.2, 0.08, 0.6, 0.04], facecolor='#ecf0f1')
slider_t = Slider(ax_t, r'Pulse Timing $t$ ($\mu$s)', -0.4, 0.4, valinit=init_t_us, color='#e74c3c')

def update(val):
    t_val = slider_t.val
    y_bound, I_out, grating = compute_physics(t_val)
    
    y_bound_mm = y_bound * 1e3
    line_bound_left.set_ydata([y_bound_mm, y_bound_mm])
    
    # 更新光栅与光路
    im_grating.set_data(grating)
    update_rays(y_bound_mm)
    
    # 更新远场
    im_out.set_data(I_out)
    fig.canvas.draw_idle()

slider_t.on_changed(update)

fig.suptitle("AOD Time-to-Space Mapping & Ray Tracing Simulator", fontsize=18, fontweight='bold', y=0.98)
plt.show()