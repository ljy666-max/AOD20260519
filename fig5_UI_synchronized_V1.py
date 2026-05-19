import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle

# ==========================================
# 全局审美与字体设置 (学术级精美排版)
# ==========================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

# ==========================================
# 1. 物理参数与包络加速网格
# ==========================================
wl = 355e-9           # 波长 (m)
V = 5750.0            # 声速 (m/s)
z_prop = 1.7          # 传播距离 (m)

f1 = 110e6            # 旧频率
f2 = 140e6            # 新频率
f_center = 125e6      # 基础载波频率

# 【物理优化】计算相对频率偏移，防止高频相位混叠和光斑飞出边界
df1 = f1 - f_center   # -15 MHz
df2 = f2 - f_center   # +15 MHz

L = 20e-3             # 计算网格 20mm
N = 512               # 512网格足以保证高帧率和低混叠
dx = L / N            
x = np.linspace(-L/2, L/2, N)
y = np.linspace(-L/2, L/2, N)
X, Y = np.meshgrid(x, y)
k = 2 * np.pi / wl

# 预计算 ASM 传递函数
fx = np.fft.fftfreq(N, d=dx)
fy = np.fft.fftfreq(N, d=dx)
FX, FY = np.meshgrid(fx, fy)
term = np.maximum(1.0 - (wl * FX)**2 - (wl * FY)**2, 0)
H_asm = np.exp(1j * k * z_prop * np.sqrt(term))

# ==========================================
# 2. 核心场计算函数
# ==========================================
def compute_fields(t_us, w0_mm):
    t = t_us * 1e-6
    w0 = w0_mm * 1e-3
    
    E_in = np.exp(-(X**2 + Y**2) / (w0**2))
    y_bound = V * t
    
    # 使用减去载波后的相对频率 (df1, df2) 计算相位
    phi = np.zeros_like(Y)
    mask = Y < y_bound
    
    phi[~mask] = (2 * np.pi / V) * df1 * Y[~mask]
    C = (2 * np.pi / V) * (df1 - df2) * y_bound
    phi[mask] = (2 * np.pi / V) * df2 * Y[mask] + C
    
    # 传播
    E_mod = E_in * np.exp(1j * phi)
    E_out = np.fft.ifft2(np.fft.fft2(E_mod) * H_asm)
    I_out = np.abs(E_out)**2
    I_out /= np.max(I_out)
    
    return y_bound, I_out

# ==========================================
# 3. 构建高颜值 UI 界面
# ==========================================
bg_color = '#f4f6f9'  # 现代灰蓝色背景
fig = plt.figure(figsize=(13, 6.5), facecolor=bg_color)
plt.subplots_adjust(left=0.08, bottom=0.28, right=0.95, top=0.88, wspace=0.3)
gs = GridSpec(1, 2)

init_t_us = -0.4
init_w0_mm = 1.35
y_bound_init, I_out_init = compute_fields(init_t_us, init_w0_mm)

lim_span = 4 # 视图显示半宽

# --- 左图：晶体内部界面 ---
ax1 = fig.add_subplot(gs[0, 0], facecolor='white')
line_bound = ax1.axhline(y_bound_init*1e3, color='#e74c3c', linewidth=2.5, label='Acoustic Boundary')
ax1.axhline(0, color='gray', linestyle='--', alpha=0.5, label='Beam Center')

# 使用 Patch 画出完美圆形光斑，避免了 Contour 的报错，且渲染极快
beam_circle = Circle((0, 0), radius=init_w0_mm, fill=True, color='#2ecc71', alpha=0.3)
beam_edge = Circle((0, 0), radius=init_w0_mm, fill=False, color='#27ae60', linewidth=2, linestyle='--')
ax1.add_patch(beam_circle)
ax1.add_patch(beam_edge)

ax1.set_xlim([-lim_span, lim_span])
ax1.set_ylim([-lim_span, lim_span])
ax1.set_title("AOD Crystal (Time-to-Space Mapping)", fontweight='bold', pad=15)
ax1.set_xlabel("$x$ (mm)")
ax1.set_ylabel("$y$ (mm)")
ax1.legend(loc='upper right', framealpha=0.9, edgecolor='none')
ax1.grid(True, linestyle=':', alpha=0.6)

# --- 右图：远场光斑 ---
# 计算真实的 y 中心坐标，以在图中映射出真实偏转高度
y_center_mm = z_prop * np.tan((wl * f_center) / V) * 1e3
extent_right = [-L/2*1e3, L/2*1e3, y_center_mm - L/2*1e3, y_center_mm + L/2*1e3]

ax2 = fig.add_subplot(gs[0, 1])
im_out = ax2.imshow(I_out_init, extent=extent_right, cmap='jet', origin='lower')

ax2.set_xlim([-lim_span, lim_span])
ax2.set_ylim([y_center_mm - lim_span*1.2, y_center_mm + lim_span*1.2])

y1_th = z_prop * np.tan((wl * f1) / V) * 1e3
y2_th = z_prop * np.tan((wl * f2) / V) * 1e3
ax2.axhline(y1_th, color='white', linestyle=':', linewidth=1.5, alpha=0.7)
ax2.axhline(y2_th, color='white', linestyle=':', linewidth=1.5, alpha=0.7)
# 加上靶点文字标注
ax2.text(lim_span-0.2, y1_th+0.2, '110 MHz Target', color='white', ha='right', fontsize=10)
ax2.text(lim_span-0.2, y2_th+0.2, '140 MHz Target', color='white', ha='right', fontsize=10)

ax2.set_title("Resulting Far-Field Intensity", fontweight='bold', pad=15)
ax2.set_xlabel("$x$ (mm)")
ax2.set_ylabel("$y$ (mm)")

# ==========================================
# 4. 配置美观的交互滑块
# ==========================================
slider_bg = '#ecf0f1'
ax_t = plt.axes([0.15, 0.12, 0.7, 0.04], facecolor=slider_bg)
ax_w0 = plt.axes([0.15, 0.05, 0.7, 0.04], facecolor=slider_bg)

# 使用 r 字符串修复 \m 报错
slider_t = Slider(ax_t, r'Timing $t$ ($\mu$s)', -0.5, 0.5, valinit=init_t_us, valstep=0.01, color='#3498db')
slider_w0 = Slider(ax_w0, r'Beam $w_0$ (mm)', 0.5, 3.0, valinit=init_w0_mm, valstep=0.05, color='#9b59b6')

# 自定义滑块数值文本字体
slider_t.valtext.set_fontfamily('serif')
slider_w0.valtext.set_fontfamily('serif')

def update(val):
    t_val = slider_t.val
    w0_val = slider_w0.val
    
    y_bound, I_out = compute_fields(t_val, w0_val)
    
    # 极速更新左图对象
    line_bound.set_ydata([y_bound*1e3, y_bound*1e3])
    beam_circle.set_radius(w0_val)
    beam_edge.set_radius(w0_val)
    
    # 极速更新右图热力图
    im_out.set_data(I_out)
    
    fig.canvas.draw_idle()

slider_t.on_changed(update)
slider_w0.on_changed(update)

# 总标题
fig.suptitle("Ultrafast Laser Pulse & AOD Synchronization Dynamics", fontsize=16, fontweight='bold', y=0.96)

plt.show()