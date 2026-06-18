import numpy as np
import matplotlib.pyplot as plt

# 1. System Physics Parameters (Aligned with report data)
kappa = 0.6779      # System constant factor (MHz/rad) [cite: 98]
A_max = 0.62        # Max allowable amplitude for Z11 (rad) [cite: 101, 102]

# 2. Generate Normalized Coordinate zn (-1.0 to +1.0)
zn = np.linspace(-1.0, 1.0, 500)

# 3. Calculate Spatial Gradients G = dPhi_norm / dzn
# U-axis channel: 4*sqrt(5)*u^4 -> Derivative: 16*sqrt(5)*u^3 
g_u = 16 * np.sqrt(5) * (zn**3)
# X-axis channel: 4*sqrt(5)*x^4 - 6*sqrt(5)*x^2 -> Derivative: 16*sqrt(5)*x^3 - 12*sqrt(5)*x
g_x = 16 * np.sqrt(5) * (zn**3) - 12 * np.sqrt(5) * zn

# 4. Calculate RF Frequency Deviations: Delta_f = kappa * A_max * G
df_u = kappa * A_max * g_u
df_x = kappa * A_max * g_x

# 5. Plotting Academic-grade Comparison Figure (Pure English)
fig, ax1 = plt.subplots(figsize=(8.0, 6.0), facecolor='white')

# Set standard academic fonts
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 11

# Plot Left Axis: Normalized Spatial Gradients G
line_u, = ax1.plot(zn, g_u, color='#d62728', linewidth=3, label='U-axis Channel (System Short-board)')
line_x, = ax1.plot(zn, g_x, color='#1f77b4', linewidth=2.5, linestyle='--', label='X-axis Channel (Multi-peak Risk)')

ax1.set_xlabel('Normalized Coordinate ($z_n = x$ or $u$)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Normalized Spatial Gradient $G$', color='black', fontsize=12, fontweight='bold')
ax1.set_xlim([-1.05, 1.05])
ax1.set_ylim([-45, 45])
ax1.grid(True, linestyle=':', alpha=0.6)

# Plot Right Axis: RF Frequency Deviation Delta_f (MHz)
ax2 = ax1.twinx()
ax2.set_ylabel('RF Frequency Deviation $\Delta f$ (MHz)', color='black', fontsize=12, fontweight='bold')
ax2.set_ylim([-19, 19])

# Draw Hardware Bandwidth Limits (±15 MHz Red Dashed Lines) [cite: 99]
ax2.axhline(15, color='#d62728', linestyle=':', linewidth=1.5, alpha=0.7)
ax2.axhline(-15, color='#d62728', linestyle=':', linewidth=1.5, alpha=0.7)

# ==========================================
# 🔴 CRITICAL POINTS MARKING FOR U-AXIS
# ==========================================
# Edge Maximum and Minimum Points (+1.0 and -1.0)
ax2.plot(1.0, 15.0, 'ro', markersize=8, zorder=5)
ax2.text(0.18, 15.6, 'Hits HW Limit (+15.0 MHz)', color='#d62728', fontsize=9.5, fontweight='bold')

ax2.plot(-1.0, -15.0, 'ro', markersize=8, zorder=5)
ax2.text(-0.95, -14.2, 'Hits HW Limit (-15.0 MHz)', color='#d62728', fontsize=9.5, fontweight='bold')

# ==========================================
# 🔵 CRITICAL POINTS MARKING FOR X-AXIS
# ==========================================
# 1. Edge Points (+1.0 and -1.0) -> Delta_f = +3.76 MHz and -3.76 MHz
ax2.plot(1.0, 3.76, 'bo', markersize=7, zorder=5)
ax2.text(0.55, 4.6, 'Edge (+3.76 MHz)', color='#1f77b4', fontsize=9, fontweight='bold')

ax2.plot(-1.0, -3.76, 'bo', markersize=7, zorder=5)
ax2.text(-0.98, -5.4, 'Edge (-3.76 MHz)', color='#1f77b4', fontsize=9, fontweight='bold')

# 2. Internal Peak/Valley Points (+0.5 and -0.5) -> Delta_f = -3.76 MHz and +3.76 MHz
ax2.plot(0.5, -3.76, 'b^', markersize=8, zorder=5)
ax2.text(0.18, -5.4, 'Valley (-3.76 MHz)', color='#1f77b4', fontsize=9, fontweight='bold')

ax2.plot(-0.5, 3.76, 'b^', markersize=8, zorder=5)
ax2.text(-0.48, 4.6, 'Peak (+3.76 MHz)', color='#1f77b4', fontsize=9, fontweight='bold')

# Vertical auxiliary lines to guide the eyes for X-axis inner peaks
ax2.axvline(0.5, color='#1f77b4', linestyle=':', linewidth=1, alpha=0.4)
ax2.axvline(-0.5, color='#1f77b4', linestyle=':', linewidth=1, alpha=0.4)

# ==========================================
# 🟢 SAFE DRIVING ZONE SHADING
# ==========================================
ax2.fill_between(zn, -15, 15, color='green', alpha=0.03)
ax2.text(-0.2, +10, 'Safe Driving Zone', color='green', fontsize=12, alpha=0.5, fontweight='bold')

# Merge Legends from both lines
lines = [line_u, line_x]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper left', framealpha=0.9)

plt.title('Multi-axis Bandwidth Bottleneck & Risk Points Analysis for $Z_{11}$\n(At Critical Amplitude $A = 0.62$ rad)', 
          fontsize=12, fontweight='bold', pad=15)
fig.tight_layout()

# Save high-definition image to the same folder (600 DPI for publication standard)
plt.savefig('Z11_bandwidth_bottleneck.png', dpi=600, bbox_inches='tight')

# Show the elegant figure
plt.show()