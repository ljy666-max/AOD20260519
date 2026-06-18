import numpy as np
import math
import sys

def print_header():
    print("=" * 60)
    print(" 🌟 级联 AOD 全阶次波前编译器 (终极版 v1.0) 🌟")
    print(" 硬件架构: 4-Axis AOD Cascade [0°, 90°, 45°, -45°]")
    print(" 核心算法: 多维矩阵广义反演 + 阶次线性叠加")
    print("=" * 60)
    print("【操作指引】系统将从最高阶(4阶)到最低阶(1阶)依次请求输入。")
    print(" 如果某阶不需要，请直接输入 0。\n")

def get_coefficients_input(k):
    """引导用户输入某阶数的所有系数"""
    # 构造公式字符串提示
    terms = []
    for j in range(k + 1):
        x_p, y_p = k - j, j
        term = ""
        if x_p > 0: term += f"x^{x_p}" if x_p > 1 else "x"
        if y_p > 0: term += f"y^{y_p}" if y_p > 1 else "y"
        if not term: term = "1"
        terms.append(f"C{j}*{term}")
        
    formula = " + ".join(terms)
    
    while True:
        print(f"\n▶ [当前处理: {k} 阶像差]")
        print(f"  公式模板: {formula}")
        user_in = input(f"  请输入 {k+1} 个系数 (用逗号分隔，全零直接填 0): ").strip()
        
        if user_in == '0':
            return np.zeros(k + 1)
            
        try:
            coeffs = [float(x) for x in user_in.split(',')]
            if len(coeffs) != (k + 1):
                print(f"  ❌ 错误：需要 {k+1} 个系数，您输入了 {len(coeffs)} 个。")
                continue
            return np.array(coeffs)
        except ValueError:
            print("  ❌ 错误：输入包含非法字符，请重新输入数字。")

def solve_for_k(k, C_target, aod_angles_deg):
    """计算单阶的 AOD 振幅系数"""
    N = len(aod_angles_deg)
    aod_angles_rad = np.radians(aod_angles_deg)
    M = np.zeros((k + 1, N))
    
    # 构建传输矩阵
    for j in range(k + 1):
        for i in range(N):
            theta = aod_angles_rad[i]
            M[j, i] = math.comb(k, j) * (np.cos(theta)**(k - j)) * (np.sin(theta)**j)
            
    # 求解 (统一使用伪逆，涵盖满秩与欠秩情况)
    A_aod = np.dot(np.linalg.pinv(M), C_target)
    
    # 残差检查
    residual = np.linalg.norm(np.dot(M, A_aod) - C_target)
    if residual > 1e-4:
        print(f"  ⚠️ 警告: {k}阶矩阵求解存在残差(不能完美拟合所有交叉项)，已自动转为最优近似解。")
        
    return A_aod

# ==========================================
# 主程序
# ==========================================
if __name__ == "__main__":
    print_header()
    
    # 硬件角度定义
    angles = [0.0, 90.0, 45.0, -45.0]
    labels = ["X", "Y", "U", "V"]
    
    # 用于累加存储每个 AOD 从 1阶到 4阶的系数
    # 结构: aod_results[i] = [0阶, 1阶, 2阶, 3阶, 4阶]
    aod_results = np.zeros((4, 5)) 
    
    # 从高阶到低阶遍历计算
    for k in [4, 3, 2, 1]:
        C_target = get_coefficients_input(k)
        
        # 如果该阶系数全为0，直接跳过计算，节省算力
        if np.all(C_target == 0):
            print(f"  ✓ {k} 阶系数为空，自动跳过。")
            continue
            
        # 求解该阶所需的 AOD 振幅
        A_k = solve_for_k(k, C_target, angles)
        
        # 将求解结果累加到对应的 AOD 和对应的阶数位置
        for i in range(len(angles)):
            aod_results[i][k] = A_k[i]
            
        print(f"  ✓ {k} 阶矩阵反演完成！")

    # ==========================================
    # 终极物理配方输出 (非常适合写进论文/报告)
    # ==========================================
    print("\n" + "=" * 60)
    print(" 🎯 编译完成！级联系统终极物理控制配方 🎯")
    print("=" * 60)
    
    for i in range(4):
        print(f"\n🔹 AOD_{i+1} (物理安装角: {angles[i]:>5.1f}°, 独立坐标轴: {labels[i]})")
        
        coeffs = aod_results[i]
        
        # 1. 打印空间相位多项式 Φ(r)
        phase_terms = []
        for p in [4, 3, 2, 1]:
            if abs(coeffs[p]) > 1e-6:
                phase_terms.append(f"{coeffs[p]:+.4f}*{labels[i]}^{p}")
                
        if not phase_terms:
            print("  ➤ 空间相位 Φ = 0.0 (该 AOD 可关闭)")
            continue
            
        phase_str = " ".join(phase_terms)
        print(f"  ➤ 空间相位: Φ({labels[i]}) = {phase_str}")
        
        # 2. 打印射频扫频频率方程 f(t) ∝ dΦ/dt
        # 频率是相位的空间导数 (体现了时频控制的核心逻辑)
        freq_terms = []
        for p in [4, 3, 2, 1]:
            if abs(coeffs[p]) > 1e-6:
                derive_coeff = coeffs[p] * p
                if p - 1 == 0:
                    freq_terms.append(f"{derive_coeff:+.4f}")
                elif p - 1 == 1:
                    freq_terms.append(f"{derive_coeff:+.4f}*t")
                else:
                    freq_terms.append(f"{derive_coeff:+.4f}*t^{p-1}")
                    
        freq_str = " ".join(freq_terms)
        print(f"  ➤ 射频扫频: f(t) ∝ {freq_str}")
        
    print("\n" + "=" * 60)
    print(" 💡 恭喜！您已成功生成包含全阶次波前补偿的 AOD 硬件驱动方程。")