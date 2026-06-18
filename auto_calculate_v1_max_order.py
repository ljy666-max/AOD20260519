import numpy as np
import math
import sys

def print_polynomial_format(k):
    """根据 k 打印多项式的排列顺序，方便用户对照输入系数"""
    terms = []
    for j in range(k + 1):
        x_power = k - j
        y_power = j
        
        term = ""
        if x_power > 0:
            term += f"x^{x_power}" if x_power > 1 else "x"
        if y_power > 0:
            term += f"y^{y_power}" if y_power > 1 else "y"
            
        if not term:
            term = "1"
            
        terms.append(f"C{j}*{term}")
        
    print(f"\n当前 k={k}，对应的多项式格式为:")
    print(" + ".join(terms))
    print(f"你需要输入 {k+1} 个系数 [C0, C1, ... C{k}]\n")

def wavefront_compiler(k, C_target):
    # 固定的 4 轴 AOD 配置
    aod_angles_deg = [0.0, 90.0, 45.0, -45.0]
    N = len(aod_angles_deg)
    aod_angles_rad = np.radians(aod_angles_deg)
    
    M = np.zeros((k + 1, N))
    
    # 构建传输矩阵 M
    for j in range(k + 1):
        for i in range(N):
            theta = aod_angles_rad[i]
            M[j, i] = math.comb(k, j) * (np.cos(theta)**(k - j)) * (np.sin(theta)**j)
            
    C_target = np.array(C_target).reshape(-1, 1)
    
    print("-" * 50)
    # 求解
    if N == k + 1:
        print("[状态] AOD 数量与维数匹配，执行精确求逆...")
        A_aod = np.dot(np.linalg.inv(M), C_target)
    elif N > k + 1:
        print("[状态] AOD 数量冗余，执行最小二乘最优解 (伪逆)...")
        A_aod = np.dot(np.linalg.pinv(M), C_target)
    else:
        print("[警告] AOD 数量不足以完美拟合任意交叉项！")
        print("[状态] 正在强行求解最佳近似解 (最小误差)...")
        A_aod = np.dot(np.linalg.pinv(M), C_target)
        
    # 验证是否为精确解
    C_calc = np.dot(M, A_aod)
    residual = np.linalg.norm(C_calc - C_target)
    
    print("\n【编译结果】各 AOD 所需的振幅系数 (A):")
    for idx, angle in enumerate(aod_angles_deg):
        print(f"  -> AOD_{idx+1} ({angle:>5.1f}°): {A_aod[idx][0]:>10.4f}")
        
    if residual < 1e-10:
        print("\n✅ 误差分析: 零残差完美拟合！")
    else:
        print(f"\n⚠️ 误差分析: 存在无法完全消去的残差 (误差范数: {residual:.4f})")
        print("原因: k=4 时全相空间需要 5 个 AOD，当前的 4 个 AOD 只能做近似，或只能完美拟合偶对称的特殊项(如球差)。")
    print("-" * 50)


# ==========================================
# 终端交互主循环
# ==========================================
if __name__ == "__main__":
    print("=" * 50)
    print(" 🚀 欢迎使用级联 AOD 波前编译器 (4轴版) 🚀")
    print(" 硬件配置: 0°, 90°, 45°, -45°")
    print("=" * 50)
    
    while True:
        try:
            k_input = input("\n👉 请输入想要拟合的总阶数 k (1-4，输入 q 退出): ").strip()
            if k_input.lower() == 'q':
                print("再见！")
                sys.exit(0)
                
            k = int(k_input)
            if k < 1 or k > 4:
                print("为了保证系统稳定性，目前仅支持 k = 1, 2, 3, 4。请重新输入。")
                continue
                
            # 打印多项式提示
            print_polynomial_format(k)
            
            c_input = input(f"👉 请输入 {k+1} 个系数 (用逗号分隔, 例如 '0,1,1,0'): ").strip()
            
            # 解析系数向量
            C_list = [float(x) for x in c_input.split(',')]
            
            if len(C_list) != (k + 1):
                print(f"❌ 错误: 你输入了 {len(C_list)} 个系数，但 k={k} 需要 {k+1} 个系数！")
                continue
                
            # 调用编译器
            wavefront_compiler(k, C_list)
            
        except ValueError:
            print("❌ 输入格式错误！请输入合法的数字。")
        except Exception as e:
            print(f"❌ 发生未知错误: {e}")

            