import numpy as np

def wavefront_compiler(k, aod_angles_deg, C_target):
    """
    波前编译器核心矩阵求解器
    :param k: 像差的总阶数 (例如 2阶像散, 3阶彗差)
    :param aod_angles_deg: 列表中包含每个AOD的物理旋转角度 (单位: 度)
    :param C_target: 目标像差分量的系数列向量 (长度必须为 k+1)
    :return: 每个AOD所需的振幅系数向量 A_aod
    """
    N = len(aod_angles_deg)  # AOD 的数量
    aod_angles_rad = np.radians(aod_angles_deg)  # 转换为弧度
    
    # 1. 初始化传输矩阵 M，尺寸为 (k+1) x N
    M = np.zeros((k + 1, N))
    
    # 阶乘计算函数，用于计算二项式系数 C(k, j)
    from scipy.special import comb
    
    # 2. 自动构建传输矩阵 M
    # 矩阵的每一行对应多项式展开的某一项 (x^k, x^{k-1}y, ..., y^k)
    # 矩阵的每一列对应一个特定的 AOD
    for j in range(k + 1):
        for i in range(N):
            theta = aod_angles_rad[i]
            # 根据二项式定理定理：C(k, j) * cos(theta)^(k-j) * sin(theta)^j
            M[j, i] = comb(k, j) * (np.cos(theta)**(k - j)) * (np.sin(theta)**j)
            
    print("--- 1. 成功构建传输矩阵 M (尺寸 {}x{}) ---".format(k+1, N))
    print(np.round(M, 4))
    
    # 3. 求解 AOD 振幅系数向量 A
    C_target = np.array(C_target).reshape(-1, 1) # 确保是列向量
    
    if N == k + 1:
        # 情况A：刚好满秩方阵，直接求逆
        print("\n--- [情况A] AOD数量刚好等于空间维数，执行标准矩阵求逆 ---")
        try:
            M_inv = np.linalg.inv(M)
            A_aod = np.dot(M_inv, C_target)
        except np.linalg.LinAlgError:
            print("警告：矩阵奇异（不可逆），说明选择的AOD角度存在线性相关！")
            return None
    elif N > k + 1:
        # 情况B：AOD数量冗余，求解最小二乘/最小功耗解，使用伪逆
        print("\n--- [情况B] AOD数量存在冗余，执行广义伪逆求解(最小功耗优化) ---")
        M_pinv = np.linalg.pinv(M)
        A_aod = np.dot(M_pinv, C_target)
    else:
        # 情况C：AOD数量不足，方程无解
        print("\n--- [情况C] AOD数量不足，无法完全拟合该阶数的所有交叉项！ ---")
        # 此时可以用最小二乘求近似解（残差最小）
        M_pinv = np.linalg.pinv(M)
        A_aod = np.dot(M_pinv, C_target)
        print("注意：此解为最佳近似（残差最小化解），无法做到零残差。")
        
    return A_aod

# ==========================================
# 验证：复现我们之前手算的 2阶 xy 像差 ($Z_5$)
# ==========================================
# 2阶像差包含 3 项：[x^2, xy, y^2]
k_test = 3 
# 硬件配置：4 个 AOD，角度分别为 0度, 90度, 45度
angles_test = [0.0, 90.0, 45.0, -45.0] 
# 目标：我们想要纯 xy 项（假设系数为1），不要 x^2 和 y^2。
# 对应的目标列向量为：[x^2的系数=0, xy的系数=1, y^2的系数=0]
C_target_test = [0.0, 1.0, 1.0,0]

# 运行编译器
A_result = wavefront_compiler(k_test, angles_test, C_target_test)

if A_result is not None:
    print("\n--- 2. 自动编译输出：各 AOD 振幅系数 ---")
    for idx, angle in enumerate(angles_test):
        print("AOD_{} ({:>.1f}°): 振幅系数 = {:>.4f}".format(idx+1, angle, A_result[idx][0]))