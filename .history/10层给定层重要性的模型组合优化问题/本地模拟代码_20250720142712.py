import kaiwu as kw
import numpy as np
import os
import time
start_time= time.perf_counter()

def generate_layer_parameters(num_layers: int) -> np.ndarray:
    """
    为每个层随机生成参数数量 (单位: 百万)。

    Args:
        num_layers (int): 模型的总层数。

    Returns:
        np.ndarray: 一个包含每层参数量的NumPy数组。
    """
    # 假设每层的参数量在100万到1000万之间
    return np.random.uniform(1, 10, size=num_layers)


def calculate_memory_costs(num_params: np.ndarray, bit_widths: list) -> np.ndarray:
    """
    根据参数数量和目标位宽，计算每层在不同位宽下的内存成本 (单位: 兆比特 Mb)。

    Args:
        num_params (np.ndarray): 每层的参数量数组 (单位: 百万)。
        bit_widths (list): 目标位宽的列表。

    Returns:
        np.ndarray: 一个(层数 x 位宽数)的成本矩阵。
    """
    num_layers = len(num_params)
    num_levels = len(bit_widths)
    costs = np.zeros((num_layers, num_levels))

    for i in range(num_layers):
        for j in range(num_levels):
            # 内存成本 = 参数量(百万) * 位宽
            costs[i, j] = num_params[i] * bit_widths[j]

    return costs


if __name__ == "__main__":
    # --- 1. 定义问题和动态生成数据 ---
    NUM_LAYERS = 10
    TARGET_BIT_WIDTHS = [3, 4, 8]
    NUM_LEVELS = len(TARGET_BIT_WIDTHS)
    ORIGINAL_BIT_WIDTH = 16

    # 随机生成每层的重要性和参数量
    importances = np.random.randint(10, 101, size=NUM_LAYERS)
    num_params_in_millions = generate_layer_parameters(NUM_LAYERS)

    # 动态计算成本矩阵和精度下降值
    mem_costs = calculate_memory_costs(num_params_in_millions, TARGET_BIT_WIDTHS)
    precision_drop_values = np.array([ORIGINAL_BIT_WIDTH - b for b in TARGET_BIT_WIDTHS])

    # 核心参数：内存权重因子
    # 用于平衡“精度下降”和“内存成本”这两个目标
    # 一个好的初始值可以让两个目标的量级大致相当
    memory_weight = 20.0

    print("--- 问题设置 (软目标模型) ---")
    print(f"各层随机生成的重要性: \n{importances}")
    print(f"各层参数量 (百万): \n{np.round(num_params_in_millions, 2)}")

    # --- 2. 使用 Kaiwu SDK 建模 ---
    qubo_model = kw.qubo.QuboModel()
    x = kw.qubo.ndarray((NUM_LAYERS, NUM_LEVELS), 'x', kw.qubo.Binary)

    # 构建新的统一目标函数
    # 目标第一部分：加权精度下降
    objective_precision = kw.qubo.quicksum([
        x[i, j] * importances[i] * precision_drop_values[j]
        for i in range(NUM_LAYERS) for j in range(NUM_LEVELS)])

    # 目标第二部分：总内存成本
    objective_memory = kw.qubo.quicksum([
        x[i, j] * mem_costs[i, j]
        for i in range(NUM_LAYERS) for j in range(NUM_LEVELS)
    ])

    # 将两部分加权合并成最终目标，不再有内存硬约束
    final_objective = objective_precision + memory_weight * objective_memory
    qubo_model.set_objective(final_objective)

    # 保留绝对必要的逻辑约束：每个层必须选择一个位宽
    # 这个约束的惩罚值需要非常高，以确保其被严格满足
    for i in range(NUM_LAYERS):
        one_choice_per_layer_expr = kw.qubo.quicksum([x[i, j] for j in range(NUM_LEVELS)])
        qubo_model.add_constraint(one_choice_per_layer_expr == 1, name=f"layer_{i}_choice", penalty=50000.0)

    # --- 3. 求解 (使用模拟退火) ---
    print("\n--- 开始求解... ---")

    # 使用一套更“耐心”的参数，以适应新的平滑模型
    optimizer = kw.classical.SimulatedAnnealingOptimizer(
        initial_temperature=1000,
        alpha=0.998,
        cutoff_temperature=0.01,
        iterations_per_t=1000,
        rand_seed=42
    )

    solver = kw.solver.SimpleSolver(optimizer)
    sol_dict, qubo_val = solver.solve_qubo(qubo_model)

    # --- 4. 解释结果 ---
    unsatisfied_count, result_dict = qubo_model.verify_constraint(sol_dict)
    print(f"未满足的约束数量: {unsatisfied_count}")

    if unsatisfied_count == 0:
        print("✅ 找到了一个满足所有约束的可行解！")
        x_solution = kw.qubo.get_array_val(x, sol_dict)
        chosen_level_indices = np.argmax(x_solution, axis=1)

        total_final_cost = 0
        total_final_drop = 0

        print("\n--- 最终分配方案 ---")
        print("层\t重要性\t分配位宽\t内存成本(Mb)")
        print("-" * 50)
        for i in range(NUM_LAYERS):
            level_idx = chosen_level_indices[i]
            chosen_bits = TARGET_BIT_WIDTHS[level_idx]
            cost = mem_costs[i, level_idx]
            drop = importances[i] * precision_drop_values[level_idx]
            total_final_cost += cost
            total_final_drop += drop
            print(f"{i}\t{importances[i]}\t\t{chosen_bits}-bit\t\t{cost:.2f}")
        print("-" * 50)
        print(f"最终总成本 (软目标结果): {total_final_cost:.2f} Mb")
        print(f"最终加权精度下降值 (软目标结果): {total_final_drop:.2f}")
    else:
        print("❌ 未找到可行解。")
        print("详细约束检查结果:", result_dict)
end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(f"总耗时：{elapsed_time}")