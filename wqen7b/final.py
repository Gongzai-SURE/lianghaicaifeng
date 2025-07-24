import numpy as np
import torch
from transformers import AutoModelForCausalLM
from typing import Tuple


# ==============================================================================
# 模块一：从模型中提取信息 (无修改)
# ==============================================================================

def get_quantizable_layers(model: torch.nn.Module) -> list:
    quantizable_layers = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            quantizable_layers.append(module)
    return quantizable_layers


def calculate_layer_metrics(layers: list) -> Tuple[np.ndarray, np.ndarray]:
    num_params_list, sensitivities_list = [], []
    print(f"开始重新计算 {len(layers)} 个层的指标以进行结果分析...")
    with torch.no_grad():
        for i, layer in enumerate(layers):
            print(f"\r  正在处理第 {i + 1}/{len(layers)} 层...", end="")
            params_in_millions = layer.weight.nelement() / 1_000_000
            num_params_list.append(params_in_millions)
            weight_tensor = layer.weight
            materialized_weight = weight_tensor.to(device="cpu", dtype=torch.float32)
            sensitivity = torch.linalg.norm(materialized_weight).item()
            sensitivities_list.append(sensitivity)
    print("\n指标计算完成！")
    sensitivities_np = np.array(sensitivities_list)
    s_min, s_max = sensitivities_np.min(), sensitivities_np.max()
    if s_max > s_min:
        sensitivities_normalized = 10 + 90 * (sensitivities_np - s_min) / (s_max - s_min)
    else:
        sensitivities_normalized = np.full_like(sensitivities_np, 50)
    return np.array(num_params_list), sensitivities_normalized


def calculate_precision_loss(bit: int, original_bit_width: int, sensitivity: float, alpha: float) -> float:
    denominator = np.exp(-alpha * (1.0 / original_bit_width)) - np.exp(-alpha)
    if np.isclose(denominator, 0): return np.inf
    numerator = np.exp(-alpha * (bit / original_bit_width)) - np.exp(-alpha)
    return sensitivity * (numerator / denominator)


def calculate_memory_costs(num_params: np.ndarray, bit_widths: list) -> np.ndarray:
    num_layers, num_levels = len(num_params), len(bit_widths)
    costs = np.zeros((num_layers, num_levels))
    for i in range(num_layers):
        for j in range(num_levels):
            costs[i, j] = num_params[i] * bit_widths[j]
    return costs


# ==============================================================================
# 主程序：解读结果
# ==============================================================================

if __name__ == "__main__":
    # --- 步骤1: 将云平台返回的最佳解向量粘贴到这里 ---
    solution_vector = [1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0,
                       0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1,
                       0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0,
                       1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,
                       1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0,
                       0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1,
                       0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0,
                       1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0,
                       1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1,
                       0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1,
                       0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0,
                       1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0,
                       0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0,
                       1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1,
                       0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0,
                       1, 0, 0]

    # --- 步骤2: 重新生成与求解时一致的上下文数据 ---
    qwen7b_model_path = r"D:\working\project\挑战杯量子ai\kaiwu\wqen实现\Qwen\Qwen-7B-Chat"

    print(f"--- 重新加载模型以生成分析所需的上下文数据 ---")
    model = AutoModelForCausalLM.from_pretrained(qwen7b_model_path, trust_remote_code=True)

    quantizable_layers = get_quantizable_layers(model)
    NUM_LAYERS = len(quantizable_layers)

    num_params_in_millions, sensitivities = calculate_layer_metrics(quantizable_layers)

    TARGET_BIT_WIDTHS = [3, 4, 8]
    NUM_LEVELS = len(TARGET_BIT_WIDTHS)
    ORIGINAL_BIT_WIDTH = 16
    ALPHA = 2.0

    mem_costs = calculate_memory_costs(num_params_in_millions, TARGET_BIT_WIDTHS)
    precision_loss_matrix = np.zeros((NUM_LAYERS, NUM_LEVELS))
    for i in range(NUM_LAYERS):
        for j in range(NUM_LEVELS):
            precision_loss_matrix[i, j] = calculate_precision_loss(
                TARGET_BIT_WIDTHS[j], ORIGINAL_BIT_WIDTH, sensitivities[i], ALPHA
            )

    # --- 步骤3: 解读结果并以易懂的格式输出 ---
    if len(solution_vector) != NUM_LAYERS * NUM_LEVELS:
        print(f"错误：结果向量的长度({len(solution_vector)})与预期({NUM_LAYERS * NUM_LEVELS})不符！")
    else:
        solution_matrix = np.array(solution_vector).reshape((NUM_LAYERS, NUM_LEVELS))
        chosen_level_indices = np.argmax(solution_matrix, axis=1)

        total_final_cost = 0
        total_final_loss = 0

        print("\n\n--- Qwen-7B最优量化分配方案 (来自云平台) ---")
        print("层Idx\t敏感度(S)\t参数(M)\t分配位宽\t内存成本(Mb)\t精度损失(ΔAcc)")
        print("-" * 90)
        for i in range(NUM_LAYERS):
            level_idx = chosen_level_indices[i]
            chosen_bits = TARGET_BIT_WIDTHS[level_idx]
            cost = mem_costs[i, level_idx]
            loss = precision_loss_matrix[i, level_idx]
            total_final_cost += cost
            total_final_loss += loss
            print(
                f"{i}\t{sensitivities[i]:.2f}\t\t{num_params_in_millions[i]:.2f}\t{chosen_bits}-bit\t\t{cost:.2f}\t\t{loss:.4f}")
        print("-" * 90)

        # --- 新增：计算并显示原始成本与压缩率 ---
        total_params_in_millions = np.sum(num_params_in_millions)
        total_original_cost_mb = total_params_in_millions * ORIGINAL_BIT_WIDTH

        print(f"原始总成本 (16-bit): {total_original_cost_mb:.2f} Mb")
        print(f"量化后总成本: {total_final_cost:.2f} Mb")
        # 避免除以零的错误
        if total_final_cost > 0:
            print(f"模型压缩率: {total_original_cost_mb / total_final_cost:.2f}x")
        print(f"最终总精度损失: {total_final_loss:.4f}")
        # ------------------------------------