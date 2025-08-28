import os
from metrics import run_metrics
from standardized_pred import get_pred_icd_11_parallel,correct_pred_icd_11_none
from run import main_eval_parallel, main_eval
from summarize import summarize


def run_pipeline(benchmark_jsonl, models, res_dir, max_retries=3, max_workers=8, parallel=False, api_key=None):
    """
    执行完整的评估管道：
    1️⃣ 模型预测
    2️⃣ ICD-11 编码
    3️⃣ 计算指标
    4️⃣ 汇总结果
    """
    os.makedirs(res_dir, exist_ok=True)  # 创建主结果目录

    # Step 1: 模型预测
    raw_dir = os.path.join(res_dir, 'raw')
    os.makedirs(raw_dir, exist_ok=True)
    print("\033[94m🚀 开始模型预测...\033[0m")  # 蓝色
    for model in models:
        print(f"\033[94m🤖 运行模型：{model}\033[0m")
        if parallel:
            main_eval_parallel(benchmark_jsonl, model, raw_dir, max_retries, max_workers, api_key)
        else:
            main_eval(benchmark_jsonl, model, raw_dir, max_retries, api_key)

    # Step 2: ICD-11 编码
    icd_dir = os.path.join(res_dir, 'standard')
    os.makedirs(icd_dir, exist_ok=True)
    print("\033[92m🔖 开始 ICD-11 编码...\033[0m")  # 绿色
    for pred_file in os.listdir(raw_dir):
        if pred_file.endswith('_diagnoses.jsonl'):
            get_pred_icd_11_parallel(os.path.join(raw_dir, pred_file), icd_dir, max_workers)
            correct_pred_icd_11_none(os.path.join(raw_dir, pred_file), icd_dir)

    # Step 3: 计算指标
    metrics_dir = os.path.join(res_dir, 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)
    print("\033[93m📊 开始计算指标...\033[0m")  # 黄色
    run_metrics(icd_dir, benchmark_jsonl, metrics_dir)

    # Step 4: 汇总结果
    sum_dir = os.path.join(res_dir, 'summary')
    os.makedirs(sum_dir, exist_ok=True)
    print("\033[95m📑 开始汇总结果...\033[0m")  # 品红
    summarize(metrics_dir, sum_dir)

    print("\033[96m✅ Pipeline 执行完成！\033[0m")  # 青色


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the evaluation pipeline for ICD-11 diagnosis prediction.")
    parser.add_argument('--benchmark_jsonl', type=str, required=True, help='Path to the benchmark JSONL file.')
    parser.add_argument('--models', type=str, nargs='+', required=True, help='List of model names to evaluate.')
    parser.add_argument('--res_dir', type=str, default='results', help='Directory to save the results.')
    parser.add_argument('--max_retries', type=int, default=3, help='Maximum number of retries.')
    parser.add_argument('--max_workers', type=int, default=8, help='Number of parallel workers for evaluation.')
    parser.add_argument('--parallel', action='store_true', help='Use parallel processing for model evaluation.')
    parser.add_argument('--api_key', type=str, default=None, help='API key for OpenAI models (if required).')
    args = parser.parse_args()

    run_pipeline(args.benchmark_jsonl, args.models, args.res_dir,
                 max_retries=args.max_retries, max_workers=args.max_workers, parallel=args.parallel, api_key=args.api_key)