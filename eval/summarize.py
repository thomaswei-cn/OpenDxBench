import json
import os
import pandas as pd
import numpy as np

def read_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return [json.loads(line) for line in file]

def write_xlsx(data, output_path, sheet_name='Sheet1'):
    df = pd.DataFrame(data)
    if os.path.exists(output_path):
        # 追加模式，并替换同名 sheet
        with pd.ExcelWriter(
            output_path,
            engine='openpyxl',
            mode='a',
            if_sheet_exists='replace'
        ) as writer:
            df.to_excel(writer, index=False, sheet_name=sheet_name)
    else:
        with pd.ExcelWriter(
            output_path,
            engine='openpyxl',
            mode='w'
        ) as writer:
            df.to_excel(writer, index=False, sheet_name=sheet_name)



def get_actual_valid_output_count(input_dir):
    """
    计算每个模型的实际有效输出数量。
    """
    data_metric_files = [f for f in os.listdir(input_dir) if f.endswith('.jsonl')]
    valid_counts = {"valid_preds": {}, "valid_standardized_preds": {}}

    for data_metric_file in data_metric_files:
        model_name = data_metric_file.split('_')[0]
        file_path = os.path.join(input_dir, data_metric_file)
        records = read_jsonl(file_path)

        if model_name not in valid_counts["valid_preds"]:
            valid_counts["valid_preds"][model_name] = 0
        if model_name not in valid_counts["valid_standardized_preds"]:
            valid_counts["valid_standardized_preds"][model_name] = 0

        for item in records:
            if 'pred_diagnoses' in item and item['pred_diagnoses']!= []:
                # 只计算有预测诊断的条目
                valid_counts["valid_preds"][model_name] += 1
            if 'standardized_pred_diagnosis' in item and item['standardized_pred_diagnosis'] != []:
                # 只计算有标准化预测诊断的条目
                valid_counts["valid_standardized_preds"][model_name] += 1

    return valid_counts

def summarize(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    data_files = [f for f in os.listdir(input_dir) if f.endswith('.jsonl')]
    valid_counts = get_actual_valid_output_count(input_dir)

    for top_k in [5, 10]:
        rate_lists = {}

        for fn in data_files:
            model = fn.split('_')[0]

            rate_lists.setdefault(model, {
                'contains_primary_rate_all': [],
                'gt_full_coverage_rate_all': [],
                'sim_contains_primary_rate_all': [],
                'sim_gt_full_coverage_rate_all': [],
                'avg_contains_primary_rate_all': [],        # 新增
                'avg_gt_full_coverage_rate_all': []         # 新增
            })

            records = read_jsonl(os.path.join(input_dir, fn))
            for item in records:
                m = item['metrics'][f'top_{top_k}']
                sim_m = item['sim_metrics'][f'top_{top_k}']
                dept = item['department']

                # 原有 & sim 指标
                rate_lists[model]['contains_primary_rate_all'].append(m['contains_primary'])
                rate_lists[model]['gt_full_coverage_rate_all'].append(m['gt_full_coverage'])
                rate_lists[model]['sim_contains_primary_rate_all'].append(sim_m['contains_primary'])
                rate_lists[model]['sim_gt_full_coverage_rate_all'].append(sim_m['gt_full_coverage'])

                # 计算平均指标
                avg_cp = (m['contains_primary'] + sim_m['contains_primary']) / 2
                avg_fc = (m['gt_full_coverage'] + sim_m['gt_full_coverage']) / 2
                rate_lists[model]['avg_contains_primary_rate_all'].append(avg_cp)
                rate_lists[model]['avg_gt_full_coverage_rate_all'].append(avg_fc)



        summary_all = []
        for model, lists in rate_lists.items():
            n = len(lists['contains_primary_rate_all'])
            # 均值计算
            mean_cp = np.mean(lists['contains_primary_rate_all'])
            mean_fc = np.mean(lists['gt_full_coverage_rate_all'])
            mean_sim_cp = np.mean(lists['sim_contains_primary_rate_all'])
            mean_sim_fc = np.mean(lists['sim_gt_full_coverage_rate_all'])
            mean_avg_cp = np.mean(lists['avg_contains_primary_rate_all'])
            mean_avg_fc = np.mean(lists['avg_gt_full_coverage_rate_all'])


            summary_all.append({
                'model': model,
                'icd_primary': mean_cp,
                'icd_complete': mean_fc,
                'sim_primary': mean_sim_cp,
                'sim_complete': mean_sim_fc,
                'avg_primary': mean_avg_cp,          # 平均值
                'avg_complete': mean_avg_fc,           # 平均值
                'n_all': n,
                'valid_preds_count': valid_counts['valid_preds'].get(model, 0),
                'valid_standardized_preds_count': valid_counts['valid_standardized_preds'].get(model, 0)
            })

        out_xlsx = os.path.join(output_dir, f'summary_top_{top_k}.xlsx')
        write_xlsx(summary_all, out_xlsx, sheet_name='summary')

        print(f"Summary written to {out_xlsx}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Summarize diagnosis ICD metrics.')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing the input JSONL files.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the output Excel file.')
    args = parser.parse_args()

    summarize(args.input_dir, args.output_dir)