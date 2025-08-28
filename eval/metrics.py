import glob
import os
from typing import List, Dict
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F


def read_jsonl(file_path: str) -> List[Dict]:
    """
    Reads a JSONL file and returns a list of dictionaries.

    :param file_path: Path to the JSONL file.
    :return: List of dictionaries parsed from the JSONL file.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def write_jsonl(data: List[Dict], file_path: str):
    """
    Writes a list of dictionaries to a JSONL file.

    :param data: List of dictionaries to write.
    :param file_path: Path to the output JSONL file.
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def get_metric_one(pred_diag: List[Dict], gt_diag: List[Dict]):
    pred_codes = list({pred['code'] for pred in pred_diag[:10]})
    gt_codes = [gt['code'] for gt in gt_diag]
    primary_diag_code = next((gt['code'] for gt in gt_diag if gt['primary']), None)
    if primary_diag_code is None:
        raise Exception('No primary diag code', gt_diag)

    final_res = {}
    for top_k in [5, 10]:
        subset = pred_codes[:top_k]
        contains_primary = int(primary_diag_code in subset)

        full_coverage = int(set(gt_codes).issubset(subset))

        final_res[f'top_{top_k}'] = {
            'contains_primary': contains_primary,
            'gt_full_coverage': full_coverage
        }

    return final_res

def embed_texts(texts: List[str],
                tokenizer,
                model,
                max_length: int = 25) -> torch.Tensor:
    """
    批量编码文本并返回 CLS 向量 (N, hidden_size)。
    自动将输入搬到 model 的 device。
    """
    device = next(model.parameters()).device
    enc = tokenizer.batch_encode_plus(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    # 转到同一设备
    for k, v in enc.items():
        enc[k] = v.to(device)
    with torch.no_grad():
        out = model(**enc).last_hidden_state  # (N, L, H)
    cls_rep = out[:, 0, :]  # 取 CLS 位置
    return cls_rep.cpu()

def get_sim_metric_one(pred_diag: List[Dict], gt_diag: List[Dict], tokenizer, model, threshold=0.85):
    preds = pred_diag[:10]
    gts   = [d.get("corrected_term") or d["original_term"] for d in gt_diag]
    primary = next((d.get("corrected_term") or d["original_term"] for d in gt_diag if d.get("primary")), None)
    if primary is None:
        raise Exception("No primary diag term", gt_diag)
    if len(preds) == 0:
        return {
            "top_5": {"contains_primary": 0, "gt_full_coverage": 0},
            "top_10": {"contains_primary": 0, "gt_full_coverage": 0}
        }
    # 计算向量
    emb_preds = embed_texts(preds, tokenizer, model)  # (<=10, H)
    emb_gts = embed_texts(gts, tokenizer, model)  # (M, H)
    emb_pri = embed_texts([primary], tokenizer, model)  # (1, H)

    final_res = {}
    for top_k in [5, 10]:
        sub_emb = emb_preds[:top_k]  # (top_k, H)
        # 主诊断最大相似度
        sim_pri = F.cosine_similarity(sub_emb, emb_pri, dim=1).max().item()
        contains_primary = int(sim_pri >= threshold)
        # 全覆盖：每个 gt 是否有 pred 相似度 >= threshold
        sims = F.cosine_similarity(
            sub_emb.unsqueeze(1), emb_gts.unsqueeze(0), dim=-1
        )  # (top_k, M)
        match_per_gt = (sims >= threshold).any(0)
        full_coverage = int(match_per_gt.all().item())

        final_res[f"top_{top_k}"] = {
            "contains_primary": contains_primary,
            "gt_full_coverage": full_coverage
        }
    return final_res

def get_metrics(pred_file, ori_file, output_dir, model, tokenizer):
    preds = read_jsonl(pred_file)
    gts = read_jsonl(ori_file)
    output_file = os.path.join(output_dir, str(os.path.basename(pred_file)).replace('.jsonl', '_metric.jsonl'))
    res = []

    threshold = 0.85  # 相似度阈值
    for pred in preds:
        pred_pmid = pred['pmid']
        for gt in gts:
            if gt['pmid'] == pred_pmid:
                metric = get_metric_one(pred['standardized_pred_diagnosis'], gt['patient_info']['standardized_diagnosis'])
                sim_metric = get_sim_metric_one(pred['pred_diagnoses'], gt['patient_info']['standardized_diagnosis'], tokenizer, model, threshold)
                pred['metrics'] = metric
                pred['sim_metrics'] = sim_metric
                pred['department'] = gt['classification']
                res.append(pred)
    write_jsonl(res, output_file)


def run_metrics(pred_dir, benchmark_file, output_dir):
    """
    Main function to calculate metrics for ICD-11 predictions.

    :param pred_dir: Directory containing the prediction files.
    :param benchmark_file: Path to the benchmark file.
    :param output_dir: Directory to save the output metrics file.
    """
    pred_files = glob.glob(os.path.join(pred_dir, '*.jsonl'))
    # 全局加载模型和分词器
    tokenizer = AutoTokenizer.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
    model = AutoModel.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext").to('cuda' if torch.cuda.is_available() else 'cpu')
    for pred_file in tqdm(pred_files, desc="Processing prediction files"):
        get_metrics(pred_file, benchmark_file, output_dir, model, tokenizer)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Calculate metrics for ICD-11 predictions.")
    parser.add_argument('--pred_dir', type=str, required=True, help='Path to the model prediction JSONL file.')
    parser.add_argument('--ori_file', type=str, required=True, help='Path to the original data JSONL file.')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to save the output metrics JSONL file.')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    run_metrics(args.pred_dir, args.ori_file, args.output_dir)