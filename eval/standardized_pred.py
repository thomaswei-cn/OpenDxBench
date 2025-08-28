import os
from tqdm import tqdm
from icd_api import try_icd_encoding
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed



def get_todo_data(input_path, output_path):
    """
    从输入 JSONL 文件中提取待处理数据，并写入输出 JSONL 文件。

    :param input_path: 输入 JSONL 文件路径
    :param output_path: 输出 JSONL 文件路径
    """
    data = read_jsonl(input_path)
    if os.path.exists(output_path):
        parsed_data = read_jsonl(output_path)
    else:
        parsed_data = []
    parsed_pmid = [item['pmid'] for item in parsed_data]
    todo_data = []
    for item in data:
        pmid = item['pmid']
        if pmid not in parsed_pmid:
            todo_data.append(item)
    return todo_data

def read_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def write_jsonl_append(data, file_path):
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False) + '\n')

def write_jsonl(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def correct_pred_icd_11_none(pred_file, output_dir):
    """
    对存在 code=None 的条目重新尝试 ICD-11 编码，合并所有结果并写入文件。
    """
    os.makedirs(output_dir, exist_ok=True)
    basename = str(os.path.basename(pred_file)).replace('.jsonl', '_icd.jsonl')
    output_path = os.path.join(output_dir, basename)

    pred_list = read_jsonl(pred_file)
    if os.path.exists(output_path):
        processed_list = read_jsonl(output_path)
    else:
        processed_list = []
    processed_pmids = {item['pmid'] for item in processed_list}

    for pred in tqdm(pred_list, desc="修正 ICD-11 编码"):
        pmid = pred['pmid']
        if pmid not in processed_pmids:
            # 新条目，生成 standardized_pred_diagnosis
            std = []
            for term in pred['pred_diagnoses']:
                term = term.strip("**")
                record = {"original_term": term}
                icd = try_icd_encoding(term)
                if icd:
                    record.update(code=icd['code'], title=icd['title'], chapter=icd['chapter'])
                else:
                    record.update(code=None, title=None, chapter=None)
                std.append(record)
            pred['standardized_pred_diagnosis'] = std
            processed_list.append(pred)
            processed_pmids.add(pmid)
        else:
            # 已存在条目，仅更新 code=None 的项
            for proc in processed_list:
                if proc['pmid'] == pmid:
                    for item in proc.get('standardized_pred_diagnosis', []):
                        if item.get('code') is None:
                            icd = try_icd_encoding(item['original_term'].strip("**"))
                            if icd:
                                item.update(code=icd['code'], title=icd['title'], chapter=icd['chapter'])
                    break

    write_jsonl(processed_list, output_path)
    return output_path



def get_pred_icd_11_parallel(pred_file, output_dir, max_workers=8):
    """
    并行获取 ICD-11 编码，使用全局进度条并发写入输出文件。
    """
    output_path = os.path.join(output_dir, str(os.path.basename(pred_file)).replace('.jsonl', '_icd.jsonl'))
    os.makedirs(output_dir, exist_ok=True)

    data = get_todo_data(pred_file, output_path)
    total = len(data)

    pbar = tqdm(total=total, desc="Doing ICD encoding")
    pbar_lock = threading.Lock()
    write_lock = threading.Lock()

    def process_item(item):
        preds = item['pred_diagnoses']
        standard = []
        for pred in preds:
            pred = pred.strip("**")
            st_pred = {"original_term": pred}
            icd = try_icd_encoding(pred)
            if icd:
                st_pred.update(code=icd['code'], title=icd['title'], chapter=icd['chapter'])
            else:
                st_pred.update(code=None, title=None, chapter=None)
            standard.append(st_pred)
        item['standardized_pred_diagnosis'] = standard

        with write_lock, open(output_path, 'a', encoding='utf-8') as fout:
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")

        with pbar_lock:
            pbar.update(1)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_item, itm) for itm in data]
        for _ in as_completed(futures):
            pass

    pbar.close()
    return output_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Get ICD-11 predictions from the model output.")
    parser.add_argument('--pred_file', type=str, help='Path to the model prediction JSONL file.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the output JSONL file with ICD-11 predictions.')
    parser.add_argument('--max_workers', type=int, default=8, help='Number of parallel workers for processing ICD-11 predictions.')
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("\033[92m🔖 开始 ICD-11 编码...\033[0m")
    get_pred_icd_11_parallel(args.pred_file, args.output_dir, max_workers=args.max_workers)
    correct_pred_icd_11_none(args.pred_file, args.output_dir)

