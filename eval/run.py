import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import base64
import re
import ast
import os
from PIL import Image
from request_handler import *
from tqdm import tqdm


def validate_benchmark(benchmark_jsonl):
    """
    验证 benchmark_jsonl 文件的格式是否正确。
    每行应为一个 JSON 对象，包含 'pmid' 和 'patient_info' 字段。
    """
    base_dir = os.path.dirname(benchmark_jsonl)
    if not os.path.isfile(benchmark_jsonl):
        raise ValueError(f"Benchmark file {benchmark_jsonl} does not exist.")
    data = read_jsonl(benchmark_jsonl)
    for item in data:
        if 'pmid' not in item or 'patient_info' not in item:
            raise ValueError(f"Invalid format in benchmark file: {item}")
        patient_info = item['patient_info']
        if 'basic_info' not in patient_info or 'supplementary_info' not in patient_info:
            raise ValueError(f"Invalid patient_info format in benchmark file: {item}")
        sups = patient_info['supplementary_info']
        if not isinstance(sups, list):
            raise ValueError(f"Invalid supplementary_info format in benchmark file: {item}")
        for sup in sups:
            if 'caption' not in sup or 'path' not in sup:
                raise ValueError(f"Invalid supplementary_info item in benchmark file: {sup}")
            if not isinstance(sup['path'], list) or len(sup['path']) != 1:
                raise ValueError(f"Invalid path format in supplementary_info item: {sup}")
            path = sup['path'][0]
            if not os.path.isfile(os.path.join(base_dir, path)):
                error_path = os.path.join(base_dir, path)
                raise ValueError(f"Image file does not exist: {error_path}")

    print(f"Benchmark file {benchmark_jsonl} is valid.")


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
    # sort by pmid for consistency
    todo_data.sort(key=lambda x: x['pmid'])
    return todo_data


def write_jsonl_append(data, file_path):
    """
    将 JSON 对象追加到 JSONL 文件中。

    :param data: 要写入的 JSON 对象
    :param file_path: 输出 JSONL 文件路径
    """
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False) + '\n')



def read_jsonl(file_path):
    """
    读取 JSONL 文件并返回一个列表，每个元素是一个 JSON 对象。
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Error: {e}")
                raise ValueError(f"Error decoding JSON: {line.strip()}")
    return data


# ========== 2. 工具函数：图像编码 ==========
def encode_image_to_base64(image_path):
    """将图像文件编码为 base64 字符串"""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')


# ========== 3. 构建多模态 Prompt ==========
def build_multimodal_prompt(data, base_dir):
    """
    构建 GPT-4o 支持的 user_content 格式：病史 + 图像 + caption + 输出指令。

    输入：
        - history_text: str，主诉和病史
        - image_caption_list: list of (image_path, caption) 元组
    输出：
        - user_content: 可直接用于 ChatCompletion 的多模态输入
    """
    history_text = data['patient_info']['basic_info']
    sups = data['patient_info']['supplementary_info']
    user_content = []

    # 病史部分
    user_content.append({
        "type": "text",
        "text": f"[Chief Complaint and Medical History]\n{history_text}"
    })

    # 每张图片及其 caption
    for idx, sup in enumerate(sups):
        user_content.append({
            "type": "text",
            "text": f"Figure {idx + 1}: {sup['caption']}"
        })
        img_b64 = encode_image_to_base64(os.path.join(base_dir, sup['path'][0]))
        user_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{img_b64}",
                "detail": "high"
            }
        })
    user_content.append({
        "type": "text",
        "text": (
            "Please output your final answer in the following format:\n\n"
            "### Output ###\n[\"Diagnosis A\", \"Diagnosis B\"]"
        )
    })
    system_msg = {
        "role": "system",
        "content": (
            "You are a medical expert. Given the patient’s demographic information, chief complaint, medical history, and results from various examinations, "
            "your task is to identify possible diagnoses. Please enumerate the top 10 most likely diagnoses in order, with the most likely disease listed first."
            "Each item in the list must represent a **single, independent disease**. If the patient has multiple diseases or complications, please list them separately."
            "Output only under the heading '### Output ###'."
        )
    }
    messages = [
        system_msg,
        {"role": "user", "content": user_content}
    ]

    return messages


# ========== 4. 提取诊断列表（使用正则） ==========
def extract_diagnosis_list_from_output(text: str):
    """
    提取 ### Output ### 后的诊断列表。
    支持以下格式：
    1. JSON/AST 列表
    2. 双引号括起的单项匹配
    3. 编号列表 (例如 1. xxx)
    """
    if not isinstance(text, str):
        return None
    text = text.strip('.').strip()

    # 0. 直接解析整个字符串为 JSON/AST 列表
    try:
        lst = json.loads(text)
        if isinstance(lst, list):
            return lst
    except json.JSONDecodeError:
        try:
            lst = ast.literal_eval(text)
            if isinstance(lst, list):
                return lst
        except Exception:
            pass

    # 1. JSON/AST 列表解析
    bracket_pattern = r"### Output ###\s*(\[[\s\S]*?\])"
    match = re.search(bracket_pattern, text)
    if match:
        snippet = match.group(1)
        try:
            return json.loads(snippet)
        except json.JSONDecodeError:
            try:
                result = ast.literal_eval(snippet)
                if isinstance(result, list):
                    return result
            except Exception:
                    pass

        # 2. 引号提取（优先双引号，再尝试单引号）
        parts = text.split("### Output ###", 1)
        if len(parts) < 2:
            return None
        output_part = parts[1]
        quoted = re.findall(r'"([^"]+)"', output_part)
        if not quoted:
            quoted = re.findall(r"'([^']+)'", output_part)
        if quoted:
            return list(dict.fromkeys(quoted))

    # 3. 编号列表提取
    enum_items = []
    for line in output_part.splitlines():
        m = re.match(r"\s*\d+\.\s*(.+)", line)
        if m:
            enum_items.append(m.group(1).strip())
    if enum_items:
        return enum_items

    return None


def make_up_illegal_output(input_jsonl, output_jsonl):
    input_data = read_jsonl(input_jsonl)
    output_data = read_jsonl(output_jsonl)
    input_pmids = [item['pmid'] for item in input_data]
    output_pmids = [item['pmid'] for item in output_data]
    missing_pmids = list(set(input_pmids) - set(output_pmids))
    # 用红色打印提示：在多次尝试后，模型仍有len(missing_pmids)个 PMID 没有预测结果，使用空列表填充。
    if missing_pmids:
        print(f"\033[91mWarning: After multiple attempts, {len(missing_pmids)} PMIDs have no prediction results. Filling with empty lists.\033[0m")
        for pmid in missing_pmids:
            dummy_output = {
                'pmid': pmid,
                'pred_diagnoses': []
            }
            write_jsonl_append(dummy_output, output_jsonl)


def main_eval(input_jsonl, model_name, output_dir, max_retries=3, api_key=None):

    validate_benchmark(input_jsonl)
    # ===2. eval===
    base_dir = os.path.dirname(input_jsonl)
    output_path = os.path.join(output_dir, f'{model_name}_diagnoses.jsonl')
    components = init_components(model_name, api_key)
    for i in range(max_retries):
        print(f'Retry {i + 1}/{max_retries}...')
        todo_data = get_todo_data(input_jsonl, output_path)
        for data in tqdm(todo_data):
            temp_res = {'pmid': data['pmid'], 'pred_diagnoses': None}
            # 构建 prompt
            messages = build_multimodal_prompt(data, base_dir)
            # 请求模型
            result = model(model_name, messages, components)
            # print(result)
            # 输出结果
            diagnosis_list = extract_diagnosis_list_from_output(result)
            if diagnosis_list is not None:
                temp_res['pred_diagnoses'] = diagnosis_list
                write_jsonl_append(temp_res, output_path)
            else:
                print(f"Failed to parse diagnoses for PMID {data['pmid']}, the output was:\n{result}")
    make_up_illegal_output(input_jsonl, output_path)
    # 打印 任务完成 与输出路径
    print(f"Task completed. Results saved to {output_path}")


def main_eval_parallel(input_jsonl, model_name, output_dir, max_retries=3, max_workers=8, api_key=None):
    base_dir = os.path.dirname(input_jsonl)
    validate_benchmark(input_jsonl)
    components = init_components(model_name, api_key)
    output_path = os.path.join(output_dir, f'{model_name}_diagnoses.jsonl')
    for i in range(max_retries):
        print(f'Retry {i + 1}/{max_retries}...')
        todo_data = get_todo_data(input_jsonl, output_path)
        if not todo_data:
            break

        write_lock = threading.Lock()
        def process_item(data):
            temp_res = {'pmid': data['pmid'], 'pred_diagnoses': None}
            messages = build_multimodal_prompt(data, base_dir)
            result = model(model_name, messages, components)
            diags = extract_diagnosis_list_from_output(result)
            if diags is not None:
                temp_res['pred_diagnoses'] = diags
                with write_lock:
                    write_jsonl_append(temp_res, output_path)
            else:
                print(f"Failed to parse diagnoses for PMID {data['pmid']}, the output was:\n{result}")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_item, d) for d in todo_data]
            for _ in tqdm(as_completed(futures), total=len(futures)):
                pass
    make_up_illegal_output(input_jsonl, output_path)
    print(f"Task completed. Results saved to {output_path}")

# ========== 6. 用法示例 ==========
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate medical diagnoses using OpenAI models.")
    parser.add_argument('--benchmark_jsonl', type=str, required=True, help="Path to input JSONL file with patient data.")
    parser.add_argument('--model', type=str, required=True, help="OpenAI model name (e.g., 'gpt-4o').")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save output JSONL file.")
    parser.add_argument('--max_retries', type=int, default=3, help="Number of retries for processing.")
    parser.add_argument('--parallel', action='store_true', help="Use parallel processing for evaluation.")
    parser.add_argument('--max_workers', type=int, default=8, help="Number of parallel workers for processing.")
    parser.add_argument('--api_key', type=str, help="API key for OpenAI models (required for OpenAI models).")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.parallel:
        main_eval_parallel(args.benchmark_jsonl, args.model, args.output_dir, args.max_retries, args.max_workers, args.api_key)
    else:
        main_eval(args.benchmark_jsonl, args.model, args.output_dir, args.max_retries, args.api_key)
