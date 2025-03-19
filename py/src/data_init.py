import csv
import json
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from concurrent.futures import ThreadPoolExecutor, as_completed
from .config import data_path, data_path_json, bert_model_path


tokenizer = AutoTokenizer.from_pretrained(bert_model_path)
model_bert = AutoModelForMaskedLM.from_pretrained(bert_model_path, output_hidden_states=True)


def get_embedding(text, tokenizer, model_bert):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256)

    with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.float16):
        outputs = model_bert(**inputs)
        last_hidden_state = outputs.hidden_states[-1]

        # 只取 CLS token 的嵌入
        cls_embedding = last_hidden_state[:, 0, :].cpu().numpy()
        cls_embedding = cls_embedding.reshape(1, 1, cls_embedding.shape[1])  # Reshape

    return cls_embedding.tolist()[0][0]


def process_row(args):
    row, tokenizer, model_bert = args
    row["CR Vector"] = get_embedding(row["CR主题"])
    return row


def csv_to_json(csv_file_path, json_file_path):
    data = []
    with open(csv_file_path, mode='r', encoding='gbk') as csvfile:
        print('打开文件')
        csv_reader = csv.DictReader(csvfile)
        rows = list(csv_reader)
        total_rows = len(rows)

        # 创建线程池
        with ThreadPoolExecutor(max_workers=5) as executor:
            progress = tqdm(total=total_rows, desc="处理进度", unit="行")
            futures = {executor.submit(process_row, (row, tokenizer, model_bert)): row for row in rows}
            for future in as_completed(futures):
                result = future.result()
                data.append(result)
                progress.update(1)

    with open(json_file_path, mode='w', encoding='gbk') as jsonfile:
        json.dump(data, jsonfile, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    csv_to_json(data_path, data_path_json)
