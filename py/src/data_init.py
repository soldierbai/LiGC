import csv
import json
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import *
from es import *

tokenizer = AutoTokenizer.from_pretrained("../chinese-roberta-wwm-ext")
model_bert = AutoModelForMaskedLM.from_pretrained("../chinese-roberta-wwm-ext", output_hidden_states=True)

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

    print(data[0])
    with open(json_file_path, mode='w', encoding='gbk') as jsonfile:
        json.dump(data, jsonfile, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    csv_to_json(file_path, file_path_json)