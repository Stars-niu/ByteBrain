import os
import json
import requests
from bs4 import BeautifulSoup
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling

# 定义下载函数
def download_file(url, dest_path):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(dest_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

# 获取文件下载链接
def get_download_links(base_url):
    response = requests.get(base_url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')
    links = {}
    for a in soup.find_all('a', href=True):
        if 'files' in a['href']:
            filename = a['href'].split('/')[-1]
            links[filename] = base_url + a['href']
    return links

# 基础URL
base_url = "https://www.modelscope.cn/models/IEITYuan/Yuan2-2B-Mars-hf/files"

# 获取下载链接
model_files = get_download_links(base_url)

# 本地模型路径
model_path = './local_model'
os.makedirs(model_path, exist_ok=True)

# 下载模型文件
for filename, url in model_files.items():
    dest_path = os.path.join(model_path, filename)
    if not os.path.exists(dest_path):
        print(f"Downloading {filename}...")
        download_file(url, dest_path)
    else:
        print(f"{filename} already exists, skipping download.")

# 加载数据
with open('finetune_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 准备数据集
class QADataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        question = item['question']
        answer = item['answer']
        prompt = f"问题：{question}\n回答：{answer}<sep>"
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
        inputs['labels'] = inputs.input_ids.clone()
        return inputs

# 从本地路径加载预训练的分词器和模型
tokenizer = AutoTokenizer.from_pretrained(model_path, add_eos_token=False, add_bos_token=False, eos_token='<eod>', legacy=False)
tokenizer.add_tokens(['<sep>', '<pad>', '<mask>', '<predict>', '<FIM_SUFFIX>', '<FIM_PREFIX>', '<FIM_MIDDLE>','<commit_before>','<commit_msg>','<commit_after>','<jupyter_start>','<jupyter_text>','<jupyter_code>','<jupyter_output>','<empty_output>'], special_tokens=True)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)

# 创建数据集和数据加载器
dataset = QADataset(data, tokenizer)
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# 设置训练参数
training_args = TrainingArguments(
    output_dir='./finetuned_model',
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,
    fp16=True if torch.cuda.is_available() else False,
)

# 创建 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

# 开始训练
trainer.train()

# 保存微调后的模型
trainer.save_model('./finetuned_model')
tokenizer.save_pretrained('./finetuned_model')

# 确认模型保存成功
print("Model and tokenizer saved to './finetuned_model'")
