import os
import json
import requests
from bs4 import BeautifulSoup
from peft import PeftModel
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling


# 定义模型路径
model_download_path = './local_model'
model_finetune_path = './local_model/IEITYuan/Yuan2-2B-Mars-hf'
lora_path = './output/Yuan2.0-2B_lora_bf16/checkpoint-51'

# 定义模型数据类型
torch_dtype = torch.bfloat16  # A10
# torch_dtype = torch.float16  # P100

# 检查模型文件是否存在
required_files = ["pytorch_model.bin", "model.safetensors", "tf_model.h5", "model.ckpt.index", "flax_model.msgpack", "config.json"]
model_files_exist = any(os.path.isfile(os.path.join(model_download_path, file)) for file in required_files)

if not model_files_exist:
    # 源大模型下载并直接保存到 local_model 文件夹
    os.makedirs(model_download_path, exist_ok=True)
    from modelscope import snapshot_download
    try:
        model_dir = snapshot_download('IEITYuan/Yuan2-2B-Mars-hf', cache_dir=model_download_path)
    except Exception as e:
        print(f"Error downloading model: {e}")
        exit(1)
else:
    print("Model files are already present.")

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

# 从调整后的路径加载预训练的分词器和模型
tokenizer = AutoTokenizer.from_pretrained(model_finetune_path, add_eos_token=False, add_bos_token=False, eos_token='<eod>', legacy=False)
tokenizer.add_tokens(['<sep>', '<pad>', '<mask>', '<predict>', '<FIM_SUFFIX>', '<FIM_PREFIX>', '<FIM_MIDDLE>','<commit_before>','<commit_msg>','<commit_after>','<jupyter_start>','<jupyter_text>','<jupyter_code>','<jupyter_output>','<empty_output>'], special_tokens=True)
model = AutoModelForCausalLM.from_pretrained(model_finetune_path, torch_dtype=torch.bfloat16, trust_remote_code=True)

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
try:
    trainer.train()
except Exception as e:
    print(f"Error during training: {e}")
    exit(1)

# 保存微调后的模型
try:
    trainer.save_model('./finetuned_model')
    tokenizer.save_pretrained('./finetuned_model')
except Exception as e:
    print(f"Error saving model: {e}")
    exit(1)

# 确认模型保存成功
print("Model and tokenizer saved to './finetuned_model'")
