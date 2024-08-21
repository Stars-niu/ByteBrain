import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer
from modelscope import snapshot_download
from peft import LoraConfig, TaskType, get_peft_model

# 下载模型
def download_model(model_name, cache_dir='.'):
    return snapshot_download(model_name, cache_dir=cache_dir)

# 读取数据
def load_data(file_path):
    df = pd.read_json(file_path)
    return Dataset.from_pandas(df)

# 数据处理函数
def process_func(example, tokenizer, max_length=384):
    instruction = tokenizer(f"{example['input']}<sep>")
    response = tokenizer(f"{example['output']}<eod>")
    input_ids = instruction["input_ids"] + response["input_ids"]
    attention_mask = [1] * len(input_ids)
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"]

    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]
        labels = labels[:max_length]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

# 加载和处理数据集
def prepare_dataset(file_path, tokenizer):
    ds = load_data(file_path)
    return ds.map(lambda x: process_func(x, tokenizer), remove_columns=ds.column_names)

# 加载模型和tokenizer
def load_model_and_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, add_eos_token=False, add_bos_token=False, eos_token='<eod>')
    tokenizer.add_tokens(['<sep>', '<pad>', '<mask>', '<predict>', '<FIM_SUFFIX>', '<FIM_PREFIX>', '<FIM_MIDDLE>','<commit_before>','<commit_msg>','<commit_after>','<jupyter_start>','<jupyter_text>','<jupyter_code>','<jupyter_output>','<empty_output>'], special_tokens=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
    model.enable_input_require_grads()

    return model, tokenizer

# 配置Lora
def configure_lora(model):
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1
    )
    return get_peft_model(model, config)

# 训练模型
def train_model(model, tokenizer, dataset, output_dir):
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=12,
        gradient_accumulation_steps=1,
        logging_steps=1,
        save_strategy="epoch",
        num_train_epochs=3,
        learning_rate=5e-5,
        save_on_each_node=True,
        gradient_checkpointing=True,
        bf16=True
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )
    trainer.train()

# 生成函数
def generate(model, tokenizer, prompt):
    prompt = prompt + "<sep>"
    inputs = tokenizer(prompt, return_tensors="pt")["input_ids"].cuda()
    outputs = model.generate(inputs, do_sample=False, max_length=256)
    output = tokenizer.decode(outputs[0])
    print(output.split("<sep>")[-1])

if __name__ == "__main__":
    model_name = 'IEITYuan/Yuan2-2B-Mars-hf'
    data_file = './data.json'
    output_dir = "./output/Yuan2.0-2B_lora_bf16"

    model_path = download_model(model_name)
    model, tokenizer = load_model_and_tokenizer(model_path)
    model = configure_lora(model)
    dataset = prepare_dataset(data_file, tokenizer)
    train_model(model, tokenizer, dataset, output_dir)

    # Example usage
    template = '''
    # 任务描述
    假设你是一个计算机科学智能知识助手，能够回答关于计算机科学领域的问题，并提供详细的解释。

    # 任务要求
    回答应包括以下内容：定义、背景信息、相关技术、应用场景、示例代码（如果适用）。

    # 样例
    输入：
    什么是机器学习？
    输出：
    {"定义": ["机器学习是一种人工智能技术，允许系统在没有明确编程的情况下学习和改进。"], "背景信息": ["机器学习起源于模式识别和计算学习理论。"], "相关技术": ["监督学习、无监督学习、强化学习。"], "应用场景": ["图像识别、语音识别、推荐系统。"], "示例代码": ["from sklearn import datasets\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.linear_model import LogisticRegression\n\n# 加载数据集\niris = datasets.load_iris()\nX = iris.data\ny = iris.target\n\n# 拆分数据集\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)\n\n# 训练模型\nmodel = LogisticRegression()\nmodel.fit(X_train, y_train)\n\n# 预测\npredictions = model.predict(X_test)\nprint(predictions)"]}

    # 当前问题
    input_str

    # 任务重述
    请参考样例，按照任务要求，回答当前问题，并提供详细的解释。
    '''
    input_str = '什么是深度学习？'
    prompt = template.replace('input_str', input_str).strip()
    generate(model, tokenizer, prompt)
