# 查看已安装依赖
! pip list
# 安装 streamlit
! pip install streamlit==1.24.0
！pip install torch
！pip install pandas
！pip install datasets
！pip install transformers
！pip install peft
from modelscope import snapshot_download
model_dir = snapshot_download('IEITYuan/Yuan2-2B-Mars-hf', cache_dir='.')
# 导入环境
import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer
# 读取数据
df = pd.read_json('./data.json')
ds = Dataset.from_pandas(df)
# 查看数据
len(ds)
ds[:1]
# 加载 tokenizer
path = './IEITYuan/Yuan2-2B-Mars-hf'

tokenizer = AutoTokenizer.from_pretrained(path, add_eos_token=False, add_bos_token=False, eos_token='<eod>')
tokenizer.add_tokens(['<sep>', '<pad>', '<mask>', '<predict>', '<FIM_SUFFIX>', '<FIM_PREFIX>', '<FIM_MIDDLE>','<commit_before>','<commit_msg>','<commit_after>','<jupyter_start>','<jupyter_text>','<jupyter_code>','<jupyter_output>','<empty_output>'], special_tokens=True)
tokenizer.pad_token = tokenizer.eos_token
# 定义数据处理函数
def process_func(example):
    MAX_LENGTH = 384    # Llama分词器会将一个中文字切分为多个token，因此需要放开一些最大长度，保证数据的完整性

    instruction = tokenizer(f"{example['input']}<sep>")
    response = tokenizer(f"{example['output']}<eod>")
    input_ids = instruction["input_ids"] + response["input_ids"]
    attention_mask = [1] * len(input_ids) 
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] # instruction 不计算loss

    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }
# 处理数据集
tokenized_id = ds.map(process_func, remove_columns=ds.column_names)
tokenized_id
# 数据检查
tokenizer.decode(tokenized_id[0]['input_ids'])
tokenizer.decode(list(filter(lambda x: x != -100, tokenized_id[0]["labels"])))
# 模型加载
model = AutoModelForCausalLM.from_pretrained(path, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
model
model.enable_input_require_grads() # 开启gradient_checkpointing时，要执行该方法
# 查看模型数据类型
model.dtype
# 配置Lora
from peft import LoraConfig, TaskType, get_peft_model

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False, # 训练模式
    r=8, # Lora 秩
    lora_alpha=32, # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.1# Dropout 比例
)
config
# 构建PeftModel
model = get_peft_model(model, config)
model
# 打印需要训练的参数
model.print_trainable_parameters()
# 设置训练参数
args = TrainingArguments(
    output_dir="./output/Yuan2.0-2B_lora_bf16",
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
# 初始化Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_id,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)
# 模型训练
trainer.train()
# 定义生成函数
def generate(prompt):
    prompt = prompt + "<sep>"
    inputs = tokenizer(prompt, return_tensors="pt")["input_ids"].cuda()
    outputs = model.generate(inputs, do_sample=False, max_length=256)
    output = tokenizer.decode(outputs[0])
    print(output.split("<sep>")[-1])
# 输入prompt template
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
generate(prompt)
