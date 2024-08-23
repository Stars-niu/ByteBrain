import streamlit as st
from modelscope import snapshot_download
from typing import List, Dict
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import os
import json
from datasets import load_dataset, Dataset

# 设置 CUDA 环境变量
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# 设置页面配置
st.set_page_config(
    page_title="ByteBrain",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="auto",
    menu_items={
        "Get Help": "https://bytebrain.com/help",
        "Report a bug": "https://bytebrain.com/bug-report",
        "About": "https://bytebrain.com/about"
    }
)

# 添加自定义 CSS 样式
st.markdown(
    """
    <style>
    body {
        background-image: url('Background.png');
        background-size: cover;
        background-position: center;
        padding: 10px;
        color: #fff;
    }
  .main {
        padding: 10px;
    }
  .stButton>button {
        background-color: #4CAF50;
        color: #fff;
        border: none;
        padding: 10px 24px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 12px;
    }
  .stTextInput>div>div>input {
        border: 2px solid #4CAF50;
        border-radius: 12px;
        padding: 10px;
        background-color: #fff;
        color: #000;
    }
  .fixed-right {
        position: fixed;
        top: 20px;
        right: 20px;
        width: 30%;
    }
  .chat-container {
        width: 100%;
        max-height: 70vh;
        overflow-y: auto;
        padding-right: 20px;
    }
  .stChatMessage {
        width: 100%;
    }
  .answer-box {
        background-color: rgba(0, 0, 0, 0.5);
        padding: 15px;
        border-radius: 12px;
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# 创建一个标题和一个副标题
st.markdown(
    '<span style="font-size: 60px">✨ ByteBrain</span>&nbsp;&nbsp;<span style="font-size: 24px">——计算机科学智能知识助手</span>',
    unsafe_allow_html=True
)

# 向量模型下载
embed_model_dir = snapshot_download("AI-ModelScope/bge-small-zh-v1.5", cache_dir='.')
# 源大模型下载
llm_model_dir = snapshot_download('IEITYuan/Yuan2-2B-Mars-hf', cache_dir='.')

# 定义向量模型类
class EmbeddingModel:
    """
    class for EmbeddingModel
    """

    def __init__(self, path: str) -> None:
        # 加载预训练的分词器和模型
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModel.from_pretrained(path)
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        print(f'Loading EmbeddingModel from {path}.')

    def get_embeddings(self, texts: List[str]) -> List[float]:
        """
        calculate embedding for text list
        """
        # 对输入文本进行编码
        encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        if torch.cuda.is_available():
            encoded_input = {k: v.cuda() for k, v in encoded_input.items()}
        with torch.no_grad():
            # 获取模型输出
            model_output = self.model(**encoded_input)
            sentence_embeddings = model_output[0][:, 0]
        # 对嵌入向量进行归一化
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings.tolist()

print("> Create embedding model...")
embed_model = EmbeddingModel(embed_model_dir)

# 定义向量库索引类
class VectorStoreIndex:
    """
    class for VectorStoreIndex
    """

    def __init__(self, document_path: str, embed_model: EmbeddingModel) -> None:
        # 加载文档
        self.documents = []
        for line in open(document_path, 'r', encoding='utf-8'):
            line = line.strip()
            self.documents.append(line)

        self.embed_model = embed_model
        # 获取文档的嵌入向量
        self.vectors = self.embed_model.get_embeddings(self.documents)

        print(f'Loading {len(self.documents)} documents for {document_path}.')

    def get_similarity(self, vector1: List[float], vector2: List[float]) -> float:
        """
        calculate cosine similarity between two vectors
        """
        dot_product = np.dot(vector1, vector2)
        magnitude = np.linalg.norm(vector1) * np.linalg.norm(vector2)
        if not magnitude:
            return 0
        return dot_product / magnitude

    def query(self, question: str, k: int = 1) -> List[str]:
        # 获取问题的嵌入向量
        question_vector = self.embed_model.get_embeddings([question])[0]
        # 计算问题向量与文档向量的相似度
        result = np.array([self.get_similarity(question_vector, vector) for vector in self.vectors])
        # 返回相似度最高的文档
        return np.array(self.documents)[result.argsort()[-k:][::-1]].tolist()

print("> Create index...")
document_path = './knowledge.txt'
index = VectorStoreIndex(document_path, embed_model)

# 定义大语言模型类（包含微调功能）
class LLM:
    """
    class for Yuan2.0 LLM
    """

    def __init__(self, model_path: str) -> None:
        print("Create tokenizer...")
        # 加载预训练的分词器
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, add_eos_token=False, add_bos_token=False, eos_token='<eod>', legacy=False)
        self.tokenizer.add_tokens(['<sep>', '<pad>', '<mask>', '<predict>', '<FIM_SUFFIX>', '<FIM_PREFIX>', '<FIM_MIDDLE>','<commit_before>','<commit_msg>','<commit_after>','<jupyter_start>','<jupyter_text>','<jupyter_code>','<jupyter_output>','<empty_output>'], special_tokens=True)

        print("Create model...")
        # 加载预训练的语言模型
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
        if torch.cuda.is_available():
            self.model = self.model.cuda()

        print(f'Loading Yuan2.0 model from {model_path}.')

    def generate(self, question: str, context: List[str]) -> str:
        # 构建提示词
        if context:
            prompt = f'背景：{context}\n问题：{question}\n请基于背景，回答问题。'
        else:
            prompt = question

        prompt += "<sep>"
        inputs = self.tokenizer(prompt, return_tensors="pt")["input_ids"]
        if torch.cuda.is_available():
            inputs = inputs.cuda()

        # 截断输入以确保长度不超过最大长度
        max_length = 2048
        if inputs.shape[1] > max_length:
            inputs = inputs[:, -max_length:]

        # 检查并转换输入数据类型
        if inputs.dtype == torch.bfloat16:
            print(f"Input data type before conversion: {inputs.dtype}")
            inputs = inputs.to(torch.long)
            print(f"Input data type after conversion: {inputs.dtype}")

        # 生成输出时确保模型和输入数据的数据类型一致
        if torch.cuda.is_available():
            self.model.to(torch.bfloat16)
        inputs = inputs.to(torch.bfloat16)

        outputs = self.model.generate(inputs, do_sample=False, max_new_tokens=512)
        output = self.tokenizer.decode(outputs[0])

        # 移除不需要的字符
        output = output.split("<sep>")[-1].replace("<eod>", "").strip()

        return output

    def fine_tune(self, data_path: str) -> None:
        # 加载微调数据
        dataset = load_dataset('json', data_files=data_path, split='train')

        def preprocess_function(examples):
            inputs = examples['input_text']
            targets = examples['output_text']
            model_inputs = self.tokenizer(inputs, max_length=512, truncation=True)

            # Setup the tokenizer for targets
            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(targets, max_length=512, truncation=True)

            model_inputs['labels'] = labels['input_ids']
            return model_inputs

        tokenized_dataset = dataset.map(preprocess_function, batched=True)
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)

        training_args = TrainingArguments(
            output_dir='./results',
            overwrite_output_dir=True,
            num_train_epochs=3,
            per_device_train_batch_size=4,
            save_steps=10_000,
            save_total_limit=2,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
        )

        trainer.train()

print("> Create Yuan2.0 LLM model...")
llm = LLM(llm_model_dir)

# Streamlit 界面
def main():
    st.title("ByteBrain - 智能知识助手")

    # 用户输入
    question = st.text_input("请输入您的问题：")

    if st.button("查询"):
        if question:
            # 向量库检索
            st.write("正在检索相关文档...")
            context = index.query(question, k=3)
            st.write("检索到的相关文档：", context)

            # 大语言模型生成回答
            st.write("正在生成回答...")
            answer = llm.generate(question, context)
            with st.container():
                st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)
        else:
            st.write("请输入问题后再点击查询按钮。")

if __name__ == "__main__":
    main()
