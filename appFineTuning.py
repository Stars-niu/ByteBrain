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

# 添加自定义 CSS 样式，美化页面
st.markdown(
    """
    <style>
    body {
        background-color: #282c34; /* 深灰色背景 */
        color: #fff; /* 白色文字 */
        font-family: Arial, sans-serif; /* 字体 */
    }
.main {
        padding: 20px;
    }
.stButton>button {
        background-color: #61dafb; /* 浅蓝色按钮背景 */
        color: #000; /* 黑色文字 */
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
        border: 2px solid #61dafb; /* 浅蓝色边框 */
        border-radius: 12px;
        padding: 10px;
        background-color: #fff; /* 白色输入框背景 */
        color: #000; /* 黑色文字 */
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
    h1 {
        color: #61dafb; /* 浅蓝色标题 */
    }
    h2 {
        color: #98c379; /* 浅绿色副标题 */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# 创建一个标题和一个副标题
st.markdown(
    '<h1>✨ ByteBrain</h1><h2>——计算机科学智能知识助手</h2>',
    unsafe_allow_html=True
)

# 向量模型下载
try:
    embed_model_dir = snapshot_download("AI-ModelScope/bge-small-zh-v1.5", cache_dir='.')
    print(f'Successfully loaded embedding model from cache.')
except Exception as e:
    st.error(f'Error loading embedding model: {e}')
    embed_model_dir = None

# 源大模型下载
try:
    llm_model_dir = snapshot_download('IEITYuan/Yuan2-2B-Mars-hf', cache_dir='.')
    print(f'Successfully loaded LLM model from cache.')
except Exception as e:
    st.error(f'Error loading LLM model: {e}')
    llm_model_dir = None

# 定义向量模型类
class EmbeddingModel:
    """
    class for EmbeddingModel
    """

    def __init__(self, path: str) -> None:
        try:
            # 加载预训练的分词器和模型
            self.tokenizer = AutoTokenizer.from_pretrained(path)
            self.model = AutoModel.from_pretrained(path)
            if torch.cuda.is_available():
                self.model = self.model.cuda()
            print(f'Loading EmbeddingModel from {path}.')
        except Exception as e:
            print(f'Error initializing EmbeddingModel: {e}')

    def get_embeddings(self, texts: List[str]) -> List[float]:
        """
        calculate embedding for text list
        """
        try:
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
        except Exception as e:
            print(f'Error in get_embeddings: {e}')
            return []

print("> Create embedding model...")
if embed_model_dir:
    embed_model = EmbeddingModel(embed_model_dir)
else:
    st.error('Embedding model not loaded. Cannot continue.')
    embed_model = None

# 定义向量库索引类
class VectorStoreIndex:
    """
    class for VectorStoreIndex
    """

    def __init__(self, document_path: str, embed_model: EmbeddingModel) -> None:
        try:
            # 加载文档
            self.documents = []
            for line in open(document_path, 'r', encoding='utf-8'):
                line = line.strip()
                self.documents.append(line)

            self.embed_model = embed_model
            # 获取文档的嵌入向量
            self.vectors = self.embed_model.get_embeddings(self.documents)

            print(f'Loading {len(self.documents)} documents for {document_path}.')
        except Exception as e:
            print(f'Error initializing VectorStoreIndex: {e}')

    def get_similarity(self, vector1: List[float], vector2: List[float]) -> float:
        """
        calculate cosine similarity between two vectors
        """
        try:
            dot_product = np.dot(vector1, vector2)
            magnitude = np.linalg.norm(vector1) * np.linalg.norm(vector2)
            if not magnitude:
                return 0
            return dot_product / magnitude
        except Exception as e:
            print(f'Error in get_similarity: {e}')
            return 0

    def query(self, question: str, k: int = 1) -> List[str]:
        """
        query the vector store for similar documents
        """
        try:
            # 获取问题的嵌入向量
            question_vector = self.embed_model.get_embeddings([question])[0]
            # 计算问题向量与文档向量的相似度
            result = np.array([self.get_similarity(question_vector, vector) for vector in self.vectors])
            # 返回相似度最高的文档
            return np.array(self.documents)[result.argsort()[-k:][::-1]].tolist()
        except Exception as e:
            print(f'Error in query: {e}')
            return []

print("> Create index...")
document_path = './knowledge.txt'
if embed_model:
    index = VectorStoreIndex(document_path, embed_model)
else:
    st.error('Embedding model not available. Cannot create index.')
    index = None

# 定义大语言模型类（包含微调功能）
class LLM:
    """
    class for Yuan2.0 LLM
    """

    def __init__(self, model_path: str) -> None:
        try:
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
        except Exception as e:
            print(f'Error initializing LLM: {e}')

    def generate(self, question: str, context: List[str]) -> str:
        """
        generate answer using the language model
        """
        try:
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

            # 确保输入数据类型正确
            if inputs.dtype == torch.bfloat16:
                inputs = inputs.to(torch.long)
            elif inputs.dtype!= torch.long:
                st.error(f'Unexpected input data type: {inputs.dtype}. Expected torch.long.')
                return 'Sorry, an error occurred while generating the answer.'

            # 生成输出时确保模型和输入数据的数据类型一致
            if torch.cuda.is_available():
                self.model.to(torch.bfloat16)
            inputs = inputs.to(torch.bfloat16)

            outputs = self.model.generate(inputs, do_sample=False, max_new_tokens=512)
            output = self.tokenizer.decode(outputs[0])

            # 移除不需要的字符
            output = output.split("<sep>")[-1].replace("<eod>", "").strip()

            return output
        except Exception as e:
            print(f'Error in generate: {e}')
            return 'Sorry, an error occurred while generating the answer.'

    def fine_tune(self, data_path: str) -> None:
        """
        fine-tune the language model
        """
        try:
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
                logging_dir='./logs',
                logging_steps=200,
            )

            trainer = Trainer(
                model=self.model,
                args=training_args,
                data_collator=data_collator,
                train_dataset=tokenized_dataset,
            )

            trainer.train()
        except Exception as e:
            print(f'Error in fine_tune: {e}')

print("> Create LLM model...")
if llm_model_dir:
    llm_model = LLM(llm_model_dir)
else:
    st.error('LLM model not loaded. Cannot continue.')
    llm_model = None

# Streamlit 用户界面逻辑
st.sidebar.title("ByteBrain 侧边栏")
user_input = st.sidebar.text_area("请输入你的问题：", "")

if st.sidebar.button("查询"):
    if user_input:
        st.sidebar.markdown("### 查询结果")

# Streamlit 用户界面逻辑
st.sidebar.title("ByteBrain 侧边栏")
user_input = st.sidebar.text_area("请输入你的问题：", "")

if st.sidebar.button("查询"):
    if user_input:
        st.sidebar.markdown("### 查询结果")
        # 查询 VectorStoreIndex
        try:
            context = index.query(user_input)
            st.sidebar.markdown("#### 上下文")
            for doc in context:
                st.sidebar.markdown(f"- {doc}")
        except Exception as e:
            st.sidebar.error(f'查询时出现错误：{e}')
        
        # 使用大语言模型生成回答
        if llm_model:
            try:
                answer = llm_model.generate(user_input, context)
                st.sidebar.markdown("#### 回答")
                st.sidebar.markdown(f"{answer}")
            except Exception as e:
                st.sidebar.error(f'生成回答时出现错误：{e}')
        else:
            st.sidebar.error("语言模型加载失败，无法生成回答。")
    else:
        st.sidebar.warning("请输入你的问题！")

# 主页面
st.title("ByteBrain")
st.write("欢迎使用 ByteBrain，一个计算机科学智能知识助手！请输入你的问题，获取专业的回答。")
