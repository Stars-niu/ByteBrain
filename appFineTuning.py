# 导入所需的库
import streamlit as st
from modelscope import snapshot_download
from typing import List
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForSeq2Seq
import os
from datasets import load_dataset, Dataset

# 设置CUDA环境变量
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

# 添加自定义CSS样式
st.markdown(
    """
    <style>
    .main {
        background-image: url('Background.png');
        background-size: cover;
        background-position: center;
        padding: 10px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: blue;
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

    def get_embeddings(self, texts: List) -> List[float]:
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

# 定义大语言模型类
class LLM:
    """
    class for Yuan2.0 LLM
    """

    def __init__(self, model_path: str) -> None:
        print("Create tokenizer...")
        # 加载预训练的分词器
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, add_eos_token=False, add_bos_token=False, eos_token='<eod>')
        self.tokenizer.add_tokens(['<sep>', '<pad>', '<mask>', '<predict>', '<FIM_SUFFIX>', '<FIM_PREFIX>', '<FIM_MIDDLE>','<commit_before>','<commit_msg>','<commit_after>','<jupyter_start>','<jupyter_text>','<jupyter_code>','<jupyter_output>','<empty_output>'], special_tokens=True)

        print("Create model...")
        # 加载预训练的语言模型
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
        if torch.cuda.is_available():
            self.model = self.model.cuda()

        print(f'Loading Yuan2.0 model from {model_path}.')

    def generate(self, question: str, context: List):
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
        max_length = 2048  # 增大最大长度限制
        if inputs.shape[1] > max_length:
            inputs = inputs[:, -max_length:]

        # 生成输出
        outputs = self.model.generate(inputs, do_sample=False, max_new_tokens=512)  # 增大最大生成长度
        output = self.tokenizer.decode(outputs[0])

        # 移除不需要的字符
        output = output.split("<sep>")[-1].replace("<eod>", "").strip()

        return output

print("> Create Yuan2.0 LLM...")
llm = LLM(llm_model_dir)

# 微调大模型
def fine_tune_model(model, tokenizer, train_dataset):
    # 定义训练参数
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=2,
        save_steps=10_000,
        save_total_limit=2,
        logging_dir='./logs',
    )

    # 定义数据整理器
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # 定义Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    # 开始训练
    trainer.train()

# 加载微调数据
def load_fine_tune_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            question, answer = line.strip().split('\t')
            data.append({'question': question, 'answer': answer})
    return Dataset.from_list(data)

# 加载微调数据
fine_tune_data_path = './fine_tune_data.txt'
fine_tune_data = load_fine_tune_data(fine_tune_data_path)

# 微调模型
fine_tune_model(llm.model, llm.tokenizer, fine_tune_data)

# 初次运行时，session_state中没有"messages"，需要创建一个空列表
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# 使用分栏布局
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    # 每次对话时，都需要遍历session_state中的所有消息，并显示在聊天界面上
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='fixed-right'>", unsafe_allow_html=True)
    st.image("logo.png", caption="ByteBrain Logo", width=150)
    st.markdown("ByteBrain——一个智能知识助手，旨在帮助用户快速获取信息和解决问题。")
    st.markdown("### 联系我们")
    st.markdown("如果您有任何问题或建议，请通过以下方式联系我们：")
    st.markdown("- 邮箱: support@bytebrain.com")
    st.markdown("- 电话: 520-1314")
    st.markdown("</div>", unsafe_allow_html=True)

# 聊天输入框放在col1之外
prompt = st.chat_input("请输入您的问题:")

if prompt:
    # 将用户的输入添加到session_state中的messages列表中
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 在聊天界面上显示用户的输入
    st.chat_message("user").write(prompt)

    # 检索相关知识
    context = index.query(prompt)

    # 调用模型
    response = llm.generate(prompt, context)

    # 将模型的输出添加到session_state中的messages列表中
    st.session_state.messages.append({"role": "assistant", "content": response})

    # 在聊天界面上显示模型的输出
    st.chat_message("assistant").write(response)
