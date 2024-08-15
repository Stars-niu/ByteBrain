# 导入所需的库
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import streamlit as st
from modelscope import snapshot_download
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

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

# 源大模型下载
model_dir = snapshot_download('IEITYuan/Yuan2-2B-July-hf', cache_dir='./')

# 定义模型路径
path = './IEITYuan/Yuan2-2B-July-hf'

# 定义模型数据类型
torch_dtype = torch.bfloat16 # A10

# 定义一个函数，用于获取模型和tokenizer
@st.cache_resource
def get_model():
    print("创建tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(path, add_eos_token=False, add_bos_token=False, eos_token='<eod>')
    tokenizer.add_tokens(['<sep>', '<pad>', '<mask>', '<predict>', '<FIM_SUFFIX>', '<FIM_PREFIX>', '<FIM_MIDDLE>','<commit_before>','<commit_msg>','<commit_after>','<jupyter_start>','<jupyter_text>','<jupyter_code>','<jupyter_output>','<empty_output>'], special_tokens=True)

    print("创建模型...")
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch_dtype, trust_remote_code=True).cuda()

    print("完成.")
    return tokenizer, model

# 加载model和tokenizer
tokenizer, model = get_model()

# 加载知识库
@st.cache_resource
def load_knowledge_base(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        knowledge_base = f.readlines()
    return knowledge_base

knowledge_base = load_knowledge_base('knowledge.txt')

# 初始化TF-IDF向量器
vectorizer = TfidfVectorizer()
vectorizer.fit(knowledge_base)

# 检索函数
def retrieve_relevant_knowledge(query, knowledge_base, vectorizer, top_k=5):
    query_vec = vectorizer.transform([query])
    knowledge_vecs = vectorizer.transform(knowledge_base)
    similarities = cosine_similarity(query_vec, knowledge_vecs).flatten()
    relevant_indices = similarities.argsort()[-top_k:][::-1]
    relevant_knowledge = [knowledge_base[i] for i in relevant_indices]
    return "\n".join(relevant_knowledge)

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
    relevant_knowledge = retrieve_relevant_knowledge(prompt, knowledge_base, vectorizer)

    # 调用模型
    combined_prompt = f"用户问题: {prompt}\n相关知识:\n{relevant_knowledge}\n回答:"
    inputs = tokenizer(combined_prompt, return_tensors="pt")["input_ids"].cuda()
    outputs = model.generate(inputs, do_sample=False, max_length=1024) # 设置解码方式和最大生成长度
    output = tokenizer.decode(outputs[0])
    response = output.split("回答:")[-1].replace("<eod>", '')

    # 将模型的输出添加到session_state中的messages列表中
    st.session_state.messages.append({"role": "assistant", "content": response})

    # 在聊天界面上显示模型的输出
    st.chat_message("assistant").write(response)
