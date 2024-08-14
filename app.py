# 导入所需的库
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import streamlit as st
from modelscope import snapshot_download

# 设置页面配置
st.set_page_config(
    page_title="ByteBrain",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 添加自定义CSS样式
st.markdown(
    """
    <style>
    .main {
        background-color: #f0f2f6;
        padding: 20px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
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
    </style>
    """,
    unsafe_allow_html=True,
)

# 创建一个标题和一个副标题
st.title("✨ ByteBrain")
st.subheader("智能知识助手")

# 源大模型下载
model_dir = snapshot_download('IEITYuan/Yuan2-2B-Mars-hf', cache_dir='./')
# model_dir = snapshot_download('IEITYuan/Yuan2-2B-July-hf', cache_dir='./')

# 定义模型路径
path = './IEITYuan/Yuan2-2B-Mars-hf'
# path = './IEITYuan/Yuan2-2B-July-hf'

# 定义模型数据类型
torch_dtype = torch.bfloat16 # A10
# torch_dtype = torch.float16 # P100

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

# 初次运行时，session_state中没有"messages"，需要创建一个空列表
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# 使用分栏布局
col1, col2 = st.columns([3, 1])

with col1:
    # 每次对话时，都需要遍历session_state中的所有消息，并显示在聊天界面上
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # 如果用户在聊天输入框中输入了内容，则执行以下操作
    if prompt := st.chat_input("请输入您的问题:"):
        # 将用户的输入添加到session_state中的messages列表中
        st.session_state.messages.append({"role": "user", "content": prompt})

        # 在聊天界面上显示用户的输入
        st.chat_message("user").write(prompt)

        # 调用模型
        prompt = "<n>".join(msg["content"] for msg in st.session_state.messages) + "<sep>" # 拼接对话历史
        inputs = tokenizer(prompt, return_tensors="pt")["input_ids"].cuda()
        outputs = model.generate(inputs, do_sample=False, max_length=1024) # 设置解码方式和最大生成长度
        output = tokenizer.decode(outputs[0])
        response = output.split("<sep>")[-1].replace("<eod>", '')

        # 将模型的输出添加到session_state中的messages列表中
        st.session_state.messages.append({"role": "assistant", "content": response})

        # 在聊天界面上显示模型的输出
        st.chat_message("assistant").write(response)

with col2:
    st.image("https://via.placeholder.com/150", caption="ByteBrain Logo")
    st.markdown("### 关于ByteBrain")
    st.markdown("ByteBrain是一个智能知识助手，旨在帮助用户快速获取信息和解决问题。")
    st.markdown("### 联系我们")
    st.markdown("如果您有任何问题或建议，请通过以下方式联系我们：")
    st.markdown("- 邮箱: support@bytebrain.com")
    st.markdown("- 电话: 123-456-7890")
