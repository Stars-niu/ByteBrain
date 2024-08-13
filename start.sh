#!/bin/bash

# 安装Git LFS
git lfs install

# 克隆仓库
git clone https://github.com/your-username/ByteBrain.git

# 安装Streamlit
pip install streamlit==1.24.0

# 进入项目目录
cd ByteBrain

# 启动Streamlit应用
streamlit run demo/app.py --server.address 127.0.0.1 --server.port 6006
