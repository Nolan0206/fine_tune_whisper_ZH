#! /bin/bash
echo "play"
pip install --upgrade huggingface_hub -i https://pypi.tuna.tsinghua.edu.cn/simple
#add-apt-repository -y ppa:jonathonf/ffmpeg-4
#apt install -y ffmpeg
pip install evaluate>=0.30 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install jiwer -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install gradio -i https://pypi.tuna.tsinghua.edu.cn/simple