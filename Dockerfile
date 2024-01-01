FROM python:3.10.13-bullseye

USER 0
#RUN sed -i 's/http:\/\/deb.debian.org/https:\/\/mirrors.cloud.tencent.com/g' /etc/apt/sources.list

RUN apt-get clean &&  apt update && apt install ffmpeg vim -y

WORKDIR /usr/src

ADD requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt  -i https://mirrors.aliyun.com/pypi/simple

RUN pip install --no-cache-dir opencv-python torchvision matplotlib librosa scikit-image color_matcher numexpr pandas py-cpuinfo moviepy scikit-build diskcache gitpython -i https://mirrors.aliyun.com/pypi/simple

COPY ./models ./models

COPY ./custom_nodes ./custom_nodes

COPY ./comfy_extras ./comfy_extras

COPY ./web ./web

RUN mkdir input && mkdir output && mkdir output2

COPY ./comfy ./comfy

RUN mkdir ./output2/videos

COPY *.py *.txt *.json *.yaml ./

RUN mkdir -p /root/.cache/torch/hub/checkpoints && touch /root/.cache/torch/hub/checkpoints/mobilenet_v2-b0353104.pth
