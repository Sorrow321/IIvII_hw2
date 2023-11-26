FROM ubuntu:22.04

WORKDIR /solution

# dependencies
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get upgrade -y

RUN apt-get install -y \
        build-essential git python3 python3-pip wget \
        ffmpeg libsm6 libxext6 libxrender1 libglib2.0-0
RUN apt-get install -y unzip

RUN pip3 install -U pip
RUN pip3 install --upgrade pip
RUN pip3 install gdown
RUN gdown 1vstfcOpIt9mlauf9h4Kee2t30BmBe6wd
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
RUN pip3 install git+https://github.com/LAION-AI/dalle2-laion
RUN unzip weights_configs.zip
COPY . .
RUN mv modified_files/dalle2_pytorch.py /usr/local/lib/python3.10/dist-packages/dalle2_pytorch/dalle2_pytorch.py
RUN mv modified_files/InferenceScript.py /usr/local/lib/python3.10/dist-packages/dalle2_laion/scripts/InferenceScript.py