FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04
LABEL authors="lukashinterleitner"

RUN apt update && apt upgrade -y
RUN apt install -y python3 python3-pip
RUN pip install --upgrade pip
RUN apt clean
RUN rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
COPY submodules/open-instruct /app/submodules/open-instruct

RUN pip install --no-cache-dir -r /app/requirements.txt
RUN pip install --no-cache-dir --no-build-isolation traker[fast]==0.3.2