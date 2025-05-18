FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04
LABEL authors="lukashinterleitner"

RUN apt update && apt upgrade -y
RUN apt install -y python3 python3-pip
RUN apt clean
RUN rm -rf /var/lib/apt/lists/*

COPY requirements.txt /requirements/requirements.txt
COPY submodules/open-instruct /requirements/submodules/open-instruct

WORKDIR /requirements

RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r /requirements/requirements.txt
RUN pip install --no-cache-dir --no-build-isolation traker[fast]==0.3.2

WORKDIR /app

CMD ["python3"]