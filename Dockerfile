FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04
LABEL authors="lukashinterleitner"

RUN apt update && apt upgrade -y
RUN apt install -y python3 python3-pip
RUN pip install --upgrade pip
RUN apt clean
RUN rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
COPY submodules/open-instruct ./submodules/open-instruct

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir --no-build-isolation traker[fast]==0.3.2

CMD ["python3"]