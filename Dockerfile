FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04
LABEL authors="lukashinterleitner"

RUN apt update && apt upgrade -y
RUN apt install -y python3 python3-pip
RUN pip install --upgrade pip

# Create a non-root user
ARG USER_ID=1001
ARG GROUP_ID=1001
RUN groupadd -g ${GROUP_ID} appuser && \
    useradd -u ${USER_ID} -g appuser -s /bin/bash -m appuser

WORKDIR /app

COPY requirements.txt /app/requirements.txt
COPY submodules/open-instruct /app/submodules/open-instruct

RUN pip install -r /app/requirements.txt
RUN pip install --no-build-isolation traker[fast]==0.3.2

COPY application /app/application
COPY data /app/data

# Set proper ownership
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser