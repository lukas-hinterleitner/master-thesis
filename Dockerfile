FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04
COPY --from=ghcr.io/astral-sh/uv:0.8.10 /uv /uvx /bin/

LABEL authors="lukashinterleitner"

RUN apt update && apt upgrade -y
RUN apt clean
RUN rm -rf /var/lib/apt/lists/*

WORKDIR /uv

COPY submodules/open-instruct /uv/submodules/open-instruct
COPY uv.lock /uv/uv.lock
COPY pyproject.toml /uv/pyproject.toml

RUN uv sync --extra build --locked --no-install-project --no-dev --no-cache
RUN uv sync --extra build --extra trakfast

WORKDIR /app

# Place executables in the environment at the front of the path
ENV PATH="/uv/.venv/bin:$PATH"

CMD ["python3"]