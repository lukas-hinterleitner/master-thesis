FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

LABEL authors="lukashinterleitner"

RUN apt update && apt upgrade -y
RUN apt clean
RUN rm -rf /var/lib/apt/lists/*

WORKDIR /uv

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

WORKDIR /uv

COPY submodules/open-instruct /uv/submodules/open-instruct
COPY uv.lock /uv/uv.lock
COPY pyproject.toml /uv/pyproject.toml

RUN uv sync --locked --no-install-project --no-dev --no-cache
RUN uv sync --locked --no-install-project --no-dev --no-cache --extra build --extra compile

WORKDIR /app

# Place executables in the environment at the front of the path
ENV PATH="/uv/.venv/bin:$PATH"

CMD ["python3"]