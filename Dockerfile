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

# Install the project's dependencies using the lockfile and settings
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-dev --no-cache

WORKDIR /app

# Place executables in the environment at the front of the path
ENV PATH="/uv/.venv/bin:$PATH"

CMD ["python3"]