FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04 AS build

# uv only needed in build stage
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /uv
ENV UV_NO_CACHE=1 PIP_NO_CACHE_DIR=1

COPY pyproject.toml uv.lock ./
COPY submodules/open-instruct submodules/open-instruct

RUN --mount=type=cache,target=/root/.cache \
    uv sync --extra build --locked --no-dev --no-install-project \
 && uv sync --extra build --extra trakfast --locked --no-dev --no-install-project

FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04 AS runtime

COPY --from=build /uv/.venv /uv/.venv

ENV PATH="/uv/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
CMD ["python"]