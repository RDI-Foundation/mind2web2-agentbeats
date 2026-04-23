FROM ghcr.io/astral-sh/uv:python3.13-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends git xvfb xauth \
    && rm -rf /var/lib/apt/lists/*

RUN adduser agent
USER agent
WORKDIR /home/agent

COPY pyproject.toml uv.lock README.md ./
COPY src src

RUN --mount=type=cache,target=/home/agent/.cache/uv,uid=1000 \
    uv sync --locked

# Install Chromium for patchright (webpage capture during evaluation)
RUN uv run patchright install chromium
USER root
RUN uv run patchright install-deps chromium
USER agent

COPY --chown=agent:agent mind2web2-data mind2web2-data

RUN if [ ! -d mind2web2-data/evaluation_scripts ]; then \
        echo "ERROR: mind2web2-data/evaluation_scripts is missing from the build context." >&2; \
        echo "Download the dataset locally before building (see README.md > Setup):" >&2; \
        echo "" >&2; \
        echo "  HF_TOKEN=hf_... uv run python -c \"from huggingface_hub import snapshot_download; snapshot_download('osunlp/Mind2Web-2', repo_type='dataset', local_dir='mind2web2-data', token='\$HF_TOKEN')\"" >&2; \
        echo "" >&2; \
        exit 1; \
    fi

ENV DATA_DIR=/home/agent/mind2web2-data
ENV CACHE_DIR=/home/agent/cache
ENTRYPOINT ["xvfb-run", "uv", "run", "src/server.py"]
CMD ["--host", "0.0.0.0", "--port", "9009"]
EXPOSE 9009
