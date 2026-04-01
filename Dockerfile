# Build:
#   docker build -t anneal .
#
# Run (mount the target repository as /repo):
#   docker run --rm -v $(pwd):/repo -w /repo anneal run --target my-target --experiments 20
#
# The container does not include API credentials. Pass them via environment:
#   docker run --rm -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
#     -v $(pwd):/repo -w /repo anneal run --target my-target --experiments 20

FROM python:3.12-slim

# Install system dependencies required by uv and git operations
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Create a non-root user
RUN useradd --create-home --shell /bin/bash anneal
USER anneal
WORKDIR /home/anneal

# Copy project files
COPY --chown=anneal:anneal pyproject.toml uv.lock ./
COPY --chown=anneal:anneal anneal/ ./anneal/

# Install the package
RUN uv pip install --system .

ENTRYPOINT ["anneal"]
