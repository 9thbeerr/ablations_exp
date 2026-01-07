#!/bin/bash

# Install uv
pip install uv

uv venv --python 3.13
# Create and activate virtual environment
source .venv/bin/activate 

# Install packages from pyproject.toml or requirements.txt
uv run

export TOKENIZERS_PARALLELISM="true"
