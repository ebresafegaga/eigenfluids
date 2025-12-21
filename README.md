uv venv -p /opt/homebrew/bin/python3.11
source .venv/bin/activate

# Install dependencies
uv pip install -e .[dev]

# Run the first scene
uv run cube.py

# Run the second scene
uv run cuboid.py
