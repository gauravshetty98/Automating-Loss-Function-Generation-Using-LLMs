"""
Rubick: An LLM-based agent for generating PyTorch-compatible loss functions.

This package provides the Rubick class, which:
- Uses a large language model (e.g., CodeQwen, StarCoder) to generate loss functions
- Extracts valid Python code blocks from raw model output
- Automatically tests and fixes the generated loss functions

Usage Example:
--------------
from rubick import Rubick

rubick = Rubick(model_id="your-model-id", token="your-token", prompt="Task description")
rubick.process_start()
"""

from .rubick import Rubick
