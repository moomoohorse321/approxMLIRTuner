import os
import re
from typing import Callable, List, Optional, Tuple, Set
import csv
import time
# Set JAX to use GPU memory if available.
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.00"

import jax
from gemma import gm

class LLMManager:
    """Manages the Gemma models (fast and pro) and the sampler."""
    def __init__(self, model_sizes: List[str] = ["270m", "1b"]):
        self.models = []
        self.params = []
        self.samplers = []
        self.model_map = {size: i for i, size in enumerate(model_sizes)}
        self._load_models(model_sizes)

    def _load_models(self, model_sizes: List[str]):
        """Loads the specified Gemma models and parameters."""
        try:
            for size in model_sizes:
                if size == "270m":
                    print("Loading Gemma 270M model (fast agent)...")
                    model = gm.nn.Gemma3_270M()
                    params = gm.ckpts.load_params(gm.ckpts.CheckpointPath.GEMMA3_270M_IT)
                elif size == "1b":
                    print("Loading Gemma 1B model (pro agent)...")
                    model = gm.nn.Gemma3_1B()
                    params = gm.ckpts.load_params(gm.ckpts.CheckpointPath.GEMMA3_1B_IT)
                else: raise ValueError(f"Unsupported model size: {size}")
                self.models.append(model)
                self.params.append(params)
                self.samplers.append(gm.text.ChatSampler(model=model, params=params))
            print("All models loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def generate(self, prompt: str, model_size: str) -> str:
        """Generates a response from a specific LLM."""
        model_idx = self.model_map.get(model_size)
        if model_idx is None: raise ValueError(f"Model size '{model_size}' not loaded.")
        reply = self.samplers[model_idx].chat(prompt, multi_turn=False, print_stream=False)
        return reply