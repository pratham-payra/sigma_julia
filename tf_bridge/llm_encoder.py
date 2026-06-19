"""
llm_encoder.py
LLM-based emergency priority encoder.

Uses HuggingFace transformers to load LLaMA-2-7B (or any causal LM)
and produce the 16-dim binary priority vector θ from natural language.

Falls back to a rule-based encoder if the model is not available,
matching the deterministic surrogate used in Sigma_manual/Environment.jl.

Usage:
    encoder = EmergencyEncoder(use_llm=True)   # loads LLaMA-2
    theta   = encoder.encode("Ambulance from North toward East")
"""

import re
import numpy as np

# Direction name → index mapping
DIR_MAP = {
    "east": 0, "e": 0,
    "north": 1, "n": 1,
    "west": 2,  "w": 2,
    "south": 3, "s": 3,
}

PROMPT_TEMPLATE = """You are a traffic signal controller assistant.
An emergency vehicle is approaching an intersection.
Instruction: {instruction}

Based on the instruction, identify:
1. The incoming direction (East/North/West/South or Unknown)
2. The outgoing direction (East/North/West/South or Unknown)

Respond ONLY in this format:
INCOMING: <direction>
OUTGOING: <direction>
"""


class EmergencyEncoder:
    def __init__(self, use_llm: bool = False,
                 model_name: str = "meta-llama/Llama-2-7b-chat-hf",
                 device: str = "cpu"):
        self.use_llm = use_llm
        self._model    = None
        self._tokenizer = None

        if use_llm:
            self._load_llm(model_name, device)

    def _load_llm(self, model_name: str, device: str):
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            print(f"[LLM] Loading {model_name} on {device}...")
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if device != "cpu" else torch.float32,
                device_map=device
            )
            self._model.eval()
            print("[LLM] Model loaded.")
        except Exception as e:
            print(f"[LLM] Failed to load model: {e}")
            print("[LLM] Falling back to rule-based encoder.")
            self.use_llm = False

    def _llm_parse(self, instruction: str):
        """Query the LLM and parse INCOMING/OUTGOING directions."""
        prompt = PROMPT_TEMPLATE.format(instruction=instruction)
        inputs = self._tokenizer(prompt, return_tensors="pt")
        import torch
        with torch.no_grad():
            output = self._model.generate(
                **inputs,
                max_new_tokens=40,
                temperature=0.1,
                do_sample=False,
            )
        text = self._tokenizer.decode(output[0], skip_special_tokens=True)
        return self._parse_response(text)

    def _parse_response(self, text: str):
        """Extract incoming/outgoing indices from LLM response."""
        inc_match = re.search(r"INCOMING:\s*(\w+)", text, re.IGNORECASE)
        out_match = re.search(r"OUTGOING:\s*(\w+)", text, re.IGNORECASE)

        inc = DIR_MAP.get(inc_match.group(1).lower(), -1) if inc_match else -1
        out = DIR_MAP.get(out_match.group(1).lower(), -1) if out_match else -1
        return inc, out

    def _rule_parse(self, instruction: str):
        """
        Rule-based fallback: scan instruction string for direction keywords.
        Matches patterns like "from North", "toward East", "coming from West".
        """
        text = instruction.lower()

        from_dirs = re.findall(
            r'(?:from|approaching from|coming from)\s+(\w+)', text)
        to_dirs   = re.findall(
            r'(?:toward|to|heading to|going to|towards)\s+(\w+)', text)

        inc = DIR_MAP.get(from_dirs[0], -1) if from_dirs else -1
        out = DIR_MAP.get(to_dirs[0],   -1) if to_dirs   else -1

        # Fallback: just find any direction words in order
        if inc == -1 or out == -1:
            found = [DIR_MAP[w] for w in text.split() if w in DIR_MAP]
            if len(found) >= 2:
                inc = found[0] if inc == -1 else inc
                out = found[1] if out == -1 else out

        return inc, out

    def encode(self, instruction: str) -> np.ndarray:
        """
        Encode a natural language emergency instruction into a
        16-dim binary priority vector θ (4×4 movement matrix, flattened).

        Paper encoding rules (Section 4.3):
          exact path:    θ[origin, dest] = 1
          incoming-only: θ[origin, :]    = 1
          outgoing-only: θ[:, dest]      = 1
        """
        if self.use_llm and self._model is not None:
            inc, out = self._llm_parse(instruction)
        else:
            inc, out = self._rule_parse(instruction)

        theta = np.zeros(16, dtype=np.float32)

        if inc >= 0 and out >= 0:
            # Exact path
            theta[inc * 4 + out] = 1.0
        elif inc >= 0:
            # Incoming-only
            for j in range(4):
                theta[inc * 4 + j] = 1.0
        elif out >= 0:
            # Outgoing-only
            for i in range(4):
                theta[i * 4 + out] = 1.0
        # else: no emergency — zero vector

        return theta


# ─── Convenience wrapper for batch encoding ───────────────────────────────────
def encode_emergency_rule(origin: int, dest: int) -> np.ndarray:
    """Direct rule encoding (no LLM) matching Julia Environment.encode_emergency."""
    theta = np.zeros(16, dtype=np.float32)
    if 0 <= origin < 4 and 0 <= dest < 4:
        theta[origin * 4 + dest] = 1.0
    elif 0 <= origin < 4:
        theta[origin * 4: origin * 4 + 4] = 1.0
    return theta
