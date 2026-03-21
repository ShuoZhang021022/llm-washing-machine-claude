"""
Manual SAE implementation that loads pre-trained weights from HuggingFace
without depending on the sae_lens import chain.
"""

import json
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file


class ManualSAE(nn.Module):
    """Simple ReLU SAE: encode(x) = ReLU(W_enc @ (x - b_dec) + b_enc), decode(z) = W_dec @ z + b_dec"""

    def __init__(self, d_in, d_sae, W_enc, b_enc, W_dec, b_dec):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.register_buffer('W_enc', W_enc)
        self.register_buffer('b_enc', b_enc)
        self.register_buffer('W_dec', W_dec)
        self.register_buffer('b_dec', b_dec)

    def encode(self, x):
        """Encode input activations to SAE feature activations."""
        x_centered = x - self.b_dec
        return torch.relu(x_centered @ self.W_enc + self.b_enc)

    def decode(self, z):
        """Decode SAE feature activations back to model space."""
        return z @ self.W_dec + self.b_dec

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z

    @classmethod
    def from_pretrained(cls, layer, device="cpu"):
        """Load pre-trained SAE for GPT-2 Small from HuggingFace."""
        hook_point = f"blocks.{layer}.hook_resid_pre"
        repo_id = "jbloom/GPT2-Small-SAEs-Reformatted"

        # Download config
        cfg_path = hf_hub_download(repo_id=repo_id, filename=f"{hook_point}/cfg.json")
        with open(cfg_path) as f:
            cfg = json.load(f)

        # Download weights
        weights_path = hf_hub_download(repo_id=repo_id, filename=f"{hook_point}/sae_weights.safetensors")
        state_dict = load_file(weights_path, device=device)

        d_in = cfg["d_in"]
        d_sae = cfg["d_sae"]

        sae = cls(
            d_in=d_in,
            d_sae=d_sae,
            W_enc=state_dict["W_enc"],
            b_enc=state_dict["b_enc"],
            W_dec=state_dict["W_dec"],
            b_dec=state_dict["b_dec"],
        )
        sae = sae.to(device)
        return sae, cfg
