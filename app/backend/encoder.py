"""FashionCLIP text encoder: a sentence in, one 512-number vector out.

torch and transformers are imported inside the class so the rest of the app (and the tests)
work without them. The text vector lives in the same space as the image vectors computed in
Phase 2, which is what makes "type a sentence, find photos" possible.
"""

import numpy as np


class TextEncoder:
    def __init__(self, hf_id: str, revision: str | None = None, device: str | None = None):
        import torch
        from transformers import AutoTokenizer, CLIPModel

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(hf_id, revision=revision)
        model = CLIPModel.from_pretrained(hf_id, revision=revision)
        self.model = model.eval().to(self.device)  # float32: the text input is tiny, speed is fine

    def encode(self, text: str) -> np.ndarray:
        """float32 vector of shape (dim,). Text longer than 77 tokens is cut (CLIP's limit)."""
        tokens = self.tokenizer(
            [text], padding=True, truncation=True, max_length=77, return_tensors="pt"
        ).to(self.device)
        with self.torch.inference_mode():
            out = self.model.get_text_features(**tokens)
        features = out if self.torch.is_tensor(out) else out.pooler_output
        return features[0].float().cpu().numpy()
