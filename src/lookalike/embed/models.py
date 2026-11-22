"""Pretrained image encoders (ResNet-50, CLIP, DINOv2, FashionCLIP) through Hugging Face.

torch and transformers are imported inside the class so that the rest of the package, and the
test suite, work without them installed. They are preinstalled on Kaggle and declared in the
`ml` dependency group.

What each `kind` returns as "the embedding":
- resnet: global-average-pooled final feature map (2048 numbers).
- clip:   `get_image_features`, the projection into the shared image-text space (512 numbers).
- dinov2: the pooled CLS token (768 numbers).
"""

import numpy as np


class ImageEncoder:
    def __init__(self, spec: dict, device: str | None = None):
        import torch
        import transformers
        from transformers import AutoImageProcessor, AutoModel, CLIPModel

        self.torch = torch
        self.kind = spec["kind"]
        if self.kind not in {"resnet", "clip", "dinov2"}:
            raise ValueError(f"unknown model kind: {self.kind}")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.half = self.device == "cuda"  # fp16 on GPU; T4 has no bf16

        self.processor = AutoImageProcessor.from_pretrained(spec["hf_id"])
        model_cls = CLIPModel if self.kind == "clip" else AutoModel
        model = model_cls.from_pretrained(spec["hf_id"]).eval().to(self.device)
        if self.half:
            model = model.half()
        self.model = model

        self.meta = {
            "hf_id": spec["hf_id"],
            "kind": self.kind,
            "commit_hash": getattr(model.config, "_commit_hash", None),
            "device": self.device,
            "fp16": self.half,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        }

    def _features(self, pixel_values):
        torch = self.torch
        if self.kind == "clip":
            out = self.model.get_image_features(pixel_values=pixel_values)
            return out if torch.is_tensor(out) else out.pooler_output
        return self.model(pixel_values=pixel_values).pooler_output.flatten(1)

    def encode(self, images: list) -> np.ndarray:
        """list of PIL RGB images -> float32 array (len(images), dim)."""
        pixel_values = self.processor(images=images, return_tensors="pt")["pixel_values"]
        pixel_values = pixel_values.to(self.device)
        if self.half:
            pixel_values = pixel_values.half()
        with self.torch.inference_mode():
            features = self._features(pixel_values)
        return features.float().cpu().numpy()
