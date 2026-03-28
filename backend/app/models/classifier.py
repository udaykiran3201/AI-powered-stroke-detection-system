"""
Stroke Detection System — ResNet/EfficientNet Classification Model

Multi-label stroke classifier that distinguishes:
  • Normal (no stroke)
  • Hemorrhagic stroke  (epidural, intraparenchymal, intraventricular, subarachnoid, subdural)
  • Ischemic stroke
"""

import torch
import torch.nn as nn
import timm
from typing import List, Dict, Optional
from loguru import logger


# Hemorrhage sub-types + ischemic
STROKE_CLASSES: List[str] = [
    "epidural",
    "intraparenchymal",
    "intraventricular",
    "subarachnoid",
    "subdural",
    "ischemic",
]

NUM_CLASSES = len(STROKE_CLASSES)


class StrokeClassifier(nn.Module):
    """
    Transfer-learning classifier built on top of an EfficientNet-B3 backbone.
    Produces multi-label logits for the six stroke classes.
    """

    def __init__(
        self,
        backbone_name: str = "efficientnet_b3",
        num_classes: int = NUM_CLASSES,
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=0  # remove head
        )
        in_features = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_classes),
        )
        logger.info(
            f"StrokeClassifier initialised — backbone={backbone_name}, "
            f"classes={num_classes}, in_features={in_features}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)


def load_classifier(
    weights_path: Optional[str] = None,
    device: str = "cpu",
    backbone_name: str = "efficientnet_b3",
) -> StrokeClassifier:
    """Instantiate the classifier and optionally load saved weights."""
    model = StrokeClassifier(backbone_name=backbone_name, pretrained=(weights_path is None))
    if weights_path:
        try:
            state_dict = torch.load(weights_path, map_location=device, weights_only=True)
            model.load_state_dict(state_dict)
            logger.info(f"Loaded classifier weights from {weights_path}")
        except FileNotFoundError:
            logger.warning(
                f"Weights file not found at {weights_path} — using random head weights"
            )
    model.to(device).eval()
    return model


def predict(
    model: StrokeClassifier,
    image_tensor: torch.Tensor,
    device: str = "cpu",
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Run inference and return per-class probabilities.

    Parameters
    ----------
    model : StrokeClassifier
    image_tensor : torch.Tensor   — shape (1, C, H, W), pre-processed
    device : str
    threshold : float             — classification decision boundary

    Returns
    -------
    dict  — { class_name: probability }
    """
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        logits = model(image_tensor)
        probs = torch.sigmoid(logits).squeeze(0).cpu().tolist()

    results = {cls: round(prob, 4) for cls, prob in zip(STROKE_CLASSES, probs)}
    return results
