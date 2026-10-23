import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


@torch.no_grad()
def predict(model, images: torch.Tensor, labels: np.ndarray, device, batch_size: int = 1000) -> dict:
    """Returns predictions, per-example cross-entropy loss and max softmax confidence."""
    model.eval()
    targets = torch.from_numpy(labels)
    preds, losses, confidences = [], [], []
    for start in range(0, len(images), batch_size):
        x = images[start:start + batch_size].to(device)
        y = targets[start:start + batch_size].to(device)
        # float64 so that very small losses of confidently-classified members do not round to
        # exactly 0 and tie with each other, which would distort the low-FPR part of the ROC curve.
        logits = model(x).double()
        probs = F.softmax(logits, dim=1)
        preds.append(logits.argmax(dim=1).cpu())
        losses.append(F.cross_entropy(logits, y, reduction="none").cpu())
        confidences.append(probs.max(dim=1).values.cpu())
    return {
        "pred": torch.cat(preds).numpy(),
        "loss": torch.cat(losses).numpy(),
        "confidence": torch.cat(confidences).numpy(),
    }


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    # Macro averaging: every CIFAR-10 class counts equally.
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def evaluate_model(model, images, labels, device) -> dict:
    out = predict(model, images, labels, device)
    metrics = classification_metrics(labels, out["pred"])
    metrics["mean_loss"] = float(out["loss"].mean())
    return metrics
