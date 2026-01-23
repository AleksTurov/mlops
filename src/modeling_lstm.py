import numpy as np
import pandas as pd
import torch


def _predict_probs_from_loader(dl, model, device):
    """
    Вычисляет вероятности положительного класса для всех образцов в DataLoader.
    """
    model.eval()
    probs = np.zeros(len(dl.dataset), dtype=np.float32)
    with torch.no_grad():
        for batch in dl:
            xb_n = batch['X_num'].to(device)
            xb_c = batch['X_cat'].to(device)
            lengths = batch['lengths']
            logits = model(xb_n, xb_c, lengths)
            batch_probs = torch.sigmoid(logits).cpu().numpy().ravel()
            idxs = batch['indices'].cpu().numpy()
            probs[idxs] = batch_probs
    return probs