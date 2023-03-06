import torch

def contrastive_loss(dist, label, margin: float = 0.5, reduction="mean"):
    """
    Computes Contrastive Loss
    """

    loss = (1 - label) * torch.pow(dist, 2) + (label) * torch.pow(torch.clamp(margin - dist, min=0.0), 2)
    
    if reduction == "mean":
        loss = torch.mean(loss)
    elif reduction == "sum":
        loss = torch.sum(loss)
    else:
        raise TypeError("Reduction is not supported")
    
    return loss