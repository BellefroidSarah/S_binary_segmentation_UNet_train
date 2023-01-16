import torchgeometry as tgm
import torch.nn as nn
import numpy as np

import torch


# https://stackoverflow.com/questions/2130016/splitting-a-list-into-n-parts-of-approximately-equal-length
def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))


def select_loss_function(params):
    if params.parameters.loss_function == "Dice":
        return tgm.losses.DiceLoss()
    elif params.parameters.loss_function == "Focal":
        return tgm.losses.FocalLoss(alpha=params.parameters.focal_gamma, gamma=params.parameters.focal_gamma, reduction="mean")
    elif params.parameters.loss_function == "Tversky":
        return tgm.losses.TverskyLoss(alpha=params.parameters.tversky_alpha, beta=params.parameters.tversky_beta)
    else:
        return nn.CrossEntropyLoss()


# One epoch training
def train_model(model, loader, DEVICE, criterion, optimizer):
    model.train()
    train_loss = []
    for images, masks, _ in loader:
        outputs = model(images.to(DEVICE))
        masks = torch.squeeze(masks.type(torch.LongTensor), 1)

        loss = criterion(outputs, masks.to(DEVICE))
        train_loss.append(loss.detach().item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return np.mean(train_loss)


def validate_model(model, loader, DEVICE, criterion):
    model.eval()
    with torch.no_grad():
        val_loss = []
        for images, masks, _ in loader:
            outputs = model(images.to(DEVICE))
            masks = torch.squeeze(masks.type(torch.LongTensor), 1)

            loss = criterion(outputs, masks.to(DEVICE))
            val_loss.append(loss.detach().item())
    return np.mean(val_loss)
