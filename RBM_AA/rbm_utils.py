import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def reg_inference(rbm, fpnn, batch, tol=1E-2, epochs=1000):
    actual = batch.y[:,-1:]
    target = nn.Parameter(torch.zeros(actual.size()))
    with torch.inference_mode(True):
        fps = fpnn(batch)
    
    opt = torch.optim.Adam([target])

    delta = np.inf
    prev_loss = 0
    for epoch in range(epochs):
        inp = torch.cat([fps, target], dim=1).float()
        loss = rbm.rbm.fe_loss(inp)
        delta = loss - prev_loss
        prev_loss = loss
        print(f"Epoch {epoch}: {loss}")
        opt.zero_grad()
        loss.backward()
        opt.step()
    rescaled_target = rbm.scaler.inverse_transform(target.detach().numpy())
    return actual, rescaled_target, target
    