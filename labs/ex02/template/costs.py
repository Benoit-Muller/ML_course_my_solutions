# -*- coding: utf-8 -*-
"""a function used to compute the loss."""

import numpy as np




def compute_loss(y, tx, w):
    """Calculate the loss using either MSE or MAE.

    Args:
        y: shape=(N, )
        tx: shape=(N,D)
        w: shape=(D,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    return np.sum((y-tx@w)**2)/(2*len(y))