import numpy as np
import torch


def gaussian_window1d(sigma):
    nw = 3 * sigma + 1 if sigma % 2 == 0 else 3 * sigma
    x = np.arange(nw) - (nw - 1) / 2
    window = np.exp(-(x/sigma)**2)
    return window/window.sum()


def gaussian_window2d(sigma):
    window = gaussian_window1d(sigma)
    window = np.outer(window, window)
    return window/window.sum()


def torch_gaussian_window1d(sigma):

    nw = 6 * sigma
    nw = nw + 1 if nw % 2 == 0 else nw
    x = torch.arange(nw) - (nw - 1) / 2
    #window = torch.exp(-((x/sigma)**2)/np.sqrt(2))
    window = torch.exp(-(x**2)/(2*sigma**2)) 

    return window / window.sum()


def torch_gaussian_window2d(sigma):

    try:
        sigma_y, sigma_x = sigma
    except TypeError:
        sigma_y = sigma_x = sigma

    window_y = torch_gaussian_window1d(sigma_y)
    window_x = torch_gaussian_window1d(sigma_x)
    window = torch.outer(window_y, window_x)
    window = window.reshape(1, 1, *window.shape)
    return window / window.sum()
