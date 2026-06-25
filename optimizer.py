import torch
import torchfields
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from gaussian_windows import torch_gaussian_window2d

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def assert_tensor(x):
    if torch.is_tensor(x):
        return x.to(torch.float32).to(device)
    else:
        return torch.tensor(x, dtype=torch.float32, device=device)


def unnormalize(X, shape):
    X[0] = X[0] * shape[1] / 2
    X[1] = X[1] * shape[0] / 2
    return -X


def normalize(X, shape):
    X[0] = 2 * X[0] / shape[1]
    X[1] = 2 * X[1] / shape[0]
    return -X


def get_laplace():
    laplace = torch.tensor(
        [[1, 1, 1],
         [1, -8, 1],
         [1, 1, 1]],
        dtype=torch.float32).reshape(1, 1, 3, 3)

    return laplace

def get_log(kernel=3):
    gw = torch_gaussian_window2d(kernel)
    laplace = get_laplace()
    pgw = F.pad(gw, (1, 1, 1, 1), mode='constant')#, value=0)
    log = F.conv2d(gw, laplace, padding='valid')
    log = log / torch.abs(log).sum()
    return log

def get_d2():
    d2x = torch.tensor([[[[1, -2, 1]]]], dtype=torch.float32)
    d2z = torch.tensor([[[[1], [-2], [1]]]], dtype=torch.float32)
    return d2x, d2z


class ShiftOptim(torch.nn.Module):

    def __init__(self, f, g, v0, lmbda, beta=0, laplace_kernel=3, correlation=None):

        super().__init__()

        self.f = assert_tensor(f)
        self.g = assert_tensor(g)

        self.f = (self.f - self.f.min()) / (self.f.max() - self.f.min())
        self.g = (self.g - self.g.min()) / (self.g.max() - self.g.min())

        # log transform?
        #self.f = torch.log(torch.abs(self.f) + 1)
        #self.g = torch.log(torch.abs(self.g) + 1)


        # normalize images to [0, 1]
        #self.f = (self.f - self.f.min())/(self.f.max() - self.f.min())
        #self.g = (self.g - self.g.min())/(self.g - self.g.min())

        self.ndim = v0.shape[0]
        self.shape_full = self.f.shape
        self.shape_v0 = v0.shape[1:] # Shape of the initial velocity field

        # uncomment for random initial field
        """
        field = torch.rand(self.shape[0] * self.shape[1] * 2,  dtype=torch.float64) * 2 - 1
        field = field.reshape(2, *self.shape)
        """

        self.v0_normalized = normalize(assert_tensor(v0).clone(), self.shape_full)

        if correlation is not None:
            hlags, wlags, _, _ = correlation.shape
            self.corrmax = correlation.amax(axis=(0, 1)).unsqueeze(0).unsqueeze(0)
            self.correlation_downsampled = assert_tensor(correlation.reshape(*correlation.shape[:2], 1, -1).permute((3, 2, 0, 1))).to(device)
            #cc_range = (self.correlation.amax(dim=(2, 3), keepdim=True) - self.correlation.amin(dim=(2, 3), keepdim=True))
            #self.correlation = (self.correlation - self.correlation.amin(dim=(2, 3), keepdim=True))/cc_range

        field = self.v0_normalized.clone()
        self.weights = torch.nn.Parameter(field) # The weights are now the downsampled velocity field

        self.lmbda = assert_tensor(lmbda)
        self.beta = beta

        self.laplace = get_laplace().to(device) #
        #self.laplace  = get_log(laplace_kernel).to(device)
        laplace_kernel_length = 1 # self.laplace.shape[-1] // 2
        self.laplace_pad = (laplace_kernel_length, laplace_kernel_length, laplace_kernel_length, laplace_kernel_length)

        self.d2x, self.d2z = get_d2()
        self.d2x = self.d2x.to(device)
        self.d2z = self.d2z.to(device)

        print(self.d2x)
        print(self.d2z)
    
        self.ssim = SSIM(data_range=1.0, kernel_size=15, sigma=3, return_full_image=True).to(device)
        self.to(device)

    def forward(self):
        """
        Optimizes the downsampled velocity field and uses interpolation for warping.
        """
        downsampled_field = self.weights
        # Upsample the optimized downsampled velocity field to the original image resolution
        upsampled_field = F.interpolate(downsampled_field.unsqueeze(0), size=self.shape_full, mode='bilinear', align_corners=True)[0]
        return upsampled_field.field()(self.f)

    def loss(self, ghat):

        # mse loss related to the difference between the warped and the moved image
        err2 = torch.square(self.g - ghat)
        mse = err2.mean()

        # Structural Similarity Index measure score
        self.ssim.reset()
        #score, ssim = self.ssim(self.g.reshape(1, 1, *self.g.shape), ghat.reshape(1, 1, *ghat.shape))
        #mse = 1 - score

        w = (self.corrmax ** 1 + 1e-12)

        # spatial derivative loss on the DOWN-SAMPLED velocity field
        """tmpx = F.pad(self.weights.reshape(2, 1, *self.weights.shape[1:]), [1, 1, 0, 0], mode='reflect')
        tmpz = F.pad(self.weights.reshape(2, 1, *self.weights.shape[1:]), [0, 0, 1, 1], mode='reflect')
        
        # compute derivatives
        grad_x = F.conv2d(tmpx, self.d2x, padding='valid')
        grad_z = F.conv2d(tmpz, self.d2z, padding='valid')

        # compute relative scaling (example: normalize by average magnitude of displacements)
        scale = self.weights.amax(dim=(1, 2), keepdim=True) + 1e-12  # avoid division by zero

        #print(w.shape, grad_x.shape, grad_z.shape)

        # compute anisotropic loss
        grad = (self.lmbda[0] * ((grad_x / scale).square() * w).sum(dim=(2,3)) +
                self.lmbda[1] * ((grad_z / scale).square() * w).sum(dim=(2,3)))
        grad = grad.sum()"""
        
        tmp = F.pad(self.weights.reshape(2, 1, *self.weights.shape[1:]), self.laplace_pad, mode='reflect')
        grad = (F.conv2d(tmp, self.laplace, padding='valid') * w).norm(dim=(2, 3)) * self.lmbda
        grad = grad.sum()

        # loss related to prior knowledge of the deformation computed by the localized cross correlation
        # we need to change the coordinates relative to the lags
        xy = unnormalize(self.weights.clone(), self.shape_full)
        xy = normalize(xy, self.correlation_downsampled.shape[:-2])
        xy = -xy.reshape(2, -1, 1, 1).permute((1, 2, 3, 0))

        # interpolated correlations at the downsampled velocity field locations
        interp = 1.0 - torch.square(F.grid_sample(self.correlation_downsampled, xy, align_corners=True, mode='bilinear').flatten())

        # mass conversation term
        # divergence = dfx/dx + dfy/dy

        #gradients = torch.gradient(self.weights, axis=(1, 2)) #(2, height, widht) -> (3, 2, height, width) (dfx/dx, dfy/dx,
        #du_dx = gradients[0][0]
        #dv_dy = gradients[1][1]
        #div = du_dx + dv_dy

        #norm = self.weights.norm()

        return mse, grad, interp.mean() * self.beta

    def fit(self, optimizer, n=1e3, nsave=20, threshold=1e-3):
        save_every = n // nsave
        losses = np.zeros((n + 1, 3))

        weights = np.zeros((nsave + 1, *self.weights.shape)) # Save the downsampled weights

        loss = np.nan
        change = 0.0

        for i in range(int(n + 1)):
            print(f'\riteration {i+1}/{n + 1} = {(i+1) / (n + 1) * 100:.2f}%, loss = {loss:.8f} change = {change:.8f}', end='')

            ghat = self.forward()
            err, grad, diff = self.loss(ghat)

            loss = err + grad + diff
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            losses[i] = err.cpu().detach().numpy(), grad.cpu().detach().numpy(), diff.cpu().detach().numpy()

            if (i % save_every) == 0:
                weights[i // save_every] = unnormalize(self.weights.cpu().detach().clone().numpy(), self.shape_full)

            if i > 100:
                change = losses[i - 100:].std()
                if change < threshold:
                    print(f'\nconverged after {i} iterations')
                    weights[-1] = unnormalize(self.weights.cpu().detach().clone().numpy(), self.shape_full)
                    break

        return weights, losses[:i]