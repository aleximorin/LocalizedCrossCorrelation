import numpy as np
import matplotlib.pyplot as plt
import torch
from optimizer import ShiftOptim
from lcc2d import LCC2D
import pickle
import xarray as xr

import io

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


def lcc_fwdgpr():


    path = 'data/findelen_migration/migrated_findelen.nc'

    da = xr.load_dataarray(path)

    step = 1
    im0 = da.isel(survey=0).values[::step, ::step]#.copy()
    im1 = da.isel(survey=1).values[::step, ::step]#.copy()
    
    print(im0.shape)
    
    # parameters for the lcc
    maxlag = 45
    wlags = np.arange(-maxlag, maxlag + 1, 1).astype(int)
    hlags = np.arange(-maxlag, maxlag + 1, 1).astype(int)
    
    
    search_sigmas = [31, 81] # the current one
    #search_sigmas = [21, 61]

    # first estimation
    lcc = LCC2D(im0, im1,
                hlags, wlags,
                search_sigmas,
                threshold=3,
                stride=4,
                nugget=1e-32)

    lcc.fit() 

    median_kernel = [47, 121]
    from scipy.ndimage import median_filter
    dx = torch.from_numpy(median_filter(lcc.subdw, median_kernel, mode='reflect'))
    dy = torch.from_numpy(median_filter(lcc.subdh, median_kernel, mode='reflect'))
    shift = torch.stack((dx, dy)).to(torch.float32)
    
    #kernel = 31
    #weights = lcc.convolutions.cpu().numpy().max(axis=(0, 1)) 
    #weights = np.sqrt((weights - weights.min())/(weights.max() - weights.min()))
    #dx =  torch.from_numpy(weighted_moving_average(lcc.subdw, weights, kernel))
    #dy =  torch.from_numpy(weighted_moving_average(lcc.subdh, weights, kernel))
    #shift = torch.stack((dx, dy)).to(torch.float32)

    # optimization scheme
    lmbda = [1e0, 1e1] # [1e3, 1e1] # 1e2
    beta = 0 # 1e-1
    niter = 1e5

    model = ShiftOptim(lcc.f, lcc.g, shift, lmbda=lmbda, beta=beta, correlation=lcc.convolutions)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    out, losses = model.fit(optimizer, n=int(niter), threshold=2e-3)#, nsave=10)

    # we dump the results in a pickle file
    out_dict = {'out': out, 'losses': losses, 'lcc': lcc, }
    out_path = 'data/findelen_migration/findelen_out.p'

    with open(out_path, 'wb') as file:
        pickle.dump(out_dict, file)
    print('\nsaving done')

    return out, losses

if __name__ == '__main__':

    device = torch.device('cuda:1')
    print(device)
    field, loss = lcc_fwdgpr()



"""
import matplotlib
matplotlib.use('Qt5Agg')
vmax = np.quantile(np.abs(im0), 0.90)

kws = dict(aspect='auto', vmin=-vmax, vmax=vmax, cmap='coolwarm')

DX = 0.10
DZ = 0.10
DT = 3e-10
maxT = 1.28e-6
NT = int(1.15 * maxT/DT)
CF = 80e6

eps = np.load('data/eps.npy', allow_pickle=True)

eps0 = eps[0]
eps1 = eps[1]

vmax = np.quantile(np.abs(im0), 0.9)

fig, axs = plt.subplots(1, 2, sharex='all', sharey='all')
axs[0].imshow(im0, extent=[100, 400, NT * DT, 0], vmin=-vmax, vmax=vmax, cmap='coolwarm')
axs[0].imshow(eps0, extent=[0, 500, maxT, 0], alpha=0.3, origin='lower')    

axs[1].imshow(im1, extent=[100, 400, NT * DT, 0], vmin=-vmax, vmax=vmax, cmap='coolwarm')
axs[1].imshow(eps1, extent=[0, 500, maxT, 0], alpha=0.3, origin='lower')

for ax in axs:
    ax.set_aspect('auto')

"""
