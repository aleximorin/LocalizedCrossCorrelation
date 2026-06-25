import numpy as np
import matplotlib.pyplot as plt
import torch
from optimizer import ShiftOptim
from lcc2d import LCC2D
import pickle
import xarray as xr

from scipy.ndimage import gaussian_filter


def weighted_moving_average(im, weights, kernel_size=11):
    numerator = gaussian_filter(im * weights, sigma=kernel_size, truncate=5, mode='reflect')
    denominator = gaussian_filter(weights, sigma=kernel_size, truncate=5, mode='reflect')
    smoothed = numerator / denominator
    return smoothed

def lcc_fwdgpr():

    path = 'data/synthetic_findelen/synth2_migrated_gridded.nc'

    da = xr.load_dataarray(path)
    print(da.shape)

    im0 = da.isel(date=0, ori=0).values[::1, ::1]#.copy()
    im1 = da.isel(date=1, ori=0).values[::1, ::1]#.copy()
    
    print(im0.shape)
    
    # parameters for the lcc
    maxlag = 25
    wlags = np.arange(-maxlag, maxlag + 1, 1).astype(int)   
    hlags = np.arange(-maxlag, maxlag + 1, 1).astype(int)
    
    #sigma_y = sigma_x = np.arange(25, 27, 1)
    
    #sy, sx = np.meshgrid(sigma_y, sigma_x)

    #search_sigmas = np.array([sy.flatten(), sx.flatten()]).T

    search_sigmas = [21, 31] # the current one 
    #search_sigmas = [11, 21]

    # first estimation
    lcc = LCC2D(im0, im1,
                hlags, wlags,
                search_sigmas,
                threshold=3,
                stride=4,
                nugget=1e-32)

    lcc.fit()
    
    #with open(f'data/synthetic_findelen/out2_migrated_gridded_parallel.p', 'rb') as out:
    #   print('trying to open something')
    #   lcc = pickle.load(out)['lcc']
    #   print('it\'s open lol')
    
    median_kernel = [31, 47]
    from scipy.ndimage import median_filter
    dx = torch.from_numpy(median_filter(lcc.subdw, median_kernel, mode='reflect'))
    dy = torch.from_numpy(median_filter(lcc.subdh, median_kernel, mode='reflect'))

    #weights = np.clip(lcc.convolutions.amax(axis=(0, 1)).cpu().numpy(), 0, 1) ** 2
    #dx = torch.from_numpy(weighted_moving_average(lcc.subdw, weights, kernel_size=11))
    #dy = torch.from_numpy(weighted_moving_average(lcc.subdh, weights, kernel_size=11))

    shift = torch.stack((dx, dy)).to(torch.float32)

    # optimization sch
  
    lmbda = [1e0, 1e1] # 1e0, 1e1
    beta =  0 # 1e1 # 1e2 ## 1e2 # 1e-2
    niter = 1e5

    model = ShiftOptim(lcc.f, lcc.g, shift, lmbda=lmbda, beta=beta, correlation=lcc.convolutions)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    out, losses = model.fit(optimizer, n=int(niter), threshold=2e-3)#, nsave=10)

    # we dump the results in a pickle file
    out_dict = {'out': out, 'losses': losses, 'lcc': lcc, }
    out_path = f'data/synthetic_findelen/out2_migrated_gridded_parallel.p'

    with open(out_path, 'wb') as file:
        pickle.dump(out_dict, file)
    print('\nsaving done')

    return out, losses

if __name__ == '__main__':

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
