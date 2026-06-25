import torch
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('Qt5Agg')

import gaussian_windows as gw

from poly_interp import vec_polyfit2d, PytorchPolyFit2D, poly_surface

from itertools import product

import time

import math

import gc
import os

import torch.multiprocessing as mp
import torch.distributed as dist

try:
    mp.set_start_method('spawn')
except RuntimeError:
    pass

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def argmax2d(im):
    h, w = im.shape  # number of rows, number of columns
    index = im.argmax()  # flattened index

    j = index % w  # row
    i = index // w  # column

    return int(i), int(j)


def _get_h(f, l1, l2):
    #print(f.shape, l1, l2)
    height, width = f.shape
    h = torch.zeros((height, width)) + 1e-16

    if l1 > 0:
        if l2 > 0:
            h[l1:, l2:] = f[:height - l1, :width - l2]
        else:
            l2 = -l2
            h[l1:, :width - l2] = f[:height - l1, l2:]
    else:
        l1 = -l1
        if l2 > 0:
            h[:height - l1, l2:] = f[l1:, :width - l2]
        else:
            l2 = -l2
            h[:height - l1, :width - l2] = f[l1:, l2:]

    return h

def assert_search_sigma(search_sigma):
    try:
        sigma_y, sigma_x = search_sigma
    except ValueError:
        sigma_y = sigma_x = search_sigma[0]
        
    return sigma_y, sigma_x

class LCC2D:

    def __init__(self, f, g,
                 hlags,
                 wlags,
                 search_sigmas,
                 stride=1,
                 threshold=1,
                 nugget=1e-3,
                 verbose=True,
                 save_all_convolutions=False):

        assert f.shape == g.shape, 'f and g need to have the same shape'
        self.height, self.width = f.shape

        self.f = torch.tensor(f, dtype=torch.float32, device=device)
        self.g = torch.tensor(g, dtype=torch.float32, device=device)
        
        self.threshold = threshold
        self.verbose = verbose
        
        self.stride = stride
        
        self.hlags = hlags
        self.wlags = wlags
        
        
        if len(search_sigmas) <= 2:
            search_sigmas = [search_sigmas]
        
        self.search_sigmas = []
        for search_sigma in search_sigmas:
            self.search_sigmas.append(assert_search_sigma(search_sigma))
            
        print(self.search_sigmas)
        self.nugget = nugget
        self.save_all_convolutions = save_all_convolutions
        
        
    def fit(self):

        convolutions = []
        for search_sigma in self.search_sigmas:
            convolutions.append(self.lcc(search_sigma))
            torch.cuda.empty_cache()
       
        mean_conv = torch.stack(convolutions).mean(axis=0).to(device)
        self.dw, self.dh, self.values, self.convolutions = self.get_maximum_indices(mean_conv)
        self.poly, self.subdw, self.subdh = self.subpixel()
        
        if self.save_all_convolutions:
            self.convolutions = torch.stack(convolutions)
            self.convolutions = self.convolutions.reshape(-1, len(self.hlags), len(self.wlags), self.height // self.stride, self.width // self.stride)
        

    def process_sublags(self, rank, world_size, lags, f, g, cff, cgg, search_window, padding, inner_pad_height, inner_pad_width):
        
        batch_size = 64

        device = torch.device('cuda', rank)
        torch.cuda.set_device(device)
        
        # Set up distributed environment variables
        os.environ['MASTER_ADDR'] = 'localhost'  # or use the master node's IP if running across machines
        os.environ['MASTER_PORT'] = '12355'  # choose any available port
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['RANK'] = str(rank)

        print(device, rank, world_size)

        dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    
        start = rank * len(lags) // world_size
        end = min(len(lags) + 1, (rank + 1) * len(lags) // world_size)
        lags = lags[start:end]

        print(f'started computation of cuda:{rank} with {len(lags)} lags, {start} - {end}')

        g = g.to(device)
        search_window = search_window.to(device)
        cgg = cgg.to(device)

        all_convolutions = []

        for i0 in range(0, len(lags), batch_size):

            i1 = i0 + batch_size
            print(device, f'{i0} - {i1}')

            sublags = lags[i0:i1]

            # convolutions is a tensor of size (hlags * wlags, 1 channel, height, width)
            convolutions = torch.zeros((len(sublags), 1, self.height, self.width), device=device)
            cff_moved = torch.zeros((len(sublags), 1, self.height, self.width), device=device)

            # we need to move around one image and store it in an array
            # we do so for the image and it's normalization factor
            # this is the only part involving for loops
            for i, (l1, l2) in  enumerate(sublags):
                convolutions[i, 0] = _get_h(f, l1, l2)
                cff_moved[i, 0] = _get_h(cff, l1, l2)

            # this downsamples the image in the correct manner, taking into account the stride correctly
            cff_moved = F.avg_pool2d(cff_moved, kernel_size=self.stride, stride=self.stride)

            """# we need to downsample cff_moved to account for the stride
            cff_moved = cff_moved[:, :, 
                                  :self.height - inner_pad_height:self.stride, 
                                  :self.width - inner_pad_width:self.stride]"""
            
            # multiplication of the moved image with the unmoved image
            convolutions = convolutions * g.view(1, 1, self.height, self.width)
            
            # we localize the cross correlation with a gaussian blur
            convolutions = F.conv2d(convolutions, weight=search_window, padding=padding, stride=self.stride)
            

            # we normalize the convolutions by our factors
            convolutions = convolutions * cgg * cff_moved
            all_convolutions.append(convolutions.contiguous())

        convolutions = torch.cat(all_convolutions, dim=0)

        shapes = [torch.tensor(convolutions.shape, device=device) for _ in range(world_size)]
        dist.all_gather(shapes, torch.tensor(convolutions.shape, device=device))

        output_tensors = [torch.empty(shape.tolist(), device=device) for shape in shapes]
        dist.all_gather(output_tensors, convolutions)

        print(f'{device} gathered')

        if rank == 0:
            convolutions = torch.cat(output_tensors, dim=0)
            print(f"Total convolutions shape: {convolutions.shape}")
            torch.save(convolutions.to('cpu'), '/tmp/convolutions.pt')
            print(f'{device} saved')

        dist.barrier()
        dist.destroy_process_group()

    def lcc(self, search_sigma,):
        
        ngpus = torch.cuda.device_count()
        print(f'found {ngpus} GPUs')

        t0 = time.time()
        
        #padding_height = math.ceil(((self.stride - 1) * self.height + self.search_sigma[0] - self.stride)) 
        #padding_width = math.ceil(((self.stride - 1) * self.width + self.search_sigma[1] - self.stride))
        smooth_window = gw.torch_gaussian_window2d(1).to(self.f.dtype).to(device)
        search_window = gw.torch_gaussian_window2d(search_sigma).to(self.f.dtype).to(device)
        search_window = search_window / search_window.sum()
        
        kernel_shape = search_window.shape[2:]       
        padding_height = (kernel_shape[0] - self.stride) // 2
        padding_width = (kernel_shape[1] - self.stride) // 2
        
        padding_height = padding_height + 1 if self.height % 2 == 0 else padding_height
        padding_width = padding_width + 1 if self.width % 2 == 0 else padding_width
        
        inner_pad_height = 2 if self.height % self.stride != 0 else 0
        inner_pad_width = 2 if self.width % self.stride != 0 else 0

        print(inner_pad_height, inner_pad_width)

        padding = [padding_height, padding_width]
        
        if self.stride == 1:
            padding = 'same'
        
        print('padding', padding)
        
        # hale initially smooths the images with a sigma=1 window
        f = F.conv2d(self.f.view(1, 1, self.height, self.width),
                        weight=smooth_window, padding='same', stride=1)[0, 0]
        
        g = F.conv2d(self.g.view(1, 1, self.height, self.width),
                        weight=smooth_window, padding='same', stride=1)[0, 0]

        print('successfully convolved f and g')

        # THIS STEP here could be parallelized across multiple GPUs for different batches of lags
        lags = list(product(self.hlags, self.wlags))

        print('computed lags')


        # we also need to apply normalization factors to the images
        cff = 1/(F.conv2d((f * f).view(1, 1, self.height, self.width),
                            weight=search_window, padding='same', stride=1)).sqrt()[0, 0]
        print('computed cff')

        cgg = 1/(F.conv2d((g * g).view(1, 1, self.height, self.width),
                            weight=search_window, padding=padding, stride=self.stride)).sqrt()[0, 0]
        print('computed cgg')
        

        print('trying to spawn processes')
        mp.spawn(self.process_sublags, 
                 args=(ngpus, lags, f, g, cff, cgg, 
                       search_window, padding, inner_pad_height, inner_pad_width), nprocs=ngpus)

        convolutions = torch.load('/tmp/convolutions.pt')
            
        gc.collect()
        torch.cuda.empty_cache()

        if self.verbose:
            t1 = time.time()
            print(f'computing convolutions for {search_sigma} took {(t1 - t0):.2f} seconds')
        
        return convolutions.detach().to('cpu')

    def get_maximum_indices(self, convolutions):
        # what if we smoothed again in the xy space for every lag?
        #w = gw.torch_gaussian_window1d(self.smooth_sigma).to(device)
        #convolutions = F.conv1d(convolutions.permute(0, 1, 3, 2), w.view(1, 1, 1, -1), padding='same')
        #convolutions = F.conv1d(convolutions.permute(0, 1, 3, 2), w.view(1, 1, 1, -1), padding='same')
        
        t0 = time.time()
        
        '''
        from scipy.ndimage import convolve1d
        tmp = convolve1d(convolutions, w.numpy(), mode='constant', axis=-1)
        tmp = convolve1d(tmp, w.numpy(), mode='constant',  axis=-2)
        '''

        # we look for the maximum indices
        values, indices = torch.max(convolutions.reshape(len(self.hlags) * len(self.wlags), -1), dim=0)

        # reshaping them like the initial image was
        values = torch.reshape(values, (self.height // self.stride, self.width // self.stride))
        indices = torch.reshape(indices, (self.height // self.stride, self.width // self.stride))
        convolutions = convolutions.reshape(len(self.hlags), len(self.wlags), self.height // self.stride, self.width // self.stride)

        # changing the indices to i, j coordinates
        dw = (indices % len(self.wlags) * (self.wlags[1] - self.wlags[0]) - self.wlags.max())
        dh = (torch.div(indices, len(self.wlags), rounding_mode='floor') * (
                    self.hlags[1] - self.hlags[0]) - self.hlags.max())

        if self.verbose:
            t1 = time.time()
            print(f'\rfinding indices took {(t1 - t0):.2f} seconds')

        return dw, dh, values, convolutions
        
    def subpixel(self):

        if self.verbose:
            print(f'computing subpixel displacements')

        t0 = time.time()

        # taking into account where the index is too close to the given threshold for the rows
        dlh = (self.hlags[1] - self.hlags[0])
        dh = (self.dh - self.hlags.min()) / dlh
        h_index_left = dh < self.threshold
        h_index_right = dh >= (len(self.hlags) - self.threshold)
        dh = torch.where(h_index_left, self.threshold, dh.to(torch.int64))
        dh = torch.where(h_index_right, len(self.hlags) - self.threshold - 1, dh.to(torch.int64))

        # we do the same but for the columns
        dlw = (self.wlags[1] - self.wlags[0])
        dw = (self.dw - self.wlags.min()) / dlw
        w_index_left = dw < self.threshold
        w_index_right = dw >= (len(self.wlags) - self.threshold)
        dw = torch.where(w_index_left, self.threshold, dw.to(torch.int64))
        dw = torch.where(w_index_right, len(self.wlags) - self.threshold - 1, dw.to(torch.int64))

        # we need to extract the surrounding images to the integer maximum to be able to fit the quadratic surface
        i = dh.flatten()
        j = dw.flatten()

        # we reshape the convolutions tensor so that we can easily index it with flattened hlags * wlags coordinates
        conv = self.convolutions.view(len(self.hlags) * len(self.wlags), (self.height // self.stride) * (self.width // self.stride))
        offset = torch.arange(-self.threshold, self.threshold + 1, device=device)
        n = 2 * self.threshold + 1

        # calculate the indices for i2 and j2 using broadcasting
        # the coordinates are the same for every quadratic surface
        X = torch.stack(torch.meshgrid(offset, offset)).reshape(2, -1)
        i2, j2 = i.view(-1, 1) + X[0], j.view(-1, 1) + X[1]

        # compute the flat indices for i2 and j2
        flat_indices = (i2 * len(self.wlags) + j2).long()

        images = torch.gather(conv, 0, flat_indices.T)
        images = images.view(n, n, -1).moveaxis(0, 1).reshape(n * n, -1)

        # this vectorized implementation for fitting polynomials goes hard
        poly = PytorchPolyFit2D(X.to('cpu'), images.to('cpu'), order=2)
        dw, dh = poly.newton(nugget=self.nugget)

        if self.verbose:
            t1 = time.time()
            print(f'\rtook {(t1 - t0):.2f} seconds')

        return poly, \
            self.dw.to('cpu') + dw.reshape(self.height // self.stride, self.width // self.stride) * dlw, \
            self.dh.to('cpu') + dh.reshape(self.height // self.stride, self.width // self.stride) * dlh

    def debug_plot(self, aspect=1):

        coeffs = np.moveaxis(self.poly.coeffs.numpy().reshape(-1, self.height // self.stride, self.width // self.stride), 0, -1)
        convolutions = self.convolutions.cpu()
        # CLICKABLE FIGURE
        i0, j0 = self.height // 2 // self.stride, self.width // 2 // self.stride
        # creation of the figure and adding the main axes on which stuff will be plotted
        fig4 = plt.figure(figsize=(12, 6))
        gs = fig4.add_gridspec(2, 4)
        f_ax = fig4.add_subplot(gs[0, 0])
        g_ax = fig4.add_subplot(gs[1, 0])
        u_ax = fig4.add_subplot(gs[0, 1])
        v_ax = fig4.add_subplot(gs[1, 1])
        # main_ax is the one showing the correlation value with respect to vertical and horizontal lags
        main_ax = fig4.add_subplot(gs[:, 2:])
        main_ax.yaxis.tick_right()
        main_ax.yaxis.set_label_position('right')
        main_ax.set_xlabel('$l_2$')
        main_ax.set_ylabel('$l_1$', rotation=0)
        
        vmax = np.quantile(np.abs(self.f.cpu()), 0.95)
        
        # f_ax and g_ax are the before and after image
        f_ax.imshow(self.f.cpu(), aspect=aspect, cmap='Greys', origin='lower', vmin=-vmax, vmax=vmax, extent=[0, self.width // self.stride, 0, self.height // self.stride])
        f_ax.text(0.99, 0.99, '$f$', ha='right', va='top', transform=f_ax.transAxes)
        g_ax.imshow(self.g.cpu(), aspect=aspect, cmap='Greys', origin='lower', vmin=-vmax, vmax=vmax, extent=[0, self.width // self.stride, 0, self.height // self.stride])
        g_ax.text(0.99, 0.99, '$g$', ha='right', va='top', transform=g_ax.transAxes)

        from matplotlib.colors import TwoSlopeNorm
        '''unorm = TwoSlopeNorm(vmin=self.subdw.min(),
                             vcenter=0 if self.subdw.max() * self.subdw.min() < 0 else self.subdw.mean(),
                             vmax=self.subdw.max())
        vnorm = TwoSlopeNorm(vmin=self.subdh.min(),
                             vcenter=0 if self.subdh.max() * self.subdh.min() < 0 else self.subdh.mean(),
                             vmax=self.subdh.max())'''
        # u_ax and v_ax are the axes on which we show the estimated shifts
        u_ax.imshow(self.subdw, aspect=aspect, origin='lower', cmap='RdBu')#, norm=unorm)
        u_ax.text(0.99, 0.99, '$\hat{u}$', ha='right', va='top', transform=u_ax.transAxes)
        v_ax.imshow(self.subdh, aspect=aspect, origin='lower', cmap='RdBu')#, norm=vnorm)
        v_ax.text(0.99, 0.99, '$\hat{v}$', ha='right', va='top', transform=v_ax.transAxes)
        convnorm = TwoSlopeNorm(vcenter=0, vmin=-1, vmax=1)
        # conv_im is the 2d image that will be updated for every row and column inspected
        conv_im = main_ax.imshow(convolutions[:, :, i0, j0], aspect='auto', origin='lower',
                                 extent=[self.wlags.min() - 0.5,
                                         self.wlags.max() + 0.5,
                                         self.hlags.min() - 0.5,
                                         self.hlags.max() + 0.5],
                                 cmap='RdBu', norm=convnorm)
        # here we want to be able to have an updating polynomial surface
        dlw = self.wlags[1] - self.wlags[0]
        dlh = self.hlags[1] - self.hlags[0]
        I, J = argmax2d(convolutions[:, :, i0, j0])
        I, J = I * dlh, J * dlw
        xx, yy, z = poly_surface(coeffs[i0, j0], 0, 0, degree=2, threshold=self.threshold)
        conv_cntrs = [
            main_ax.contour(xx * dlh - self.hlags.max() + J, yy * dlw - self.wlags.max() + I, z, colors='k',
                            linewidths=0.5)]
        # those lines on main_ax show the estimated maximum value according to the polynomial surface
        lx = main_ax.axvline(self.subdw[i0, j0], c='red', lw=0.5)
        ly = main_ax.axhline(self.subdh[i0, j0], c='red', lw=0.5)
        main_ax.set_xlim(self.wlags.min(), self.wlags.max())
        main_ax.set_ylim(self.hlags.min(), self.hlags.max())
        # we are adding moving circles that show the moving windows
        patches = []
        '''from matplotlib.patches import Ellipse
        for ax1 in (f_ax, g_ax):
            for i in [1, 2, 3]:
                circle = Ellipse((j0, i0),
                                 width=i * self.search_sigma[1],
                                 height=i * self.search_sigma[0],
                                 alpha=0.1 * (4 - i),
                                 facecolor='tab:red', ec='tab:red',
                                 linewidth=1.0)
                ax1.add_patch(circle)
                patches.append(circle)'''
        # here we are adding horizontal and vertical lines that show the central point of the window
        horizontal_lines = []
        vertical_lines = []
        for ax1 in (u_ax, v_ax):
            horizontal_lines.append(ax1.axhline(i0, c='red', lw=0.25))
            vertical_lines.append(ax1.axvline(j0, c='red', lw=0.25))

        # updating function
        def on_click(event):
            # we want to make sure that we are actually clicking in the good axes
            if event.inaxes is None:
                return
            if event.inaxes != main_ax:
                # fetching the x and y position of the cursor
                j, i = int(event.xdata), int(event.ydata)
                # updating the main_ax image
                conv_im.set_data(convolutions[:, :, i, j])
                # updating the contours estimated from the polynomial surface
                # here this part is a bit tedious as we have to
                # compute the surface, remove and redraw the contours everytime
                # it doesn't seem to be too laggy however
                for tp in conv_cntrs[0].collections:
                    tp.remove()
                I, J = argmax2d(convolutions[:, :, i, j])
                I, J = I * dlh, J * dlw
                xx, yy, z = poly_surface(coeffs[i, j], 0, 0,
                                         degree=2,
                                         threshold=self.threshold)
                conv_cntrs[0] = main_ax.contour(xx * dlh - self.hlags.max() + J, yy * dlw - self.wlags.max() + I,
                                                z, colors='k', linewidths=0.5)
                # updating the horizontal and vertical lines and the moving windows
                for l1 in horizontal_lines:
                    l1.set_ydata([i, i])
                for l2 in vertical_lines:
                    l2.set_xdata([j, j])
                #for c in patches:
                #    c.set_center((j, i ))
                lx.set_xdata([self.subdw[i, j], self.subdw[i, j]])
                ly.set_ydata([self.subdh[i, j], self.subdh[i, j]])
            fig4.canvas.draw_idle()  # this is necessary so that it moves

        # this function ensures that we can keep clicking and moving
        def on_move(event):
            if event.button == 1:
                on_click(event)

        # connecting the figure to the different event functions
        fig4.canvas.mpl_connect('button_press_event', on_click)
        fig4.canvas.mpl_connect('motion_notify_event', on_move)
        
        # ensuring that zooming on one axes zooms on the others
        for ax in (g_ax, u_ax, v_ax):
            ax.sharex(f_ax)
            ax.sharey(f_ax)
        
        # ensuring that main_ax's limits do not change
        main_ax.set_xlim(main_ax.get_xlim())
        main_ax.set_ylim(main_ax.get_ylim())



if __name__ == '__main__':
    im0 = np.load('data/spongebob_warp2_0.npy', allow_pickle=True).astype(np.float32)
    im5 = np.load('data/spongebob_warp2_5.npy', allow_pickle=True).astype(np.float32)
    vf = np.load('data/vf_warp2.npy', allow_pickle=True) * 5

    maxlag = 20
    dl = 2
    wlags = np.arange(-maxlag, maxlag + dl, dl)
    hlags = np.arange(-maxlag, maxlag + dl, dl)
    
    
    search_sigma = [(10, 10), (20, 20), (30, 30), (10, 20), (10, 30), (20, 10), (30, 10), (30, 20)]

    lcc = LCC2D(im0, im5,
                hlags, wlags,
                search_sigma,
                threshold=3,
                save_all_convolutions=True)
    lcc.fit()
    
    out_dict = {'lcc': lcc, }
    out_path = f'out_tmp.p'
    import pickle
    with open(out_path, 'wb') as file:
        pickle.dump(out_dict, file)
    print('\nsaving done')
    
    lcc.debug_plot()
    plt.show()
    '''
    shift = torch.stack((lcc.subdw, lcc.subdh)).to(torch.float32)

    lmbda = 1e1
    beta = 0  # 1e5

    model = ShiftOptim(lcc.f, lcc.g, shift, lmbda=lmbda, beta=beta, correlation=lcc.convolutions)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    out, losses = model.fit(optimizer, n=3000)

    from image_warping import warp_cv2

    plt.figure()
    plt.plot(losses[:, 0], ls='dashed', label='$\epsilon$')
    plt.plot(losses[:, 1], ls='dashed', label=f'$\lambda$={lmbda:.1e}')
    plt.plot(losses[:, 2], ls='dashed', label=f'$\\beta$={beta:.1e}')
    plt.plot(losses.sum(axis=1), c='k', label='$\Sigma$')
    plt.fill_between(np.arange(len(losses)), 0, losses.sum(axis=1), alpha=0.3, fc='tab:grey', ec='k')
    plt.xlim(0, len(losses) - 1)
    plt.ylim(0, None)
    plt.legend()
    plt.xlabel('iteration')
    plt.ylabel('loss')

    fig, axs = plt.subplots(1, 5, sharex='all', sharey='all', figsize=(14, 3))
    axs[0].imshow(im0, origin='lower')
    axs[0].text(0.99, 0.99, f'Original', ha='right', va='top', transform=axs[0].transAxes)

    axs[1].imshow(im5, origin='lower')
    axs[1].text(0.99, 0.99, f'Warped', ha='right', va='top', transform=axs[1].transAxes)

    axs[2].imshow(warp_cv2(im0, torch.stack((lcc.dw, lcc.dh)).cpu().numpy()), origin='lower')
    axs[2].text(0.99, 0.99, f'Integer disp.', ha='right', va='top', transform=axs[2].transAxes)

    axs[3].imshow(warp_cv2(im0, shift.cpu().numpy()), origin='lower')
    axs[3].text(0.99, 0.99, f'Subpixel disp.', ha='right', va='top', transform=axs[3].transAxes)

    axs[4].imshow(model.forward().cpu().detach(), origin='lower')
    axs[4].text(0.99, 0.99, f'Optimized disp.', ha='right', va='top', transform=axs[4].transAxes)

    fig.subplots_adjust(hspace=0.05, wspace=0.05)

    fig, axs = plt.subplots(3, 2, sharex='all', sharey='all')
    axs[0, 0].imshow(shift[0].cpu(), origin='lower', vmin=vf[0].min(), vmax=vf[0].max())
    axs[0, 0].text(0.99, 0.99, 'lcc $\hat{v}_x$', ha='right', va='top', transform=axs[0, 0].transAxes)

    axs[0, 1].imshow(shift[1].cpu(), origin='lower', vmin=vf[1].min(), vmax=vf[1].max())
    axs[0, 1].text(0.99, 0.99, 'lcc $\hat{v}_y$', ha='right', va='top', transform=axs[0, 1].transAxes)

    axs[1, 0].imshow(out[0], origin='lower', vmin=vf[0].min(), vmax=vf[0].max())
    axs[1, 0].text(0.99, 0.99, 'optim. $\hat{v}_x$', ha='right', va='top', transform=axs[1, 0].transAxes)

    axs[1, 1].imshow(out[1], origin='lower', vmin=vf[1].min(), vmax=vf[1].max())
    axs[1, 1].text(0.99, 0.99, 'optim $\hat{v}_y$', ha='right', va='top', transform=axs[1, 1].transAxes)

    axs[2, 0].imshow(vf[0], origin='lower', vmin=vf[0].min(), vmax=vf[0].max())
    axs[2, 0].text(0.99, 0.99, '$v_x$', ha='right', va='top', transform=axs[2, 0].transAxes)

    axs[2, 1].imshow(vf[1], origin='lower', vmin=vf[1].min(), vmax=vf[1].max())
    axs[2, 1].text(0.99, 0.99, '$v_y$', ha='right', va='top', transform=axs[2, 1].transAxes)

    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    fig.savefig('deformation.png')
    plt.show()'''
    

"""
im0 = (im0 - im0.min())/(im0.max() - im0.min())
im5 = (im5 - im5.min())/(im5.max() - im5.min())
fig = plt.figure(figsize=(12.5, 3))
gs = fig.add_gridspec(2, 8)
ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[1, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 1])
h_ax = fig.add_subplot(gs[:, 2:4])
g_ax = fig.add_subplot(gs[:, 4:6])
corr_ax = fig.add_subplot(gs[:, 6:])
i, j = 150, 120
ax0.imshow(im0, origin='lower')
ax1.imshow(im5, origin='lower')
ax2.imshow(im0, origin='lower')
ax3.imshow(im5, origin='lower')
offset = 30
for ax in (ax2, ax3):
    ax.set_xlim(j-offset, j+offset)
    ax.set_ylim(i-offset, i+offset)
ax0.set_title('original', loc='left', size='small', pad=0.5)
ax1.set_title('deformed', loc='left', size='small', pad=0.5)

h, w = im0.shape
h = np.arange(h)
w = np.arange(w)
ww, hh = np.meshgrid(w, h)
gaussw = np.exp((-(w - j)**2)/10**2)
gaussh = np.exp((-(h - i)**2)/10**2)
gauss = np.outer(gaussh, gaussw)
h_ax.imshow(im0 * im5, origin='lower')
h_ax.set_xlim(j - offset, j + offset)
h_ax.set_ylim(i - offset, i + offset)
h_ax.set_xticks([])
h_ax.set_yticks([])
h_ax.set_title('original $\\times$ deformed', loc='left', size='small', pad=0.5)

g_ax.imshow(im0 * im5 * gauss, origin='lower')
g_ax.set_xlim(j - offset, j + offset)
g_ax.set_ylim(i - offset, i + offset)
g_ax.set_xticks([])
g_ax.set_yticks([])
g_ax.set_title('gaussian window', loc='left', size='small', pad=0.5)


corr_ax.imshow(lcc.convolutions[:, :, i, j], extent=[-20, 20, -20, 20], origin='lower')
corr_ax.set_xlabel('$l_w$')
corr_ax.set_ylabel('$l_h$', rotation=0)
corr_ax.yaxis.tick_right()
corr_ax.yaxis.set_label_position('right')
corr_ax.set_title(f'Localized cross correlation', loc='left', size='small', pad=0.5)
for ax in (ax0, ax1, ax2, ax3):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.scatter(j, i, c='red', ec='k', s=15)


fig.subplots_adjust(wspace=0.00)
fig.savefig('powerpoints/figures/lcc2d_exemple.png', dpi=800, bbox_inches='tight')"""

"""
# figure showing the deformation field as a streamplot
fig, axs = plt.subplots(1, 3, sharex='all', sharey='all', figsize=(15, 4))
height, width = im0.shape
H, W = np.arange(height), np.arange(width)
ww, hh = np.meshgrid(W, H)
norm = np.linalg.norm(vf, axis=0)
axs[0].streamplot(ww, hh, vf[0], vf[1], color=norm, linewidth=0.5)

norm2 = np.linalg.norm(shift, axis=0)
norm2[norm2 >= norm.max()] = norm.max()
axs[1].streamplot(ww, hh, shift[0], shift[1], color=norm2, linewidth=0.5)

norm3 = np.linalg.norm(out, axis=0)
norm3[norm3 >= norm.max()] = norm.max()
axs[2].streamplot(ww, hh, out[0], out[1], color=norm3, linewidth=0.5)

axs[0].set_title('original field', loc='left')
axs[1].set_title('LCC approx.', loc='left')
axs[2].set_title('optimized LCC', loc='left')
for ax in axs:
    ax.set_aspect(1)
    ax.set_ylim(0, height)
    ax.set_xlim(0, width)
fig.subplots_adjust(wspace=0.05)
"""
