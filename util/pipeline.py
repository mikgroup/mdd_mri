import numpy as np
import torch
import util.model as model
from util.fit import fit

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
dtype = torch.cuda.FloatTensor

def run_net(data,
            k=64,
            d=5,
            kernel_size=1,
            niter=10000,
            decoder=None,
            verbose=True,
            forward=False,
            mask=None):
    '''
    Creates and fits a MDD for the given data based on the specified parameters
    '''
    
    output_depth = data.shape[0]
    data_tensor = torch.from_numpy(data[None,...]).type(dtype)
    
    end_dim = data.shape[-2:]     
    delta = tuple([np.exp(np.log(dim/16.0)/d) for dim in end_dim]) # Calculates the upscaling factor for each layer based on network depth and data size
    up_size = [(int(16*delta[0]**i), int(16*delta[1]**i)) for i in range(1,d)] + [end_dim] # Explicitly creates list of sizes to upsample to at each layer
    
    if verbose:
        print(up_size)
    
    if decoder is None:
        decoder = model.MDD(depth=d,
                            k_out=output_depth,
                            k=k,
                            filter_size=kernel_size,
                            up_size=up_size).type(dtype)

    if forward:
        if mask is None:
            mask = np.ones(data.shape[1:])
        mask = torch.from_numpy(mask[None,...]).type(dtype)
        net = model.DecoderForward(decoder, mask).type(dtype)
    
    rn = 0.015
    LR = 0.01
    
    mse, ni, net = fit(net if forward else decoder,
                   data_tensor,
                   reg_noise_std=rn,
                   num_iter=niter,
                   LR=LR,
                   verbose=verbose,
                   forward=forward)
    
    out_img = net(ni.type(dtype)).data.cpu().numpy()[0]
    
    return out_img, mse, ni, net

def gen_inp(img, ksp=False,debug=False):
    '''
    Does the data pre-processing on the data to generate network input
    '''
    if ksp:
        norm_factor = np.max(np.abs(np.fft.ifft2(img, norm="ortho"))) # If data is subsampled ksp, normalize by largest value of ifft recon
    else:
        norm_factor = np.max(np.abs(img))
    img = img/norm_factor # Normalizing
    if img.dtype.kind == 'c':
        if debug:
            print("Complex Input")    
        inp = np.zeros((2, *img.shape))
        inp[0,:,:] = img.real # Splitting into real and imaginary channels
        inp[1,:,:] = img.imag
        rec_type = "c"
    elif len(img.shape) == 2:
        if debug:
            print("2D B&W Image")
        inp = img[None,:,:]
        rec_type = "b"
    else:
        if debug:
            print("2D RGB Image")
        inp = img.transpose(2, 0, 1)
        rec_type = "rgb"
    return inp, norm_factor, rec_type

def gen_img(net_out, rec_type, norm_factor=1.0, ksp=False):
    '''
    Does the data post-processing on the output of the network
    '''
    if rec_type == "c":
        img_out = net_out[0,:,:] + 1j*net_out[1,:,:]
    elif rec_type == "b":
        img_out = net_out[0,:,:]
    elif rec_type == "rgb":
        img_out = net_out.transpose(1, 2, 0)
    else:
        raise Exception("{} is an incorrect reconstruction type".format(rec_type))
    return img_out*norm_factor

def single_recon(data,
                 k=64,
                 d=5,
                 kernel_size=1,
                 niter=10000,
                 decoder=None,
                 verbose=True,
                 forward=False,
                 mask=None):
    '''
    Runs a pre-process -> network -> post-process chain for single channel data
    '''
    net_inp, norm_fac, rec_type = gen_inp(data)
    net_out,mse,ni,net = run_net(net_inp,k,d,kernel_size,niter,decoder,verbose,forward,mask)
    return gen_img(net_out, rec_type, norm_fac), mse, net

def multichannel_dat_prep(data, ksp=False):
    '''
    Does the data pre-processing on the multichannel data to generate network input
    '''
    assert type(data) is list
    inps, norm_factors, rec_types = list(zip(*[gen_inp(img, ksp) for img in data]))
    out_dims = [inp.shape[0] for inp in inps]
    im_shape = inps[0].shape[1:]
    net_inp = np.zeros((np.sum(out_dims), *(im_shape)))
    index = 0
    for i, out_dim in enumerate(out_dims):
        net_inp[index:index+out_dim] = inps[i]
        index += out_dim
    
    return net_inp, norm_factors, rec_types, out_dims

def multichannel_img_gen(net_out, norm_factors, rec_types, out_dims, ksp=False):
    '''
    Does the data post-processing on the output of the network for multichannel data
    '''
    index = 0
    imgs_out = []
    for i in range(len(rec_types)):
        net_out_i = net_out[index:index+out_dims[i]]
        imgs_out.append(gen_img(net_out_i, rec_types[i], norm_factors[i], ksp))
        index += out_dims[i]
        
    return imgs_out

def multichannel_recon(data,
                       k=64,
                       d=5,
                       kernel_size=1,
                       niter=10000,
                       decoder=None,
                       verbose=True,
                       forward=False,
                       mask=None):
    '''
    Runs a pre-process -> network -> post-process chain for multichannel data
    '''
    net_inp, norm_factors, rec_types, out_dims = multichannel_dat_prep(data, forward)
    net_out, mse, ni, net = run_net(net_inp,k,d,kernel_size,niter,decoder,verbose,forward,mask)
    return multichannel_img_gen(net_out, norm_factors, rec_types, out_dims, forward), mse, net