# Adapted from Heckel's fit.py

from torch.autograd import Variable
import torch
import torch.optim
import copy
import numpy as np

dtype = torch.cuda.FloatTensor

def fit(net,
        data,
        num_iter = 10000,
        LR = 0.01,
        opt_input = False,
        reg_noise_std = 0,
        reg_noise_decayevery = 100000,
        net_input = None,
        find_best=True,
        weight_decay=0,
        verbose=True,
        forward=False,
        lr_thresh=1e-6,
       ):

    if net_input is not None:
        print("input provided")
    else:
        num_channels = net.decoder.k[0] if forward else net.k[0]
        shape = [1,num_channels, 16, 16]
        if verbose:
            print("shape: ", shape)
        net_input = Variable(torch.zeros(shape))
        net_input.data.uniform_()
        net_input.data *= 1./10

    net_input_saved = net_input.data.clone()
    noise = net_input.data.clone()
    p = [x for x in net.parameters() ]

    if opt_input: # optimizer over the input as well
        net_input.requires_grad = True
        p += [net_input]

    error = np.zeros(num_iter)
    
    if verbose:
        print("optimize with adam", LR)
    optimizer = torch.optim.Adam(p, lr=LR,weight_decay=weight_decay)
    
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=500, min_lr = 1e-6)
#     cur_lr = LR
    mse = torch.nn.MSELoss()
#     mse = torch.nn.L1Loss()

    if find_best:
        best_net = copy.deepcopy(net)
        best_mse = 1000000.0

#     with open("folder_out.txt", "r") as file:
#         folder_out = file.read()
#     print(folder_out)
#     cooldown_niter = 0
    for i in range(num_iter):
        
#         if cur_lr <= lr_thresh:
#             cooldown_niter += 1
#             if cooldown_niter == 1000:
#                 break
        if reg_noise_std > 0:
            if i % reg_noise_decayevery == 0:
                reg_noise_std *= 0.7
            net_input = Variable(net_input_saved + (noise.normal_() * reg_noise_std))
        
        def closure():
            optimizer.zero_grad()
            out = net(net_input.type(dtype))

            loss = mse(out, data)
        
            loss.backward()
            error[i] = loss.data.cpu().numpy()
            
            if i % 10 == 0 and verbose:
                if i % 100 == 0:
                    net_in = Variable(net_input_saved).type(dtype)
                    obj = net.decoder if forward else net
                    out3 = obj(net_in.type(dtype)).data.cpu().numpy()[0]
                    
#                     fname = folder_out+"iter_{:06d}".format(i)
#                     print(fname)
#                     np.save(fname,out3)

                print("Iteration {0:5d} | Train loss {1:.4e} ".format(i, loss.data), '\r', end='')


            return loss
    
        loss = optimizer.step(closure)

#         scheduler.step(loss)
#         new_lr = optimizer.param_groups[0]['lr']
        
#         if new_lr != cur_lr:
#             if verbose:
# #                 print(len(optimizer.param_groups))
#                 print("\nLR Updated to {} at iter no. {}".format(new_lr, i))
#             cur_lr = new_lr
            
        if find_best:
            # if training loss improves by at least one percent, we found a new best net
            if best_mse > 1.005*loss.data:
                best_mse = loss.data
                if forward:
                    best_net = copy.deepcopy(net.decoder)
                else:
                    best_net = copy.deepcopy(net)
    if find_best:
        net = best_net
    return error,net_input_saved, net
