import numpy as np
import nibabel as nib
from IPython.display import clear_output
import h5py

def update_progress(progress):
    bar_length = 20
    progress = min(max(progress, 0), 1)
    block = int(round(bar_length * progress))
    clear_output(wait = True)
    text = "Progress: [{0}] {1:.1f}%".format( "#" * block + "-" * (bar_length - block), progress * 100)
    print(text)

# From https://github.com/MRSRL/dl-cs/blob/master/data_prep.py
def ismrmrd_to_np(filename):
    """Read ISMRMRD data file to numpy array"""
    import ismrmrd
    print('Loading file {}...'.format(filename))
    dataset = ismrmrd.Dataset(filename, create_if_needed=False)
    header = ismrmrd.xsd.CreateFromDocument(dataset.read_xml_header())
    num_kx = header.encoding[0].encodedSpace.matrixSize.x
    num_ky = header.encoding[0].encodingLimits.kspace_encoding_step_1.maximum
    num_slices = header.encoding[0].encodingLimits.slice.maximum + 1
    num_channels = header.acquisitionSystemInformation.receiverChannels

    try:
        rec_std = dataset.read_array('rec_std', 0)
        rec_weight = 1.0 / (rec_std**2)
        rec_weight = np.sqrt(rec_weight / np.sum(rec_weight))
        print('  Using rec std...')
    except Exception:
        rec_weight = np.ones(num_channels)
    opt_mat = np.diag(rec_weight)
    kspace = np.zeros([num_channels, num_slices, num_ky, num_kx],
                      dtype=np.complex64)
    num_acq = dataset.number_of_acquisitions()
    for i in range(num_acq):
        update_progress(i/num_acq)
        acq = dataset.read_acquisition(i)
        i_ky = acq.idx.kspace_encode_step_1  # pylint: disable=E1101
        # i_kz = acq.idx.kspace_encode_step_2 # pylint: disable=E1101
        i_slice = acq.idx.slice  # pylint: disable=E1101
        data = np.matmul(opt_mat.T, acq.data)
        kspace[:, i_slice, i_ky, :] = data * ((-1)**i_slice)

    dataset.close()
    
    tmp = np.fft.fftshift(kspace, axes=(1,))
    tmp = np.fft.fftn(tmp, axes=(1,), norm=None)
    ksp = np.fft.ifftshift(tmp, axes=(1,))
    
    return ksp

def pad_im(im):
    n = 256
    dim = len(im.shape)
    assert dim == 2
    l,m = im.shape
    new_im = np.zeros((n,m), dtype=im.dtype)
    new_im2 = np.zeros((n, n), dtype=im.dtype)
    a = (n-l)//2
    b = (n-m)//2
    if a > 0:
        new_im[a:-a-1,:] = im[:,:]
    elif a == 0:
        new_im[:,:] = im
    elif a < 0:
        new_im[:,:] = im[-a:a,:]

    if b > 0:
        new_im2[:,b:-b-1] = new_im
    elif b == 0: 
        new_im2[:,:] = new_im
    elif b < 0:
        new_im2[:,:] = new_im[:,-b:b]
    
    return new_im2

def ksp_to_im(ksp, coils=False):
    if len(ksp.shape)==3:
        return np.fft.ifftshift(np.fft.ifftn(np.fft.fftshift(ksp), axes=(0, 1, 2)))
    im_vol_ch = np.zeros(ksp.shape, dtype="complex64")
    for i in range(ksp.shape[0]):
        im_vol_ch[i,:,:,:] = np.fft.ifftshift(np.fft.ifftn(np.fft.fftshift(ksp[i,:,:,:]), axes=(0, 1, 2)))
    if coils:
        return im_vol_ch
    im_vol = np.linalg.norm(im_vol_ch, axis=0)
    return im_vol
    
def read_data(filename, img=False):
    ext = filename.split(".")[-1]
    if ext == "h5":
        ksp = ismrmrd_to_np(filename)
    
    if img:     
        if ext == "mnc":
            img = nib.load(filename)
            im_vol = img.get_data()
        elif ext == "h5":
            im_vol = ksp_to_im(ksp)
        else:
            raise Exception("Unsupported file format")
        return im_vol
    else:
        if ext == "mnc":
            raise Exception("Can not return kspace data for .mnc file format")
        elif ext == "h5":
            return ksp
        else:
            raise Exception("Unsupported file format")     
    
def gen_imgs_from_3d_data(filein, fileout, axis=0):
    assert 0<=axis<=2
    img_vol = read_data(filein, img=True)
    img_vol = np.moveaxis(img_vol, axis, 0)
    dir_name = "out/" + fileout
    if not os.path.exists("out"):
        os.mkdir("out")
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    else:
        files = os.listdir(dir_name)
        if len(files) != 0:
            for file in files:
                os.remove(dir_name+"/"+file)
    for i, img in enumerate(img_vol):
        np.save(dir_name+"/slice_{:03d}".format(i),pad_im(img))

def read_img_file(folder, slice_no):
    try:
        im = np.load("out/{}/slice_{:03d}.npy".format(folder, slice_no))
        return im
    except:
        raise Exception("Unable to load file. Make sure folder and slice wanted exist")
        
def normalize(img):
    return img/np.max(np.abs(img))

def gen_mask(size, calib, acc_fac):
    assert len(size) == 2
    c = size[1]//2
    mask = np.zeros(size)
    mask[:,::acc_fac] = np.ones(int(np.ceil(size[1]/acc_fac)))
    mask[:,c-calib//2:c+calib//2] = np.ones((size[0], calib))
    r_eff = np.prod(mask.shape)/np.count_nonzero(mask)
    return mask, r_eff

def gen_noisy_img(img,psnr):
    '''
    Generates a noisy version of img using a zero-mean gaussian
    '''
    sigma = img.max()/psnr
    noise = np.random.normal(scale=sigma, size=img.shape)
    noisy_img = np.clip( img + noise , img.min(), img.max()).astype(np.float32)
    return noisy_img

def fftc(x):
    '''
    Performs a centered, normalized FFT on the last two axes of x
    '''
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x, axes=(-2, -1)),norm="ortho"), axes=(-2, -1))

def ifftc(x):
    '''
    Performs a centered, normalized IFFT on the last two axes of x
    '''
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(x, axes=(-2, -1)),norm="ortho"), axes=(-2, -1))

# From the FastMRI challenge
def save_reconstructions(reconstructions, out_dir):
    """
    Saves the reconstructions from a model into h5 files that is appropriate for submission
    to the leaderboard.
    Args:
        reconstructions (dict[str, np.array]): A dictionary mapping input filenames to
            corresponding reconstructions (of shape num_slices x height x width).
        out_dir (pathlib.Path): Path to the output directory where the reconstructions
            should be saved.
    """
    out_dir.mkdir(exist_ok=True)
    for fname, recons in reconstructions.items():
        with h5py.File(out_dir / fname, 'w') as f:
            f.create_dataset('reconstruction', data=recons)
            
def crop(im, dim):
    '''
    Performs a center crop to the specificed dimensions
    '''
    c = np.array(im.shape[-2:])//2
    w0 = dim[0]//2
    w1 = dim[1]//2
    cond = len(im.shape)==3
    return im[:,c[0]-w0:c[0]+w0, c[1]-w1:c[1]+w1] if cond else im[c[0]-w0:c[0]+w0, c[1]-w1:c[1]+w1]
