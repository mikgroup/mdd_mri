# Untrained Modified Deep Decoder for Joint Denoising and Parallel Imaging Reconstruction

This is the public GitHub repository for ISMRM 2020 Conference submission 3585, which can be found [here](https://submissions2.mirasmart.com/ISMRM2020/ViewSubmission.aspx?sbmID=134&validate=false).

Contact: Sukrit Arora (sukrit.arora@berkeley.edu)

## Synopsis

An untrained deep learning model based on a Deep Decoder was used for image denoising and parallel imaging reconstruction. The flexibility of the modified Deep Decoder to output multiple images was exploited to jointly denoise images from adjacent slices and to reconstruct multi-coil data without pre-determed coil sensitivity profiles. Higher PSNR values were achieved compared to the traditional methods of denoising (BM3D) and image reconstruction (Compressed Sensing). This untrained method is particularly attractive in scenarios where access to training data is limited, and provides a possible alternative to conventional sparsity-based image priors.

## Installation

Make sure to download the relevant raw data files, which can be found [here](https://drive.google.com/file/d/1yItSevLQ17-zJ7Tlv8mI47OdfKZ43PeC/view?usp=sharing). To create the `data/` folder, put the tar file in the main folder and run:

```
tar -xvf mdd_mri_data.tar.gz
```

To get the necessary packages, you can create a conda environment using the provided `.yml` file by running:

```
conda env create -f mdd_mri_env.yml
conda activate mdd_mri
```

Or, you can take a look at the `dependencies.txt` file to see a list of relevant packages and versions.

Note that either way you will need to have the [BART Toolbox](https://mrirecon.github.io/bart) in order to perform the Compressed Sensing reconstruction.

## Usage

There are two jupyter notebooks that demo the two tasks the network can be used for, named `denoise_demo.ipynb` and `recon_demo.ipynb`. These can be run by running

```
jupyter notebook
```

Which will open a browser window from which you can select the notebook you want to run.

For each demo notebook, there is a set of flags that can be set at the bottom of the imports cell that changes the behavior of the notebook (according to the description in the comment next to it)

### File Structure Breakdown

All of the code can be found in the `util` folder. At a high level, the following files are responsible for the description that follows it:

`pipeline.py` - main interface that handles data pre-processing, creating and fitting network

`model.py` - has the PyTorch module class definitions for the MDD

`fit.py` - has the code that fits the network to the given data

## Reference

Take a look at Heckel's [Deep Decoder](https://github.com/reinhardh/supplement_deep_decoder), which this work was based on.
