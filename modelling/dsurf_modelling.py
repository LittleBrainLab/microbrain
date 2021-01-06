#!/usr/local/bin/python
import os,sys,getopt
import shutil
import glob
from os import system
from os import path
from time import time

import numpy as np

import nibabel as nib

from dipy.denoise.noise_estimate import estimate_sigma

import dipy.reconst.dti as dti
import dipy.reconst.fwdti as fwdti
import dipy.reconst.dki as dki

from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table
from dipy.reconst.dti import fractional_anisotropy, color_fa

sys.path.append('../preprocessing/')
import dsurf_preproc as preproc

def output_DTI_maps_multishell(fname, fmask, bvals, bvecs, tensorDir, shells = [0,500,1000], free_water = False, tolerance = 100, nlls = False, ols = False):
    print("Starting DTI Map Generation")
    fbasename = os.path.basename(fname)

    if free_water:
        suffix = '_FW_'
    elif ols:
        suffix = '_OLS_'
    elif nlls:
        suffix = '_NLLS_'
    else:
        suffix = '_'

    for shell in shells:
        suffix = suffix + 'b' + str(shell)

    fa_fname = tensorDir + fbasename.replace('.nii', suffix + '_FA.nii')
    md_fname = tensorDir + fbasename.replace('.nii', suffix + '_MD.nii')
    if not os.path.exists(fa_fname):

        img = nib.load(fname)
        data = img.get_data()

        mask_img = nib.load(fmask)
        mask_data = mask_img.get_data()

        # Only use the b0s, and single tensor
        b_idx = []
        for shell in shells:
            b_idx = np.append(b_idx, np.squeeze(np.array(np.where(np.logical_and(bvals < shell + tolerance, bvals > shell - tolerance)))))
        b_idx = b_idx.astype(int)
        print(b_idx) 
        print('Bvals',b_idx.shape)


        bvals = bvals[b_idx]
        bvecs = bvecs[b_idx]
        data = data[:,:,:,b_idx]
        for i in range(0,data.shape[3]):
            data[mask_data==0,i] = 0


        gtab = gradient_table(bvals, bvecs)
        fa_fname = tensorDir + fbasename.replace('.nii', suffix + '_FA.nii')

        t = time()
        if free_water:
            print("Fitting Free Water Tensor")
            tenmodel = fwdti.FreeWaterTensorModel(gtab)
        elif ols:
            tenmodel = dti.TensorModel(gtab, fit_method='LS')
        elif nlls:
            tenmodel = dti.TensorModel(gtab, fit_method='NLLS', weighting='gmm')
        else:
            print("Fitting Tensor")
            tenmodel = dti.TensorModel(gtab)
        
        tenfit = tenmodel.fit(data)
        print("Fitting Complete Total time: ", time() - t)
        
        if not os.path.exists(tensorDir):
            os.system('mkdir ' + tensorDir)

        outbase = tensorDir + fbasename

        FA = fractional_anisotropy(tenfit.evals)
        FA[np.isnan(FA)] = 0

        fa_img = nib.Nifti1Image(FA.astype(np.float32), img.affine)
        nib.save(fa_img, outbase.replace('.nii', suffix + '_FA.nii'))

        evecs_img = nib.Nifti1Image(tenfit.evecs.astype(np.float32), img.affine)
        nib.save(evecs_img, outbase.replace('.nii', suffix + '_EVECS.nii'))

        lt_tensor = tenfit.lower_triangular()
        tensor_img = nib.Nifti1Image(lt_tensor*1000, img.affine)
        nib.save(tensor_img, outbase.replace('.nii', suffix + '_tensor.nii'))

        dir_img = nib.Nifti1Image(np.squeeze(tenfit.directions.astype(np.float32)), img.affine)
        nib.save(dir_img, outbase.replace('.nii',suffix + '_Primary_Direction.nii'))

        MD = dti.mean_diffusivity(tenfit.evals)
        nib.save(nib.Nifti1Image(MD.astype(np.float32), img.affine), outbase.replace('.nii',suffix + '_MD.nii'))

        RD = tenfit.rd
        nib.save(nib.Nifti1Image(RD.astype(np.float32), img.affine), outbase.replace('.nii',suffix + '_RD.nii'))

        AD = tenfit.ad
        nib.save(nib.Nifti1Image(AD.astype(np.float32), img.affine), outbase.replace('.nii',suffix + '_AD.nii'))

        FA = np.clip(FA, 0, 1)
        RGB = color_fa(FA, tenfit.evecs)
        nib.save(nib.Nifti1Image(np.array(255 * RGB, 'uint8'), img.affine), outbase.replace('.nii',suffix + '_FA_RGB.nii'))

        if free_water:
            FW = tenfit.f
            nib.save(nib.Nifti1Image(FW.astype(np.float32), img.affine), outbase.replace('.nii',suffix + '_FW.nii'))
    else:
        print("DTI Tensor Files Already Exist: Skipping")

    return fa_fname, md_fname

def output_DKI_maps(fname, fmask, bvals, bvecs, dkiDir):
    print("Starting DKI Map Generation")
    prefix = 'dki'
    fbasename = os.path.basename(fname) 
    fa_fname = dkiDir + fbasename.replace('.nii', '_' + prefix + '_FA.nii')
    if not os.path.exists(fa_fname):

        img = nib.load(fname)
        data = img.get_data()

        mask_img = nib.load(fmask)
        mask_data = mask_img.get_data()

        for i in range(0,data.shape[3]):
            data[mask_data==0,i] = 0


        gtab = gradient_table(bvals, bvecs)
        
        print("Fitting DKI")
        t = time()
        dkimodel = dki.DiffusionKurtosisModel(gtab)
        dkifit = dkimodel.fit(data)
        print("Fitting Complete Total time:", time() - t)
        
        if not os.path.exists(dkiDir):
            os.system('mkdir ' + dkiDir)

        outbase = dkiDir + fbasename

        FA = dkifit.fa
        FA[np.isnan(FA)] = 0

        fa_img = nib.Nifti1Image(FA.astype(np.float32), img.affine)
        nib.save(fa_img, outbase.replace('.nii', '_' + prefix + '_FA.nii'))

        MD = dkifit.md
        nib.save(nib.Nifti1Image(MD.astype(np.float32), img.affine), outbase.replace('.nii','_' + prefix + '_MD.nii'))

        RD = dkifit.rd
        nib.save(nib.Nifti1Image(RD.astype(np.float32), img.affine), outbase.replace('.nii','_' + prefix + '_RD.nii'))

        AD = dkifit.ad
        nib.save(nib.Nifti1Image(AD.astype(np.float32), img.affine), outbase.replace('.nii','_' + prefix + '_AD.nii'))

        FA = np.clip(FA, 0, 1)
        RGB = color_fa(FA, dkifit.evecs)
        nib.save(nib.Nifti1Image(np.array(255 * RGB, 'uint8'), img.affine), outbase.replace('.nii','_' + prefix + '_FA_RGB.nii'))
    
        MK = dkifit.mk(0,3)
        nib.save(nib.Nifti1Image(MK.astype(np.float32), img.affine), outbase.replace('.nii','_' + prefix + '_MK.nii'))

        RK = dkifit.rk(0,3)
        nib.save(nib.Nifti1Image(RK.astype(np.float32), img.affine), outbase.replace('.nii','_' + prefix + '_RK.nii'))

        AK = dkifit.ak(0,3)
        nib.save(nib.Nifti1Image(AK.astype(np.float32), img.affine), outbase.replace('.nii','_' + prefix + '_AK.nii'))
    
        evecs_img = nib.Nifti1Image(dkifit.evecs.astype(np.float32), img.affine)
        nib.save(evecs_img, outbase.replace('.nii', '_' + prefix + '_EVECS.nii'))
    
        lt_tensor = dkifit.lower_triangular()
        tensor_img = nib.Nifti1Image(lt_tensor*1000, img.affine)
        nib.save(tensor_img, outbase.replace('.nii', '_' + prefix + '_tensor.nii'))

        dir_img = nib.Nifti1Image(np.squeeze(dkifit.directions.astype(np.float32)), img.affine)
        nib.save(dir_img, outbase.replace('.nii', '_' + prefix + '_Primary_Direction.nii'))
    else:
        print("DKI Files Already Exist: Skipping")

    return

