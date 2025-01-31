#!/usr/local/bin/python
import os
from os import path
from time import time
import subprocess

import numpy as np

import nibabel as nib

import dipy.reconst.dti as dti
import dipy.reconst.fwdti as fwdti
import dipy.reconst.dki as dki

from dipy.core.gradients import gradient_table
from dipy.reconst.dti import fractional_anisotropy, color_fa


def fsl_ext():
    """
    Gets the preferred image extension used by FSL

    Parameters
    ----------
    none

    Returns
    -------
    extension: a string containing the preferred output
    """

    fsl_extension = ''
    if os.environ['FSLOUTPUTTYPE'] == 'NIFTI':
        fsl_extension = '.nii'
    elif os.environ['FSLOUTPUTTYPE'] == 'NIFTI_GZ':
        fsl_extension = '.nii.gz'
    return fsl_extension


def output_DTI_maps_multishell(fname, fmask, bvals, bvecs, tensorDir, shells=[0, 500, 1000], free_water=False, tolerance=100, nlls=False, ols=False):
    """
    Generates DTI maps from a given diffusion MRI image

    Parameters
    ----------
    fname: string
        The path to the diffusion MRI image
    fmask: string
        The path to the mask image
    bvals: array
        The b-values for the diffusion MRI image
    bvecs: array
        The b-vectors for the diffusion MRI image
    tensorDir: string
        The path to the output directory
    shells: array
        The b-values to use for the tensor model
    free_water: boolean
        Whether to use the free water model
    tolerance: int
        The tolerance for the b-values
    nlls: boolean
        Whether to use the non-linear least squares method
    ols: boolean
        Whether to use the ordinary least squares method

    Returns
    -------
    fa_fname: string
        The path to the fractional anisotropy image
    md_fname: string
        The path to the mean diffusivity image
    fa_rgb_fname: string
        The path to the fractional anisotropy RGB image
    ad_fname: string
        The path to the axial diffusivity image
    rd_fname: string
        The path to the radial diffusivity image
    evec_fname: string
        The path to the eigenvectors image
    pevec_fname: string
        The path to the primary eigenvectors image
    tensor_fname: string
        The path to the tensor image
    """

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

    fa_fname = tensorDir + \
        fbasename.replace(fsl_ext(), suffix + '_FA' + fsl_ext())
    fa_rgb_fname = tensorDir + \
        fbasename.replace(fsl_ext(), suffix + '_FA_RGB' + fsl_ext())
    md_fname = tensorDir + \
        fbasename.replace(fsl_ext(), suffix + '_MD' + fsl_ext())
    ad_fname = tensorDir + \
        fbasename.replace(fsl_ext(), suffix + '_AD' + fsl_ext())
    rd_fname = tensorDir + \
        fbasename.replace(fsl_ext(), suffix + '_RD' + fsl_ext())
    evec_fname = tensorDir + \
        fbasename.replace(fsl_ext(), suffix + '_EVECS' + fsl_ext())
    eval_fname = tensorDir + \
        fbasename.replace(fsl_ext(), suffix + '_EVALS' + fsl_ext())
    pevec_fname = tensorDir + \
        fbasename.replace(fsl_ext(), suffix + '_Primary_Direction' + fsl_ext())
    tensor_fname = tensorDir + \
        fbasename.replace(fsl_ext(), suffix + '_tensor' + fsl_ext())
    if not os.path.exists(fa_fname):

        img = nib.load(fname)
        data = img.get_fdata(dtype=np.float32)

        mask_img = nib.load(fmask)
        mask_data = mask_img.get_fdata()

        # Only use the b0s, and single tensor
        b_idx = []
        for shell in shells:
            b_idx = np.append(b_idx, np.squeeze(np.array(np.where(
                np.logical_and(bvals < shell + tolerance, bvals > shell - tolerance)))))
        b_idx = b_idx.astype(int)
        print('Modelling Tensor using ' + str(b_idx.shape) + ' volumes:')
        print(b_idx)
        print('Bvals:')
        print(bvals[b_idx])

        bvals = bvals[b_idx]
        bvecs = bvecs[b_idx]
        data = data[:, :, :, b_idx]
        for i in range(0, data.shape[3]):
            data[mask_data == 0, i] = 0

        gtab = gradient_table(bvals, bvecs)
        fa_fname = tensorDir + \
            fbasename.replace(fsl_ext(), suffix + '_FA' + fsl_ext())

        t = time()
        if free_water:
            print("Fitting Free Water Tensor")
            tenmodel = fwdti.FreeWaterTensorModel(gtab)
        elif ols:
            tenmodel = dti.TensorModel(gtab, fit_method='LS')
        elif nlls:
            print("Fitting Tensor using NLLS and SIGMA")
            tenmodel = dti.TensorModel(gtab, fit_method='NLLS')
        else:
            print("Fitting Tensor")
            tenmodel = dti.TensorModel(gtab)

        tenfit = tenmodel.fit(data)
        print("Fitting Complete Total time: ", time() - t)

        if not os.path.exists(tensorDir):
            subprocess.run(['mkdir', tensorDir],
                           stdout=subprocess.PIPE, universal_newlines=True)

        outbase = tensorDir + fbasename

        FA = fractional_anisotropy(tenfit.evals)
        FA[np.isnan(FA)] = 0

        fa_img = nib.Nifti1Image(FA.astype(np.float32), img.affine)
        nib.save(fa_img, fa_fname)

        evecs_img = nib.Nifti1Image(
            tenfit.evecs.astype(np.float32), img.affine)
        nib.save(evecs_img, evec_fname)

        evals_img = nib.Nifti1Image(tenfit.evals.astype(np.float32), img.affine)
        nib.save(evals_img, eval_fname)

        lt_tensor = tenfit.lower_triangular()
        tensor_img = nib.Nifti1Image(lt_tensor*1000, img.affine)
        nib.save(tensor_img, tensor_fname)

        dir_img = nib.Nifti1Image(np.squeeze(
            tenfit.directions.astype(np.float32)), img.affine)
        nib.save(dir_img, pevec_fname)

        MD = dti.mean_diffusivity(tenfit.evals)
        nib.save(nib.Nifti1Image(MD.astype(np.float32), img.affine), md_fname)

        RD = tenfit.rd
        nib.save(nib.Nifti1Image(RD.astype(np.float32), img.affine), rd_fname)

        AD = tenfit.ad
        nib.save(nib.Nifti1Image(AD.astype(np.float32), img.affine), ad_fname)

        FA = np.clip(FA, 0, 1)
        RGB = color_fa(FA, tenfit.evecs)
        nib.save(nib.Nifti1Image(
            np.array(255 * RGB, 'uint8'), img.affine), fa_rgb_fname)

        if free_water:
            FW = tenfit.f
            nib.save(nib.Nifti1Image(FW.astype(np.float32), img.affine),
                     outbase.replace(fsl_ext(), suffix + '_FW' + fsl_ext()))
    else:
        print("DTI Tensor Files Already Exist: Skipping")

    return fa_fname, md_fname, fa_rgb_fname, ad_fname, rd_fname, evec_fname, pevec_fname, tensor_fname


def output_DKI_maps(fname, fmask, bvals, bvecs, dkiDir):
    """
    Generates DKI maps from a given diffusion MRI image

    Parameters
    ----------
    fname: string
        The path to the diffusion MRI image
    fmask: string
        The path to the mask image
    bvals: array
        The b-values for the diffusion MRI image
    bvecs: array
        The b-vectors for the diffusion MRI image
    dkiDir: string
        The path to the output directory

    Returns
    -------
    none
    """

    print("Starting DKI Map Generation")
    prefix = 'dki'
    fbasename = os.path.basename(fname)
    fa_fname = dkiDir + \
        fbasename.replace(fsl_ext(), '_' + prefix + '_FA' + fsl_ext())
    if not os.path.exists(fa_fname):

        img = nib.load(fname)
        data = img.get_fdata(dtype=np.float32)

        mask_img = nib.load(fmask)
        mask_data = mask_img.get_fdata()

        for i in range(0, data.shape[3]):
            data[mask_data == 0, i] = 0

        gtab = gradient_table(bvals, bvecs)

        print("Fitting DKI")
        t = time()
        dkimodel = dki.DiffusionKurtosisModel(gtab)
        dkifit = dkimodel.fit(data)
        print("Fitting Complete Total time:", time() - t)

        if not os.path.exists(dkiDir):
            subprocess.run(['mkdir', dkiDir],
                           stdout=subprocess.PIPE, universal_newlines=True)

        outbase = dkiDir + fbasename

        FA = dkifit.fa
        FA[np.isnan(FA)] = 0

        fa_img = nib.Nifti1Image(FA.astype(np.float32), img.affine)
        nib.save(fa_img, outbase.replace(
            fsl_ext(), '_' + prefix + '_FA' + fsl_ext()))

        MD = dkifit.md
        nib.save(nib.Nifti1Image(MD.astype(np.float32), img.affine),
                 outbase.replace(fsl_ext(), '_' + prefix + '_MD' + fsl_ext()))

        RD = dkifit.rd
        nib.save(nib.Nifti1Image(RD.astype(np.float32), img.affine),
                 outbase.replace(fsl_ext(), '_' + prefix + '_RD' + fsl_ext()))

        AD = dkifit.ad
        nib.save(nib.Nifti1Image(AD.astype(np.float32), img.affine),
                 outbase.replace(fsl_ext(), '_' + prefix + '_AD' + fsl_ext()))

        FA = np.clip(FA, 0, 1)
        RGB = color_fa(FA, dkifit.evecs)
        nib.save(nib.Nifti1Image(np.array(255 * RGB, 'uint8'), img.affine),
                 outbase.replace(fsl_ext(), '_' + prefix + '_FA_RGB' + fsl_ext()))

        MK = dkifit.mk(0, 3)
        nib.save(nib.Nifti1Image(MK.astype(np.float32), img.affine),
                 outbase.replace(fsl_ext(), '_' + prefix + '_MK' + fsl_ext()))

        RK = dkifit.rk(0, 3)
        nib.save(nib.Nifti1Image(RK.astype(np.float32), img.affine),
                 outbase.replace(fsl_ext(), '_' + prefix + '_RK' + fsl_ext()))

        AK = dkifit.ak(0, 3)
        nib.save(nib.Nifti1Image(AK.astype(np.float32), img.affine),
                 outbase.replace(fsl_ext(), '_' + prefix + '_AK' + fsl_ext()))

        evecs_img = nib.Nifti1Image(
            dkifit.evecs.astype(np.float32), img.affine)
        nib.save(evecs_img, outbase.replace(
            fsl_ext(), '_' + prefix + '_EVECS' + fsl_ext()))

        lt_tensor = dkifit.lower_triangular()
        tensor_img = nib.Nifti1Image(lt_tensor*1000, img.affine)
        nib.save(tensor_img, outbase.replace(
            fsl_ext(), '_' + prefix + '_tensor' + fsl_ext()))

        dir_img = nib.Nifti1Image(np.squeeze(
            dkifit.directions.astype(np.float32)), img.affine)
        nib.save(dir_img, outbase.replace(fsl_ext(), '_' +
                 prefix + '_Primary_Direction' + fsl_ext()))
    else:
        print("DKI Files Already Exist: Skipping")

    return
