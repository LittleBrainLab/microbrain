#!/usr/local/bin/python
import os
import glob
from os import system
from os import path
from time import time

import numpy as np

import nibabel as nib

from dipy.denoise.localpca import mppca
from dipy.denoise.noise_estimate import estimate_sigma
from dipy.denoise.nlmeans import nlmeans

from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table

def dcm2nii(dataDir, outDir, rawDir, thisSub):
    subDir = outDir + '/' + thisSub + '/'
    if not os.path.exists(subDir):
        os.system('mkdir ' + subDir)
    
    rawDir = subDir + rawDir + '/'
    if not os.path.exists(rawDir):
        os.system('mkdir ' + rawDir)

    niiOutFile = rawDir + thisSub + '.nii'
    niiOutJson = rawDir + thisSub + '.json'
    bvalOut = rawDir + thisSub + '.bval'
    bvecOut = rawDir + thisSub + '.bvec'
    if not os.path.exists(rawDir + thisSub + '.nii'):
        print("Converting Subject: " + thisSub)
        
        if os.path.exists(dataDir):
            os.system('dcm2niix -f ' + thisSub + ' -o ' + rawDir + ' ' + dataDir + '/*.dcm')
        else:
            print("Could not find srcDir: " + dataDir)
    else:
        print("Dicom already converted skipping subject: " + thisSub)
    
    return niiOutFile, bvalOut, bvecOut, niiOutJson 


def gibbsRingingCorrection(fdwi, gibbsDir):
    
    # Gibbs ringing correction using c-code (note the 3rdParty Directory needs to be in this folder)
    print("Performing Gibbs Ringing Correction (reisert c-code)")
    t = time()
    if not os.path.exists(gibbsDir):
        os.system('mkdir ' + gibbsDir)

    fdwi_bname = os.path.basename(fdwi)
    fgibbs = gibbsDir + fdwi_bname.replace('.nii', '_GR.nii')
    
    if not os.path.exists(fgibbs):
         unring_cmd = 'unring.a64 ' + fdwi + ' ' + fgibbs
         system(unring_cmd)
         print("Gibbs Finished, Processing Time: ", time() - t)
    else:
         print("Gibbs Already Done: Skipping")

    return fgibbs

def denoiseMPPCA(fdwi, fbval, fbvec, denoiseDir, patch_radius =2):

    # denoising using MPPCA DIPY code
    print("Performing Denoising in DIPY")
    t = time()
    if not os.path.exists(denoiseDir):
        os.system('mkdir ' + denoiseDir)

    fdwi_bname = os.path.basename(fdwi)
    fdenoise = denoiseDir + fdwi_bname.replace('.nii', '_DN.nii')
    fsigma = fdenoise.replace('.nii', 'sigma.nii')
    
    if not os.path.exists(fdenoise):
        img = nib.load(fdwi)
        data = img.get_data()

        bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
        gtab = gradient_table(bvals, bvecs)

        denoised_data, sigma = mppca(data, patch_radius=patch_radius, return_sigma=True)
        nib.save(nib.Nifti1Image(denoised_data, img.affine), fdenoise)
        nib.save(nib.Nifti1Image(sigma, img.affine), fsigma)

        print("Denoising Finished, Processing Time: ", time() - t)
    else:
        print("Denoising Already Done: Skipping")

    return fdenoise

def denoiseNONLOCAL(fdwi, fmask, denoiseDir):

    # denoising using MPPCA DIPY code
    print("Performing Denoising NONLOCAL in DIPY")
    t = time()
    if not os.path.exists(denoiseDir):
        os.system('mkdir ' + denoiseDir)

    fdwi_bname = os.path.basename(fdwi)
    fdenoise = denoiseDir + fdwi_bname.replace('.nii', '_DNNL.nii')
    fdiff = fdenoise.replace('.nii', 'diff.nii') 
    if not os.path.exists(fdenoise):
        img = nib.load(fdwi)
        data = img.get_data()
        
        mask_img = nib.load(fmask)
        mask_data = mask_img.get_data()

        sigma = estimate_sigma(data, N=64)
        denoised_data = nlmeans(data, sigma=sigma, mask=mask_data, patch_radius=1,
              block_radius=1, rician=True)
        nib.save(nib.Nifti1Image(denoised_data, img.affine), fdenoise)
        denoised_data[mask_data==0] = data[mask_data == 0]
        nib.save(nib.Nifti1Image(data - denoised_data, img.affine), fdiff)
        print("Denoising Finished, Processing Time: ", time() - t)
    else:
        print("Denoising Already Done: Skipping")

    return fdenoise

def fslv6p0_fake_brain_mask(fgibbs,bvals, tolerance=100):
    print("Performing Fake Brain Mask Extraction (FSLv6.0 bet2)")
    
    fb0avg = fgibbs.replace('.nii', '_b0avg.nii')
    in_file = fb0avg
    out_file = fb0avg.replace('.nii', '_BET')
    fmask = out_file.replace('BET','BET_mask.nii')
    if not os.path.exists(fmask):
        img = nib.load(fgibbs)
        data = img.get_data()
    
        b0_idx =np.squeeze(np.array(np.where(np.logical_and(bvals < tolerance, bvals > -tolerance))))

        b0_data = data[...,b0_idx]
        meanB0 = np.mean(b0_data,axis=3);
        nib.save(nib.Nifti1Image(meanB0, img.affine), fb0avg)
        b0_idx =np.squeeze(np.array(np.where(np.logical_and(bvals < 15, bvals > -15))))
    
        nib.save(nib.Nifti1Image(meanB0, img.affine), out_file+'.nii')
        nib.save(nib.Nifti1Image(np.ones(meanB0.shape), img.affine), fmask)

    return fmask, fb0avg

def fslv6p0_brain_mask(fgibbs,bvals, tolerance=100):
    print("Performing Brain Mask Extraction (FSLv6.0 bet2)")

    fb0avg = fgibbs.replace('.nii', '_b0avg.nii')
    in_file = fb0avg
    out_file = fb0avg.replace('.nii', '_BET')
    fmask = out_file.replace('BET','BET_mask.nii')
    if not os.path.exists(fmask):
        img = nib.load(fgibbs)
        data = img.get_data()

        b0_idx =np.squeeze(np.array(np.where(np.logical_and(bvals < tolerance, bvals > -tolerance))))

        b0_data = data[...,b0_idx]
        meanB0 = np.mean(b0_data,axis=3);
        nib.save(nib.Nifti1Image(meanB0, img.affine), fb0avg)
        b0_idx =np.squeeze(np.array(np.where(np.logical_and(bvals < 15, bvals > -15))))
        
        os.system('bet2 ' + fb0avg + ' ' + out_file + ' -m -f 0.15 -g 0.1')

    return fmask, fb0avg

def fslv6p0_eddy(fgibbs, facq, findex, fmask, fbval, fbvec, fjson, eddyDir, eddyOutputDir):
    print("Performing Eddy Current / Motion Correction (FSL eddy)")
    t = time()

    if not os.path.exists(eddyDir):
        os.system('mkdir ' + eddyDir)

    fgibbs_bname = os.path.basename(fgibbs)
    feddy = eddyDir + fgibbs_bname.replace('.nii', '_EDDY.nii')
    fbvec_rotated = feddy + '.eddy_rotated_bvecs'
    if not os.path.exists(feddy):
        if fjson:
            os.system('eddy_cuda9.1 --imain=' + fgibbs + 
                ' --acqp=' + facq + 
                ' --index=' + findex + 
                ' --mask=' + fmask + 
                ' --bvals=' + fbval + 
                ' --bvecs=' + fbvec + 
                ' --out=' + feddy + 
                ' --json=' + fjson + 
                ' --data_is_shelled')
        else:
            os.system('eddy_cuda9.1 --imain=' + fgibbs +
                ' --acqp=' + facq +
                ' --index=' + findex +
                ' --mask=' + fmask +
                ' --bvals=' + fbval +
                ' --bvecs=' + fbvec +
                ' --out=' + feddy +
                ' --data_is_shelled')

        # Move generated output files to another directory
        if not os.path.exists(eddyOutputDir):
            os.system('mkdir ' + eddyOutputDir)

        os.system('mv ' + feddy + '.eddy_command_txt ' + eddyOutputDir)
        os.system('mv ' + feddy + '.eddy_movement_rms ' + eddyOutputDir)
        os.system('mv ' + feddy + '.eddy_outlier_map ' + eddyOutputDir)
        os.system('mv ' + feddy + '.eddy_outlier_n_sqr_stdev_map ' + eddyOutputDir)
        os.system('mv ' + feddy + '.eddy_outlier_n_stdev_map ' + eddyOutputDir)
        os.system('mv ' + feddy + '.eddy_outlier_report ' + eddyOutputDir)
        os.system('mv ' + feddy + '.eddy_parameters ' + eddyOutputDir)
        os.system('mv ' + feddy + '.eddy_post_eddy_shell_alignment_parameters '  + eddyOutputDir)
        os.system('mv ' + feddy + '.eddy_post_eddy_shell_PE_translation_parameters '  + eddyOutputDir)
        os.system('mv ' + feddy + '.eddy_restricted_movement_rms '  + eddyOutputDir)
        os.system('mv ' + feddy + '.eddy_values_of_all_input_parameters '  + eddyOutputDir)

        print("Eddy Current/Motion Finished, Processing Time: ", time() - t)
    else:
        print("Eddy Correction Already Done: Skipping")
    
    return feddy, fbvec_rotated

def n4correct_by_b0(fb0avg, fmask, fgibbs, preprocDWIDir):
    print("N4 correction (dHCP toolbox)")
    t = time()

    fn4correct = fb0avg.replace('.nii','_n4.nii')
    fbias = fn4correct.replace('.nii','_bias.nii')

    if not os.path.exists(fn4correct):
        os.system('N4 -i ' + fb0avg + ' -x ' + fmask + ' -o [' + fn4correct + ',' + fbias + ']')
    else:
        print("N4 Correction previously done: Skipping")

    biasField = nib.load(fbias).get_data()
    fgibbs_basename = os.path.basename(fgibbs) 
    fn4dwi = preprocDWIDir + fgibbs_basename.replace('.nii','_N4.nii')
    if not os.path.exists(fn4dwi):
        gibbs_img = nib.load(fgibbs)
        gibbs_data = gibbs_img.get_data()

        n4dwi_data = np.zeros(gibbs_data.shape)

        for vol_idx in range(0, n4dwi_data.shape[3]):
            thisVol =  np.squeeze(gibbs_data[:,:,:,vol_idx])

            n4dwi_data[:,:,:,vol_idx] = thisVol / biasField
        nib.save(nib.Nifti1Image(n4dwi_data, gibbs_img.affine), fn4dwi)
    
    return fn4dwi

def output_DWI_maps(fb0avg, fmask, feddy, bvals, shells, meanDWIDir, preproc_suffix, dwi_shell = 1000, tolerance = 100):
    
    shell_tolerance = 50 # multiband doesn't prescribe the exact b-value
    print("N4 correction (dHCP toolbox)")
    t = time()
    
    if not os.path.exists(meanDWIDir):
        os.system('mkdir ' + meanDWIDir)

    fn4correct = fb0avg.replace('.nii','_n4.nii')
    fbias = fn4correct.replace('.nii','_bias.nii')

    if not os.path.exists(fn4correct):
        os.system('N4 -i ' + fb0avg + ' -x ' + fmask + ' -o [' + fn4correct + ',' + fbias + ']')
    else:
        print("N4 Correction previously done: Skipping")

    biasField = nib.load(fbias).get_data()
    feddy_basename = os.path.basename(feddy) 
    fmeanshell = meanDWIDir + feddy_basename.replace('.nii','mean_b' + str(shells[0]) + '.nii')
    if not os.path.exists(fmeanshell):
        eddy_img = nib.load(feddy)
        eddy_data = eddy_img.get_data()

    for shell in shells:
        fmeanshell = meanDWIDir + feddy_basename.replace('.nii','_mean_b' + str(shell) + '.nii')
        fmeanshell_n4 = meanDWIDir + feddy_basename.replace('.nii','_mean_b' + str(shell) + '_n4.nii')
        if shell == 0:
            fb0 = fmeanshell
            fb0_n4 = fmeanshell_n4

        if shell == dwi_shell:
            fdwi = fmeanshell
            fdwi_n4 = fmeanshell_n4

        if not os.path.exists(fmeanshell):
            bshell_idx = np.squeeze(np.array(np.where(np.logical_and(bvals < shell + tolerance, bvals > shell - tolerance))))
            bshell_data = eddy_data[...,bshell_idx]
        
            meanBSHELL = np.mean(bshell_data,axis=3)
            nib.save(nib.Nifti1Image(meanBSHELL, eddy_img.affine), fmeanshell)

            meanBSHELL_n4 = np.mean(bshell_data,axis=3) / biasField
            nib.save(nib.Nifti1Image(meanBSHELL_n4, eddy_img.affine), fmeanshell_n4)
    
    return fb0, fb0_n4, fdwi, fdwi_n4

def output_DWI_maps_noN4(fb0avg, fmask, feddy, bvals, shells, meanDWIDir, preproc_suffix, dwi_shell = 1000, tolerance = 100):

    shell_tolerance = 50 # multiband doesn't prescribe the exact b-value
    print("No N4 correction (dHCP toolbox)")
    t = time()

    if not os.path.exists(meanDWIDir):
        os.system('mkdir ' + meanDWIDir)

    fn4correct = fb0avg.replace('.nii','_n4.nii')

    feddy_basename = os.path.basename(feddy)
    fmeanshell = meanDWIDir + feddy_basename.replace('.nii','mean_b' + str(shells[0]) + '.nii')
    if not os.path.exists(fmeanshell):
        eddy_img = nib.load(feddy)
        eddy_data = eddy_img.get_data()

    for shell_idx in range(0,len(shells)):
        shell = shells[shell_idx]
        fmeanshell = meanDWIDir + feddy_basename.replace('.nii','_mean_b' + str(shell) + '.nii')
        fmeanshell_n4 = meanDWIDir + feddy_basename.replace('.nii','_mean_b' + str(shell) + '_n4.nii')
        if shell_idx == 0:
            fbase = fmeanshell
            fbase_n4 = fmeanshell_n4

        if shell_idx == 1:
            fdwi = fmeanshell
            fdwi_n4 = fmeanshell_n4

        if not os.path.exists(fmeanshell):
            bshell_idx = np.squeeze(np.array(np.where(np.logical_and(bvals < shell + tolerance, bvals > shell - tolerance))))
            bshell_data = eddy_data[...,bshell_idx]

            meanBSHELL = np.mean(bshell_data,axis=3)
            nib.save(nib.Nifti1Image(meanBSHELL, eddy_img.affine), fmeanshell)

            meanBSHELL_n4 = meanBSHELL 
            nib.save(nib.Nifti1Image(meanBSHELL_n4, eddy_img.affine), fmeanshell_n4)

    return fbase, fbase_n4, fdwi, fdwi_n4

