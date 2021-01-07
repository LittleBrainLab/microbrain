#!/usr/local/bin/python
import os
import glob
from os import system
from os import path
from time import time
import subprocess
from shutil import which

import numpy as np

import nibabel as nib

from dipy.denoise.localpca import mppca
from dipy.denoise.noise_estimate import estimate_sigma
from dipy.denoise.nlmeans import nlmeans

from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table


# Check to see if program is installed to path and executable before running subprocess
def is_tool(name):
    return which(name) is not None

def fsl_ext():
    fsl_extension = ''
    if os.environ['FSLOUTPUTTYPE'] == 'NIFTI':
        fsl_extension = '.nii'
    elif os.environ['FSLOUTPUTTYPE'] == 'NIFTI_GZ':
        fsl_extension = '.nii.gz'
    return fsl_extension

def dcm2nii(dataDir, outDir, rawDir, thisSub):
    print(os.environ['FSLOUTPUTTYPE'])
    subDir = outDir + '/' + thisSub + '/'
    if not os.path.exists(subDir):
        subprocess.run(['mkdir', subDir], stdout=subprocess.PIPE, universal_newlines=True)
    
    rawDir = subDir + rawDir + '/'
    if not os.path.exists(rawDir):
        subprocess.run(['mkdir', rawDir], stdout=subprocess.PIPE, universal_newlines=True)

    niiOutFile = rawDir + thisSub + fsl_ext() 
    niiOutJson = rawDir + thisSub + '.json'
    bvalOut = rawDir + thisSub + '.bval'
    bvecOut = rawDir + thisSub + '.bvec'
    if not os.path.exists(rawDir + thisSub + fsl_ext()):
        print("Converting Subject: " + thisSub)
        
        if os.path.exists(dataDir):
            if is_tool('dcm2niix'):
                dcm2nii_cmd = ['dcm2niix', 
                                '-f',thisSub,
                                '-o',rawDir]
                if fsl_ext() == '.nii.gz':
                    dcm2nii_cmd.append('-z')
                    dcm2nii_cmd.append('y')

                dcm2nii_cmd.append(dataDir + '/')
                process = subprocess.run(dcm2nii_cmd,
                            stdout=subprocess.PIPE, 
                            universal_newlines=True)

                stdout = process.stdout
                return_code = process.returncode
            else:
                print("DSurfer Preproc: Could not find dcm2niix, make sure it is installed to your path")
                stdout = ''
                return_code = 1
        else:
            print("DSurfer Preproc: Could not find srcDir: " + dataDir)
            stdout = ''
            return_code = 1
    else:
        print("DSurfer Preproc: Dicom already converted skipping subject: " + thisSub)
        stdout = ''
        return_code = 0

    return niiOutFile, bvalOut, bvecOut, niiOutJson, stdout, return_code 


def gibbsRingingCorrection(fdwi, gibbsDir):
    
    # Gibbs ringing correction using c-code (note the 3rdParty Directory needs to be in this folder)
    print("Performing Gibbs Ringing Correction (reisert c-code)")
    t = time()
    if not os.path.exists(gibbsDir):
        subprocess.run(['mkdir', gibbsDir], stdout=subprocess.PIPE, universal_newlines=True)

    fdwi_bname = os.path.basename(fdwi)
    fgibbs = gibbsDir + fdwi_bname.replace(fsl_ext(), '_GR' + fsl_ext())
    
    if not os.path.exists(fgibbs):
        if is_tool('unring.a64'):
                process = subprocess.run(['unring.a64',
                                fdwi,
                                fgibbs],
                            stdout=subprocess.PIPE,
                            universal_newlines=True)
                print("Gibbs Finished, Processing Time: ", time() - t)
                stdout = process.stdout
                return_code = process.returncode
        else:
            print("DSurfer Preproc: Could not find unring.a64, make sure it is installed to your path")
            stdout = ''
            return_code = 1
    else:
         print("Gibbs Already Done: Skipping")
         stdout = ''
         return_code = 0
        
    return fgibbs, stdout, return_code

def denoiseMPPCA(fdwi, fbval, fbvec, denoiseDir, patch_radius =2):

    # denoising using MPPCA DIPY code
    print("Performing Denoising in DIPY")
    t = time()
    if not os.path.exists(denoiseDir):
        subprocess.run(['mkdir', denoiseDir], stdout=subprocess.PIPE, universal_newlines=True)

    fdwi_bname = os.path.basename(fdwi)
    fdenoise = denoiseDir + fdwi_bname.replace(fsl_ext(), '_DN' + fsl_ext())
    fsigma = fdenoise.replace(fsl_ext(), 'sigma' + fsl_ext())
    
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
        subprocess.run(['mkdir', denoiseDir], stdout=subprocess.PIPE, universal_newlines=True)

    fdwi_bname = os.path.basename(fdwi)
    fdenoise = denoiseDir + fdwi_bname.replace(fsl_ext(), '_DNNL' + fsl_ext())
    fdiff = fdenoise.replace(fsl_ext(), 'diff' + fsl_ext()) 
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
    
    fb0avg = fgibbs.replace(fsl_ext(), '_b0avg' + fsl_ext())
    in_file = fb0avg
    out_file = fb0avg.replace(fsl_ext(), '_BET')
    fmask = out_file.replace('BET','BET_mask' + fsl_ext())
    if not os.path.exists(fmask):
        img = nib.load(fgibbs)
        data = img.get_data()
    
        b0_idx =np.squeeze(np.array(np.where(np.logical_and(bvals < tolerance, bvals > -tolerance))))

        b0_data = data[...,b0_idx]
        meanB0 = np.mean(b0_data,axis=3);
        nib.save(nib.Nifti1Image(meanB0, img.affine), fb0avg)
        b0_idx =np.squeeze(np.array(np.where(np.logical_and(bvals < 15, bvals > -15))))
    
        nib.save(nib.Nifti1Image(meanB0, img.affine), out_file + fsl_ext())
        nib.save(nib.Nifti1Image(np.ones(meanB0.shape), img.affine), fmask)

    return fmask, fb0avg

def fslv6p0_brain_mask(fgibbs,bvals, tolerance=100):
    print("Performing Brain Mask Extraction (FSLv6.0 bet2)")
    t = time()
    fb0avg = fgibbs.replace(fsl_ext(), '_b0avg' + fsl_ext())
    in_file = fb0avg
    out_file = fb0avg.replace(fsl_ext(), '_BET')
    fmask = out_file.replace('BET','BET_mask' + fsl_ext())
    if not os.path.exists(fmask):
        img = nib.load(fgibbs)
        data = img.get_data()

        b0_idx =np.squeeze(np.array(np.where(np.logical_and(bvals < tolerance, bvals > -tolerance))))

        b0_data = data[...,b0_idx]
        meanB0 = np.mean(b0_data,axis=3);
        nib.save(nib.Nifti1Image(meanB0, img.affine), fb0avg)
        b0_idx =np.squeeze(np.array(np.where(np.logical_and(bvals < 15, bvals > -15))))
        
        if is_tool('bet2'):
                process = subprocess.run(['bet2',
                                fb0avg,
                                out_file,
                                '-m',
                                '-f', '0.15',
                                '-g', '0.1'],
                            stdout=subprocess.PIPE,
                            universal_newlines=True)
                print("BET2 finished masking, Processing Time: ", time() - t)
                stdout = process.stdout
                return_code = process.returncode
        else:
            print("DSurfer Preproc: Could not find bet2, make sure FSLv6.0 or greater is installed to your path")
            stdout = ''
            return_code = 1 
    else:
        print("DSurfer Preproc: BET2 masking already performed skipping")
        stdout = ''
        return_code = 0

    return fmask, fb0avg, stdout, return_code

def fslv6p0_eddy(fdwi, facq, findex, fmask, fbval, fbvec, fjson, cuda, eddyDir, eddyOutputDir):
    print("Performing Eddy Current / Motion Correction (FSL eddy), this may take a while (will implement a progress update in the future)")
    t = time()

    if not os.path.exists(eddyDir):
        subprocess.run(['mkdir', eddyDir], stdout=subprocess.PIPE, universal_newlines=True)

    if cuda:
        eddy_command = 'eddy_cuda9.1'
    else:
        eddy_command = 'eddy'

    fdwi_bname = os.path.basename(fdwi)
    feddy = eddyDir + fdwi_bname.replace(fsl_ext(), '_EDDY' + fsl_ext())
    fbvec_rotated = feddy + '.eddy_rotated_bvecs'
    if not os.path.exists(feddy):
        if is_tool(eddy_command):
            full_eddy_command = [eddy_command,
                                '--imain=' + fdwi,
                                '--acqp=' + facq,
                                '--index=' + findex,
                                '--mask=' + fmask,
                                '--bvals=' + fbval,
                                '--bvecs=' + fbvec,
                                '--out=' + feddy,
                                '--data_is_shelled']
            if fjson:
                full_eddy_command.append('--json='+fjson)
          
            process = subprocess.run(full_eddy_command,
                            stdout=subprocess.PIPE,
                            universal_newlines=True)
            stdout = process.stdout
            return_code = process.returncode

            # Move generated output files to another directory
            if not os.path.exists(eddyOutputDir):
                subprocess.run(['mkdir', eddyOutputDir], stdout=subprocess.PIPE, universal_newlines=True)
            
            subprocess.run(['mv', feddy + '.eddy_command_txt', eddyOutputDir], stdout=subprocess.PIPE, universal_newlines=True)
            subprocess.run(['mv', feddy + '.eddy_movement_rms', eddyOutputDir], stdout=subprocess.PIPE, universal_newlines=True)
            subprocess.run(['mv', feddy + '.eddy_outlier_map', eddyOutputDir], stdout=subprocess.PIPE, universal_newlines=True)
            subprocess.run(['mv', feddy + '.eddy_outlier_n_sqr_stdev_map', eddyOutputDir], stdout=subprocess.PIPE, universal_newlines=True)
            subprocess.run(['mv', feddy + '.eddy_outlier_n_stdev_map', eddyOutputDir], stdout=subprocess.PIPE, universal_newlines=True)
            subprocess.run(['mv', feddy + '.eddy_outlier_report', eddyOutputDir], stdout=subprocess.PIPE, universal_newlines=True)
            subprocess.run(['mv', feddy + '.eddy_parameters', eddyOutputDir], stdout=subprocess.PIPE, universal_newlines=True)
            subprocess.run(['mv', feddy + '.eddy_post_eddy_shell_alignment_parameters', eddyOutputDir], stdout=subprocess.PIPE, universal_newlines=True)
            subprocess.run(['mv', feddy + '.eddy_post_eddy_shell_PE_translation_parameters', eddyOutputDir], stdout=subprocess.PIPE, universal_newlines=True)
            subprocess.run(['mv', feddy + '.eddy_restricted_movement_rms', eddyOutputDir], stdout=subprocess.PIPE, universal_newlines=True)
            subprocess.run(['mv', feddy + '.eddy_values_of_all_input_parameters', eddyOutputDir], stdout=subprocess.PIPE, universal_newlines=True)

            print("Eddy Current/Motion Finished, Processing Time: ", time() - t)
        else:
            print("DSurfer Preproc: Could not find " + eddy_command + ", make sure FSLv6.0 or greater is installed to your path")
            stdout = ''
            return_code = 1
    else:
        print("Eddy Correction Already Done: Skipping")
        stdout = ''
        return_code = 0

    return feddy, fbvec_rotated, stdout, return_code

def n4correct_by_b0(fb0avg, fmask, fgibbs, preprocDWIDir):
    print("N4 correction (dHCP toolbox)")
    t = time()

    fn4correct = fb0avg.replace(fsl_ext(),'_n4' + fsl_ext())
    fbias = fn4correct.replace(fsl_ext(),'_bias' + fsl_ext())

    if not os.path.exists(fn4correct):
        if is_tool('N4'):
                process = subprocess.run(['N4',
                                '-i', fb0avg,
                                '-x',fmask,
                                '-o', '[' + fn4correct + ',' + fbias + ']'],
                            stdout=subprocess.PIPE,
                            universal_newlines=True)
                print("N4 finished, Processing Time: ", time() - t)
                stdout = process.stdout
                return_code = process.returncode
        else:
            print("DSurfer Preproc: Could not find N4, make sure it is installed to your path")
            stdout = ''
            return_code = 1
    else:
        print("N4 Correction previously done: Skipping")
        stdout = ''
        return_code = 0
    
    biasField = nib.load(fbias).get_data()
    fgibbs_basename = os.path.basename(fgibbs) 
    fn4dwi = preprocDWIDir + fgibbs_basename.replace(fsl_ext(),'_N4' + fsl_ext())
    if not os.path.exists(fn4dwi):
        gibbs_img = nib.load(fgibbs)
        gibbs_data = gibbs_img.get_data()

        n4dwi_data = np.zeros(gibbs_data.shape)

        for vol_idx in range(0, n4dwi_data.shape[3]):
            thisVol =  np.squeeze(gibbs_data[:,:,:,vol_idx])

            n4dwi_data[:,:,:,vol_idx] = thisVol / biasField
        nib.save(nib.Nifti1Image(n4dwi_data, gibbs_img.affine), fn4dwi)
    
    return fn4dwi, stdout, return_code

def output_DWI_maps(fb0avg, fmask, feddy, bvals, shells, meanDWIDir, preproc_suffix, dwi_shell = 1000, tolerance = 100):
    
    shell_tolerance = 50 # multiband doesn't prescribe the exact b-value
    print("N4 correction (dHCP toolbox)")
    t = time()
    
    if not os.path.exists(meanDWIDir):
        subprocess.run(['mkdir', meanDWIDir], stdout=subprocess.PIPE, universal_newlines=True)

    fn4correct = fb0avg.replace(fsl_ext(),'_n4' + fsl_ext())
    fbias = fn4correct.replace(fsl_ext(),'_bias' + fsl_ext())

    if not os.path.exists(fn4correct):
        if is_tool('N4'):
                process = subprocess.run(['N4',
                                '-i', fb0avg,
                                '-x',fmask,
                                '-o', '[' + fn4correct + ',' + fbias + ']'],
                            stdout=subprocess.PIPE,
                            universal_newlines=True)
                print("N4 finished, Processing Time: ", time() - t)
                stdout = process.stdout
                return_code = process.returncode
        else:
            print("DSurfer Preproc: Could not find N4, make sure it is installed to your path")
            stdout = ''
            return_code = 1
    else:
        print("N4 Correction previously done: Skipping")
        stdout = ''
        return_code = 0

    biasField = nib.load(fbias).get_data()
    feddy_basename = os.path.basename(feddy) 
    fmeanshell = meanDWIDir + feddy_basename.replace(fsl_ext(),'mean_b' + str(shells[0]) + fsl_ext())
    if not os.path.exists(fmeanshell):
        eddy_img = nib.load(feddy)
        eddy_data = eddy_img.get_data()

    for shell in shells:
        fmeanshell = meanDWIDir + feddy_basename.replace(fsl_ext(),'_mean_b' + str(shell) + fsl_ext())
        fmeanshell_n4 = meanDWIDir + feddy_basename.replace(fsl_ext(),'_mean_b' + str(shell) + '_n4' + fsl_ext())
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
    
    return fb0, fb0_n4, fdwi, fdwi_n4, stdout, return_code

def output_DWI_maps_noN4(fb0avg, fmask, feddy, bvals, shells, meanDWIDir, preproc_suffix, dwi_shell = 1000, tolerance = 100):

    shell_tolerance = 50 # multiband doesn't prescribe the exact b-value
    print("No N4 correction (dHCP toolbox)")
    t = time()

    if not os.path.exists(meanDWIDir):
        subprocess.run(['mkdir', meanDWIDir], stdout=subprocess.PIPE, universal_newlines=True)

    fn4correct = fb0avg.replace(fsl_ext(),'_n4' + fsl_ext())

    feddy_basename = os.path.basename(feddy)
    fmeanshell = meanDWIDir + feddy_basename.replace(fsl_ext(),'mean_b' + str(shells[0]) + fsl_ext())
    if not os.path.exists(fmeanshell):
        eddy_img = nib.load(feddy)
        eddy_data = eddy_img.get_data()

    for shell_idx in range(0,len(shells)):
        shell = shells[shell_idx]
        fmeanshell = meanDWIDir + feddy_basename.replace(fsl_ext(),'_mean_b' + str(shell) + fsl_ext())
        fmeanshell_n4 = meanDWIDir + feddy_basename.replace(fsl_ext(),'_mean_b' + str(shell) + '_n4' + fsl_ext())
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

