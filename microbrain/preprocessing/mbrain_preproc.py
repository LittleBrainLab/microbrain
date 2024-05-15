#!/usr/local/bin/python
import os
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

import microbrain.utils.surf_util as sutil


def is_tool(name):
    return which(name) is not None


def fsl_ext():
    fsl_extension = ''
    if os.environ['FSLOUTPUTTYPE'] == 'NIFTI':
        fsl_extension = '.nii'
    elif os.environ['FSLOUTPUTTYPE'] == 'NIFTI_GZ':
        fsl_extension = '.nii.gz'
    return fsl_extension


def gibbsRingingCorrection(fdwi, gibbsDir):

    # Gibbs ringing correction using c-code (note the 3rdParty Directory needs to be in this folder)
    print("Performing Gibbs Ringing Correction (reisert c-code)")
    t = time()
    if not os.path.exists(gibbsDir):
        subprocess.run(['mkdir', gibbsDir],
                       stdout=subprocess.PIPE, universal_newlines=True)

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
            print(
                "microbrain Preproc: Could not find unring.a64, make sure it is installed to your path")
            stdout = ''
            return_code = 1
    else:
        print("Gibbs Already Done: Skipping")
        stdout = ''
        return_code = 0

    return fgibbs, stdout, return_code


def denoiseNLSAM(fdwi, fmask, fbval, fbvec, denoiseDir, patch_radius=2, cpu_num=0):

    # denoising using NLSAM
    print("Performing Denoising NLSAM")
    t = time()
    if not os.path.exists(denoiseDir):
        subprocess.run(['mkdir', denoiseDir],
                       stdout=subprocess.PIPE, universal_newlines=True)

    fdwi_bname = os.path.basename(fdwi)
    fdenoise = denoiseDir + \
        fdwi_bname.replace(fsl_ext(), '_DNLSAM' + fsl_ext())

    # Noise estimate files
    fsigma = denoiseDir + fdwi_bname.replace(fsl_ext(), '_sigma' + fsl_ext())
    fncoil = denoiseDir + fdwi_bname.replace(fsl_ext(), '_Ncoil' + fsl_ext())
    fstab = denoiseDir + fdwi_bname.replace(fsl_ext(), '_STAB' + fsl_ext())

    if not os.path.exists(fdenoise):
        if is_tool('nlsam_denoising'):
            full_dnlsam_command = ['nlsam_denoising',
                                   fdwi,
                                   fdenoise,
                                   fbval,
                                   fbvec,
                                   '-m', fmask,
                                   '--save_stab', fstab,
                                   '--save_sigma', fsigma,
                                   '--save_N', fncoil,
                                   '-v',
                                   '-f']
            if cpu_num > 0:
                full_dnlsam_command.append('--cores')
                full_dnlsam_command.append(str(cpu_num))

            process = subprocess.run(full_dnlsam_command,
                                     stdout=subprocess.PIPE,
                                     universal_newlines=True)
            print("NLSAM finished denoising, Processing Time: ", time() - t)
            stdout = process.stdout
            return_code = process.returncode
        else:
            print(
                "microbrain preproc: Could not find nlsam_denoising, make sure nlsam is installed to your path")
            stdout = ''
            return_code = 1
    else:
        print("Denoising Already Done: Skipping")

    return fdenoise


def stabilizeNLSAM(fdwi, fmask, fbval, fbvec, denoiseDir, patch_radius=2):

    # stabilizing signal using NLSAM
    print("Performing Signal Stabilization NLSAM")
    t = time()
    if not os.path.exists(denoiseDir):
        subprocess.run(['mkdir', denoiseDir],
                       stdout=subprocess.PIPE, universal_newlines=True)

    fdwi_bname = os.path.basename(fdwi)
    fdenoise = denoiseDir + \
        fdwi_bname.replace(fsl_ext(), '_NOTEXIST' + fsl_ext())
    fsigma = denoiseDir + fdwi_bname.replace(fsl_ext(), '_sigma' + fsl_ext())
    fncoil = denoiseDir + fdwi_bname.replace(fsl_ext(), '_Ncoil' + fsl_ext())
    fstab = denoiseDir + fdwi_bname.replace(fsl_ext(), '_STAB' + fsl_ext())

    if not os.path.exists(fstab):
        if is_tool('nlsam_denoising'):
            process = subprocess.run(['nlsam_denoising',
                                      fdwi,
                                      fdenoise,
                                      fbval,
                                      fbvec,
                                      '-m', fmask,
                                      '--save_stab', fstab,
                                      '--save_sigma', fsigma,
                                      '--save_N', fncoil,
                                      '--no_denoising',
                                      '-v',
                                      '-f'],
                                     stdout=subprocess.PIPE,
                                     universal_newlines=True)
            print("NLSAM finished stabilizing, Processing Time: ", time() - t)
            stdout = process.stdout
            return_code = process.returncode
        else:
            print(
                "microbrain preproc: Could not find nlsam_denoising, make sure nlsam is installed to your path")
            stdout = ''
            return_code = 1
    else:
        print("Stabilization Already Done: Skipping")

    return fstab


def denoiseMPPCA(fdwi, fbval, fbvec, denoiseDir, patch_radius=2):

    # denoising using MPPCA DIPY code
    print("Performing Denoising in DIPY")
    t = time()
    if not os.path.exists(denoiseDir):
        subprocess.run(['mkdir', denoiseDir],
                       stdout=subprocess.PIPE, universal_newlines=True)

    fdwi_bname = os.path.basename(fdwi)
    fdenoise = denoiseDir + \
        fdwi_bname.replace(fsl_ext(), '_DMPPCA' + fsl_ext())
    fsigma = fdenoise.replace(fsl_ext(), 'sigma' + fsl_ext())

    if not os.path.exists(fdenoise):
        img = nib.load(fdwi)
        data = img.get_fdata()

        bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
        gtab = gradient_table(bvals, bvecs)

        denoised_data, sigma = mppca(
            data, patch_radius=patch_radius, return_sigma=True)
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
        subprocess.run(['mkdir', denoiseDir],
                       stdout=subprocess.PIPE, universal_newlines=True)

    fdwi_bname = os.path.basename(fdwi)
    fdenoise = denoiseDir + fdwi_bname.replace(fsl_ext(), '_DNNL' + fsl_ext())
    fdiff = fdenoise.replace(fsl_ext(), 'diff' + fsl_ext())
    if not os.path.exists(fdenoise):
        img = nib.load(fdwi)
        data = img.get_fdata()

        mask_img = nib.load(fmask)
        mask_data = mask_img.get_fdata()

        sigma = estimate_sigma(data, N=64)
        denoised_data = nlmeans(data, sigma=sigma, mask=mask_data, patch_radius=1,
                                block_radius=1, rician=True)
        nib.save(nib.Nifti1Image(denoised_data, img.affine), fdenoise)
        denoised_data[mask_data == 0] = data[mask_data == 0]
        nib.save(nib.Nifti1Image(data - denoised_data, img.affine), fdiff)
        print("Denoising Finished, Processing Time: ", time() - t)
    else:
        print("Denoising Already Done: Skipping")

    return fdenoise


def surface_based_brain_mask_onMD(finmask, fmd, maskDir):

    if not os.path.exists(maskDir):
        subprocess.run(['mkdir', maskDir],
                       stdout=subprocess.PIPE, universal_newlines=True)

    fmd_bname = os.path.basename(fmd)

    # Invert mask
    finmask_invert = finmask.replace(fsl_ext(), '_inv' + fsl_ext())
    if not os.path.exists(finmask_invert):
        os.system('fslmaths ' + finmask + ' -mul -1 -add 1 ' + finmask_invert)

    # Make iso surface from mask
    fmask_insurf = maskDir + \
        fmd_bname.replace('MD' + fsl_ext(), 'init_mask.vtk')
    if not os.path.exists(fmask_insurf):
        os.system('mirtk extract-surface ' + finmask_invert +
                  ' ' + fmask_insurf + ' -isovalue 0.5 -close')

    # Deform mesh around MD to remove a bunch of non brain voxels
    fmask_outsurf = maskDir + \
        fmd_bname.replace('MD' + fsl_ext(), 'deform_mask.vtk')
    if not os.path.exists(fmask_outsurf):
        os.system('mirtk deform-mesh ' + fmask_insurf + ' ' + fmask_outsurf + ' -image ' + fmd + ' -edge-distance 1.0 -edge-distance-smoothing 1 -edge-distance-median 1 -edge-distance-averaging 8 4 2 1 -optimizer EulerMethod -step 0.2 -steps 100 -epsilon 1e-6 -delta 0.001 -min-active 1% -reset-status -nointersection -fast-collision-test -min-width 0.1 -min-distance 0.1 -repulsion 4 -repulsion-distance 1.0 -repulsion-width 2.0 -curvature 4.0 -gauss-curvature 1.0 -gauss-curvature-minimum .1 -gauss-curvature-maximum .2 -gauss-curvature-outside 0.5 -edge-distance-type StrongestMinimum -threads 32 -remesh 1 -min-edge-length 1.0 -max-edge-length 2.0 -edge-distance-min-intensity 0.0003 -normal -0.5')

    fmask_outsurf2 = maskDir + \
        fmd_bname.replace('MD' + fsl_ext(), 'deform_mask2.vtk')
    if not os.path.exists(fmask_outsurf2):
        os.system('mirtk deform-mesh ' + fmask_outsurf + ' ' + fmask_outsurf2 + ' -image ' + fmd + ' -edge-distance 1.0 -edge-distance-smoothing 1 -edge-distance-median 1 -edge-distance-averaging 1 -optimizer EulerMethod -step 0.2 -steps 200 -epsilon 1e-6 -delta 0.001 -min-active 1% -reset-status -nointersection -fast-collision-test -min-width 0.1 -min-distance 0.1 -repulsion 4 -repulsion-distance 1.0 -repulsion-width 2.0 -curvature 4.0 -gauss-curvature 1.0 -gauss-curvature-minimum .1 -gauss-curvature-maximum .2 -gauss-curvature-outside 0.5 -edge-distance-type StrongestMinimum -threads 32 -remesh 1 -min-edge-length 1.0 -max-edge-length 2.0 -edge-distance-min-intensity 0.0003')

    # Make voxel-based mask from surface mask
    fmask_out = maskDir + \
        fmd_bname.replace('MD' + fsl_ext(), 'micromask' + fsl_ext())
    if not os.path.exists(fmask_out):
        sutil.surf_to_volume_mask(fmd, fmask_outsurf2, 1, fmask_out)

    return fmask_out


def fslv6p0_fake_brain_mask(fgibbs, bvals, tolerance=100):
    print("Performing Fake Brain Mask Extraction (FSL bet2)")

    fb0avg = fgibbs.replace(fsl_ext(), '_b0avg' + fsl_ext())
    in_file = fb0avg
    out_file = fb0avg.replace(fsl_ext(), '_BET')
    fmask = out_file.replace('BET', 'BET_mask' + fsl_ext())
    if not os.path.exists(fmask):
        img = nib.load(fgibbs)
        data = img.get_fdata()

        b0_idx = np.squeeze(
            np.array(np.where(np.logical_and(bvals < tolerance, bvals > -tolerance))))

        b0_data = data[..., b0_idx]
        meanB0 = np.mean(b0_data, axis=3)
        nib.save(nib.Nifti1Image(meanB0, img.affine), fb0avg)
        b0_idx = np.squeeze(
            np.array(np.where(np.logical_and(bvals < 15, bvals > -15))))

        nib.save(nib.Nifti1Image(meanB0, img.affine), out_file + fsl_ext())
        nib.save(nib.Nifti1Image(np.ones(meanB0.shape), img.affine), fmask)

    return fmask, fb0avg


def fslv6p0_brain_mask(fgibbs, bvals, fval, gval, tolerance=100):
    print("Performing Brain Mask Extraction (FSL bet2)")
    t = time()
    fb0avg = fgibbs.replace(fsl_ext(), '_b0avg' + fsl_ext())
    in_file = fb0avg
    out_file = fb0avg.replace(fsl_ext(), '_BET')

    # Make b0 average
    if not os.path.exists(fb0avg):
        img = nib.load(fgibbs)
        data = img.get_fdata()
        b0_idx = np.squeeze(
            np.array(np.where(np.logical_and(bvals < tolerance, bvals > -tolerance))))

        b0_data = data[..., b0_idx]
        meanB0 = np.mean(b0_data, axis=3)
        nib.save(nib.Nifti1Image(meanB0, img.affine), fb0avg)

    fmask = out_file.replace('BET', 'BET_mask' + fsl_ext())
    if not os.path.exists(fmask):
        if is_tool('bet2'):
            process = subprocess.run(['bet2',
                                      fb0avg,
                                      out_file,
                                      '-m',
                                      '-f', str(fval),
                                      '-g', str(gval)],
                                     stdout=subprocess.PIPE,
                                     universal_newlines=True)
            print("BET2 finished masking, Processing Time: ", time() - t)
            stdout = process.stdout
            return_code = process.returncode
        else:
            print(
                "microbrain Preproc: Could not find bet2, make sure FSLv6.0 or greater is installed to your path")
            stdout = ''
            return_code = 1
    else:
        print("microbrain Preproc: BET2 masking already performed skipping")
        stdout = ''
        return_code = 0

    return fmask, fb0avg, stdout, return_code


def fslv6p0_eddy(fdwi, facq, findex, fmask, fbval, fbvec, fjson, ftopup,  ffieldmap, cuda, eddyDir, eddyOutputDir):
    print("Performing Eddy Current / Motion Correction (FSL eddy), this may take a while (will implement a progress update in the future)")
    t = time()

    if not os.path.exists(eddyDir):
        subprocess.run(['mkdir', eddyDir],
                       stdout=subprocess.PIPE, universal_newlines=True)

    if cuda:
        eddy_command = 'eddy_cuda'
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
                                 '--data_is_shelled',
                                 '-v']
            if ftopup:
                full_eddy_command.append('--topup=' + ftopup)

            if ffieldmap:
                ffieldmap_without_ext = ffieldmap.replace(fsl_ext(), '')
                full_eddy_command.append('--field=' + ffieldmap_without_ext)

            if fjson:
                full_eddy_command.append('--json=' + fjson)

            # Verbose
            full_eddy_command.append('-v')

            process = subprocess.run(full_eddy_command,
                                     stdout=subprocess.PIPE,
                                     universal_newlines=True)
            stdout = process.stdout
            return_code = process.returncode

            # Move generated output files to another directory
            if not os.path.exists(eddyOutputDir):
                subprocess.run(['mkdir', eddyOutputDir],
                               stdout=subprocess.PIPE, universal_newlines=True)

            subprocess.run(['mv', feddy + '.eddy_command_txt', eddyOutputDir],
                           stdout=subprocess.PIPE, universal_newlines=True)
            subprocess.run(['mv', feddy + '.eddy_movement_rms', eddyOutputDir],
                           stdout=subprocess.PIPE, universal_newlines=True)
            subprocess.run(['mv', feddy + '.eddy_outlier_map', eddyOutputDir],
                           stdout=subprocess.PIPE, universal_newlines=True)
            subprocess.run(['mv', feddy + '.eddy_outlier_n_sqr_stdev_map',
                           eddyOutputDir], stdout=subprocess.PIPE, universal_newlines=True)
            subprocess.run(['mv', feddy + '.eddy_outlier_n_stdev_map',
                           eddyOutputDir], stdout=subprocess.PIPE, universal_newlines=True)
            subprocess.run(['mv', feddy + '.eddy_outlier_report', eddyOutputDir],
                           stdout=subprocess.PIPE, universal_newlines=True)
            subprocess.run(['mv', feddy + '.eddy_parameters', eddyOutputDir],
                           stdout=subprocess.PIPE, universal_newlines=True)
            subprocess.run(['mv', feddy + '.eddy_post_eddy_shell_alignment_parameters',
                           eddyOutputDir], stdout=subprocess.PIPE, universal_newlines=True)
            subprocess.run(['mv', feddy + '.eddy_post_eddy_shell_PE_translation_parameters',
                           eddyOutputDir], stdout=subprocess.PIPE, universal_newlines=True)
            subprocess.run(['mv', feddy + '.eddy_restricted_movement_rms',
                           eddyOutputDir], stdout=subprocess.PIPE, universal_newlines=True)
            subprocess.run(['mv', feddy + '.eddy_values_of_all_input_parameters',
                           eddyOutputDir], stdout=subprocess.PIPE, universal_newlines=True)

            print("Eddy Current/Motion Finished, Processing Time: ", time() - t)
        else:
            print("microbrain Preproc: Could not find " + eddy_command +
                  ", make sure FSLv6.0 or greater is installed to your path")
            stdout = ''
            return_code = 1
    else:
        print("Eddy Correction Already Done: Skipping")
        stdout = ''
        return_code = 0

    return feddy, fbvec_rotated, stdout, return_code


def fslv6p0_topup(fb0s, facq):
    print("Performing Spatial Distortion Correction (Topup, FSL eddy), this may take a while (will implement a progress update in the future)")
    t = time()

    ftopup = fb0s.replace(fsl_ext(), '_topup')
    ftopup_field = fb0s.replace(fsl_ext(), '_topup_field')
    ftopup_unwarped = fb0s.replace(fsl_ext(), '_topup_uwarped')
    if not os.path.exists(ftopup_unwarped + fsl_ext()):
        if is_tool('topup'):
            process = subprocess.run(['topup', '--imain=' + fb0s, '--datain=' + facq, '--config=b02b0.cnf', '--out=' + ftopup,
                                     '--fout=' + ftopup_field, '--iout=' + ftopup_unwarped, '-v'], stdout=subprocess.PIPE, universal_newlines=True)
            stdout = process.stdout
            return_code = process.returncode
        else:
            print(
                "microbrain Preproc: Could not find topup, make sure it is installed to your path")
            stdout = ''
            return_code = 1
    else:
        print("Topup already done, skipping")
        stdout = ''
        return_code = 1

    ftopup_unwarped = ftopup_unwarped + fsl_ext()

    return ftopup, ftopup_unwarped, stdout, return_code


def n4correct_by_b0(fb0avg, fmask, fgibbs, preprocDWIDir):
    print("N4 correction (ANTS)")
    t = time()

    fn4correct = fb0avg.replace(fsl_ext(), '_n4' + fsl_ext())
    fbias = fn4correct.replace(fsl_ext(), '_bias' + fsl_ext())

    if not os.path.exists(fn4correct):
        if is_tool('N4BiasFieldCorrection'):
            print('fslcpgeom start')
            process = subprocess.run(
                ['fslcpgeom', fb0avg, fmask], stdout=subprocess.PIPE, universal_newlines=True)
            print('fslcpgeom end')
            process = subprocess.run(['N4BiasFieldCorrection',
                                      '-i', fb0avg,
                                      '-x', fmask,
                                      '-o', '[' + fn4correct + ',' + fbias + ']'],
                                     stdout=subprocess.PIPE,
                                     universal_newlines=True)
            print("N4 finished, Processing Time: ", time() - t)
            stdout = process.stdout
            return_code = process.returncode
        else:
            print(
                "microbrain Preproc: Could not find N4, make sure it is installed to your path")
            stdout = ''
            return_code = 1
    else:
        print("N4 Correction previously done: Skipping")
        stdout = ''
        return_code = 0

    biasField = nib.load(fbias).get_fdata()
    mask = nib.load(fmask).get_fdata()
    fgibbs_basename = os.path.basename(fgibbs)
    fn4dwi = preprocDWIDir + \
        fgibbs_basename.replace(fsl_ext(), '_N4' + fsl_ext())
    print(fn4dwi)
    if not os.path.exists(fn4dwi):
        gibbs_img = nib.load(fgibbs)
        gibbs_data = gibbs_img.get_fdata()

        n4dwi_data = np.zeros(gibbs_data.shape)

        for vol_idx in range(0, n4dwi_data.shape[3]):
            thisVol = np.squeeze(gibbs_data[:, :, :, vol_idx])

            n4dwi_data[:, :, :, vol_idx] = thisVol / biasField
        # n4dwi_data[mask == 0] = 0
        nib.save(nib.Nifti1Image(n4dwi_data, gibbs_img.affine), fn4dwi)

    return fn4dwi, stdout, return_code


def output_DWI_maps(fb0avg, fmask, feddy, bvals, shells, meanDWIDir, preproc_suffix, dwi_shell=1000, tolerance=100):

    shell_tolerance = 50  # multiband doesn't prescribe the exact b-value
    print("N4 correction (ANTS)")
    t = time()

    if not os.path.exists(meanDWIDir):
        subprocess.run(['mkdir', meanDWIDir],
                       stdout=subprocess.PIPE, universal_newlines=True)

    fn4correct = fb0avg.replace(fsl_ext(), '_n4' + fsl_ext())
    fbias = fn4correct.replace(fsl_ext(), '_bias' + fsl_ext())

    if not os.path.exists(fn4correct):
        if is_tool('N4BiasFieldCorrection'):
            process = subprocess.run(
                ['fslcpgeom', fb0avg, fmask], stdout=subprocess.PIPE, universal_newlines=True)
            process = subprocess.run(['N4BiasFieldCorrection',
                                      '-i', fb0avg,
                                      '-x', fmask,
                                      '-o', '[' + fn4correct + ',' + fbias + ']', '-v'],
                                     stdout=subprocess.PIPE,
                                     universal_newlines=True)
            print("N4 finished, Processing Time: ", time() - t)
            stdout = process.stdout
            return_code = process.returncode
        else:
            print(
                "microbrain Preproc: Could not find N4, make sure it is installed to your path")
            stdout = ''
            return_code = 1
    else:
        print("N4 Correction previously done: Skipping")
        stdout = ''
        return_code = 0

    biasField = nib.load(fbias).get_fdata()
    mask = nib.load(fmask).get_fdata()
    feddy_basename = os.path.basename(feddy)
    fmeanshell = meanDWIDir + \
        feddy_basename.replace(fsl_ext(), 'mean_b' +
                               str(shells[0]) + fsl_ext())
    if not os.path.exists(fmeanshell):
        eddy_img = nib.load(feddy)
        eddy_data = eddy_img.get_fdata()

    for shell in shells:
        fmeanshell = meanDWIDir + \
            feddy_basename.replace(
                fsl_ext(), '_mean_b' + str(shell) + fsl_ext())
        fmeanshell_n4 = meanDWIDir + \
            feddy_basename.replace(
                fsl_ext(), '_mean_b' + str(shell) + '_n4' + fsl_ext())
        if shell == 0:
            fb0 = fmeanshell
            fb0_n4 = fmeanshell_n4

        if shell == dwi_shell:
            fdwi = fmeanshell
            fdwi_n4 = fmeanshell_n4

        if not os.path.exists(fmeanshell):
            bshell_idx = np.squeeze(np.array(np.where(np.logical_and(
                bvals < shell + tolerance, bvals > shell - tolerance))))
            bshell_data = eddy_data[..., bshell_idx]

            meanBSHELL = np.mean(bshell_data, axis=3)
            meanBSHELL[mask == 0] = 0
            nib.save(nib.Nifti1Image(meanBSHELL, eddy_img.affine), fmeanshell)

            meanBSHELL_n4 = np.mean(bshell_data, axis=3) / biasField
            meanBSHELL_n4[mask == 0] = 0
            nib.save(nib.Nifti1Image(meanBSHELL_n4,
                     eddy_img.affine), fmeanshell_n4)

    return fb0, fb0_n4, fdwi, fdwi_n4, stdout, return_code


def output_DWI_maps_noN4(fb0avg, fmask, feddy, bvals, shells, meanDWIDir, preproc_suffix, dwi_shell=1000, tolerance=100):

    shell_tolerance = 50  # multiband doesn't prescribe the exact b-value
    print("No N4 correction")
    t = time()

    if not os.path.exists(meanDWIDir):
        subprocess.run(['mkdir', meanDWIDir],
                       stdout=subprocess.PIPE, universal_newlines=True)

    mask = nib.load(fmask).get_fdata()
    fn4correct = fb0avg.replace(fsl_ext(), '_n4' + fsl_ext())
    feddy_basename = os.path.basename(feddy)
    fmeanshell = meanDWIDir + \
        feddy_basename.replace(fsl_ext(), 'mean_b' +
                               str(shells[0]) + fsl_ext())
    if not os.path.exists(fmeanshell):
        eddy_img = nib.load(feddy)
        eddy_data = eddy_img.get_fdata()

    for shell_idx in range(0, len(shells)):
        shell = shells[shell_idx]
        fmeanshell = meanDWIDir + \
            feddy_basename.replace(
                fsl_ext(), '_mean_b' + str(shell) + fsl_ext())
        fmeanshell_n4 = meanDWIDir + \
            feddy_basename.replace(
                fsl_ext(), '_mean_b' + str(shell) + '_n4' + fsl_ext())
        if shell_idx == 0:
            fbase = fmeanshell
            fbase_n4 = fmeanshell_n4

        if shell_idx == 1:
            fdwi = fmeanshell
            fdwi_n4 = fmeanshell_n4

        if not os.path.exists(fmeanshell):
            bshell_idx = np.squeeze(np.array(np.where(np.logical_and(
                bvals < shell + tolerance, bvals > shell - tolerance))))
            bshell_data = eddy_data[..., bshell_idx]

            meanBSHELL = np.mean(bshell_data, axis=3)
            meanBSHELL[mask == 0] = 0
            nib.save(nib.Nifti1Image(meanBSHELL, eddy_img.affine), fmeanshell)

            meanBSHELL_n4 = meanBSHELL
            nib.save(nib.Nifti1Image(meanBSHELL_n4,
                     eddy_img.affine), fmeanshell_n4)

    return fbase, fbase_n4, fdwi, fdwi_n4
