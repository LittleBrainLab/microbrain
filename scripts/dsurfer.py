#!/usr/local/bin/python
import os,sys,getopt
import shutil
import glob
from os import system
from os import path
from time import time

import numpy as np
import nibabel as nib

from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table

sys.path.append('../preprocessing/')
import dsurf_preproc as dsurf_preproc

sys.path.append('../modelling/')
import dsurf_modelling as dsurf_modelling

sys.path.append('../segmentation/')
import dsurf_voxel_segmentation as dsurf_seg

sys.path.append('../surfing/')
import dsurf_cortical_segmentation as dsurf_cort

def main(argv):
    inputDir = ''
    outputDir = ''
    bval_list =[]
    
    mask = False
    denoise = False
    dnnl = False
    gibbs = False
    eddy = False
    json = True
    dti_model = False
    vb_seg = False
    cort_seg = False
    try:
        opts, args = getopt.getopt(argv,"hs:b:i:",["idcm=","subdir=","bvalues=", "denoise","dnnl", "gibbs", "eddy", "no-json","all","segall", "hcp", "cb","allxeddy", "allxn4","allxcort"])
    except getopt.GetoptError:
        print('dsurfer.py -s <Subject Directory> -b <bvaluelist> [options]')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('dsurfer.py -s <Subject Directory> -b <bvaluelist> [options]')
            sys.exit()
        elif opt in ("-i", "--idcm"):
            inputDir = os.path.normpath(arg)
        elif opt in ("-s", "--subdir"):
            outputDir = os.path.normpath(arg)
        elif opt in ("-b", "--bvalues"):
            dti_model = True
            bval_list = np.fromstring(arg.strip('[]'), np.int, sep=',')
            shell_suffix = ''
            for shell in bval_list:
                shell_suffix = shell_suffix + 'b' + str(shell)

        elif opt in ("--denoise"):
            denoise = True
        elif opt in ("--dnnl"):
            mask = True
            dnnl = True
        elif opt in ("--gibbs"):
            gibbs = True
        elif opt in ("--eddy"):
            mask = True
            eddy = True
        elif opt in ("--no-json"):
            json = False
        elif opt in ("--segall"):
            vb_seg = True
            cort_seg = True
        elif opt in ("--all"):
            mask=True
            gibbs = True
            eddy = True
            dti_model = True
            vb_seg = True
            cort_seg = True
        elif opt in ("--allxcort"):
            mask=True
            gibbs = True
            eddy = True
            dti_model = True
            vb_seg = True
            cort_seg = False
            N4 = True
        elif opt in ("--cb"): # CB_BRAIN Preproc/DTI Maps
            mask=True
            eddy = True
            dti_model = True
            vb_seg = True
            cort_seg = True
        elif opt in("--hcp"):
            mask=True
            dti_model = True
            vb_seg = True
            cort_seg = True
        elif opt in ("--allxeddy"):
            mask=True
            gibbs = True
            dti_model = True
            vb_seg = True
            cort_seg = Truei
        elif opt in ("--allxn4"):
            mask=True
            gibbs = True
            eddy = True
            N4 = False
            dti_model = True
            vb_seg = True
            cort_seg = True


    outputDir, subID = os.path.split(outputDir)
    print('Processing:' + subID)
    print('OutDir: ' + outputDir) 
    
    total_t_start = time()
    print("Starting dMRI processing for: " + subID)
    
    # Given an input folder, preprocess data and output into appropriate folder.
    [fdwi, fbval, fbvec, fjson] = dsurf_preproc.dcm2nii(inputDir, outputDir, 'orig/', subID)
    origDir = outputDir + '/' + subID + '/orig/'
    
    ## Denoising/Gibbs ringing correction
    preprocDir = outputDir + '/' + subID + '/preprocessed/'
    fout = fdwi
    preproc_suffix = ''
    if denoise:
        fout = dsurf_preproc.denoiseMPPCA(fout, fbval, fbvec, preprocDir, patch_radius=2)
        preproc_suffix = preproc_suffix + '_DN'

    if gibbs:
        fout = dsurf_preproc.gibbsRingingCorrection(fout, preprocDir)
        preproc_suffix = preproc_suffix + '_GR'
    
    # mask data using meanB0
    bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
    if mask:
        fmask, fb0avg = dsurf_preproc.fslv6p0_brain_mask(fout,bvals)
    else:
        fmask, fb0avg = dsurf_preproc.fslv6p0_fake_brain_mask(fout,bvals)

    if dnnl:
        fout = dsurf_preproc.denoiseNONLOCAL(fout, fmask, preprocDir)
        preproc_suffix = preproc_suffix + '_DNNL'

    # Eddy current correction
    if eddy:
        # Make index file based on bvalues
        findex = 'index.txt'
        index = np.ones(bvals.shape)
        np.savetxt('index.txt', index, delimiter='', newline=' ', fmt='%01d')
        
        # Make acquisition file with incorrrect echo spacing (this is only useful if you need the eddy correction maps scaled properly.
        facq = 'acqparam.txt' 
        acq = np.array([0,-1,0,0.10064]).T
        np.savetxt('acqparam.txt', acq, delimiter='', newline=' ',fmt='%.1f')

        if not json:
            fjson = False
        fout, fbvec_rotated = dsurf_preproc.fslv6p0_eddy(fout, facq, findex, fmask, fbval, fbvec, fjson, preprocDir, preprocDir + 'eddy_output_files/')
        bvals, bvecs = read_bvals_bvecs(fbval, fbvec_rotated) # update bvecs with rotated version from eddy correction
        
        # Remove the two tmp files
        system('rm ' + findex + ' ' + facq)
        preproc_suffix = preproc_suffix + '_EDDY'

    # Remove preceding _
    if preproc_suffix != '':
        preproc_suffix =  preproc_suffix[1:]
        suffix = preproc_suffix + '_' + shell_suffix
    else:
        preproc_suffix = ''
        suffix = '_' + shell_suffix

    meanDWIDir = outputDir + '/' + subID + '/meanDWI/'
    tensorDir = outputDir + '/' + subID + '/DTI_maps/'
    if dti_model:
        # Output Average DWI maps for each shell, as well as n4 corrected versions
        if N4:
            fb0, fb0_n4, fdwi, fdwi_n4 = dsurf_preproc.output_DWI_maps(fb0avg, fmask, fout, bvals, bval_list, meanDWIDir, preproc_suffix, dwi_shell = bval_list[-1])
        else: 
            fb0, fb0_n4, fdwi, fdwi_n4 = dsurf_preproc.output_DWI_maps_noN4(fb0avg, fmask, fout, bvals, bval_list, meanDWIDir, preproc_suffix, dwi_shell = bval_list[-1])

        
        # Output Tensor and associated parametric maps
        ffa, fmd = dsurf_modelling.output_DTI_maps_multishell(fout, fmask, bvals, bvecs, tensorDir, shells = bval_list)
        
        if vb_seg:
            dsurf_seg.register_and_segment(fb0_n4, fdwi_n4, ffa, fmask, outputDir, subID, suffix)

        if cort_seg:
            print("DSurfing")
            src_freesurf_subdir = '/usr/local/freesurfer/subjects/TEMP_CB_BRAIN_CS/'
            freesurf_subdir = '/usr/local/freesurfer/subjects/DSURFER_' + subID + '/'
            os.system('cp -r ' + src_freesurf_subdir + ' ' + freesurf_subdir)
            dsurf_cort.generate_surfaces_from_dwi(outputDir, subID, preproc_suffix, shell_suffix, freesurf_subdir)
    
    print("Total time for processing: ", time() - total_t_start)
    print("")

    return

if __name__ == "__main__":
   main(sys.argv[1:])
