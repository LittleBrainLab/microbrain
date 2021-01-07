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
import mbrain_preproc as mbrain_preproc

sys.path.append('../modelling/')
import mbrain_modelling as mbrain_modelling

#sys.path.append('../segmentation/')
#import mbrain_voxel_segmentation as mbrain_seg

#sys.path.append('../surfing/')
#import mbrain_cortical_segmentation as mbrain_cort

def main(argv):
    inputDir = ''
    outputDir = ''
    bval_list =[]
    
    mask = False
    denoise = False
    dnnl = False
    gibbs = False
    eddy = False
    cuda = False
    json = True
    dti_model = False
    vb_seg = False
    cort_seg = False
    N4 = True # By default do N4 correction on DWI image based on the bias field from b0

    help_string = """usage: microbrain.py -s <Subject Directory> -b <bvaluelist> [options]
    description: microbrain is a fully automated diffusion MRI analysis pipeline for measurement of grey matter microstructure.  
    Note that options below require the installation of multiple neuroimaging/python analysis software packages (see install instructions).
    VERY BETA Version use with extreme caution. For trouble shooting help contact Graham Little (gtlittle@ualberta.ca)

    options: 
    -i <dicom_directory>,--idcm= - uses dcm2niix to convert dicom files to NIFTI which are then placed in a newly created subject directory
    -s <directory>,--subdir= - specifies the directory which microbrain will place all processed files
    -b [bval_list],--bvalues= - specify the bvalues to use for diffusion modelling and segmention (for example [0,1000] to use all b0 and b1000 volumes) 
    --gibbs - perform gibbs ringing correction (Kellner et al., 2016) using included third party c program
    --denoise - performs MPPCA denoising (Veraart et al., 2016) via DIPY
    --dnnl - performs Non-Local means denoising (Coupe et al., 2011) via DIPY
    --eddy, --eddy_cuda - performs eddy current correction (Anderson et al., 2016) via FSL v6.0 or higher. CUDA mode runs faster if CUDA setup on NVIDIA GPU
    --no-N4 - does not perform N4 correction (Tustison el al., 2010) via ANTS, useful flag when data is prescan normalized (Siemens)
    --no-json - if no .json image acquistion specification file available will run without this file
    --subcort_seg - perform surface based subcortical segmentation (Little et al., 2021, ISMRM) via MIRTK
    --cort_seg - perform surface based cortical segmentation (Little et al., 2021, NeuroImage) via MIRTK
    
    packaged_pipelines
    --cb - (used for CB_BRAIN data) 1) eddy_cuda 2) modelling tensor 3) subcortical segmentation 4) cortical segmentation 
        example: microbrain.py -s CB_BRAIN_050 -b [0,1000] --cb
    --all - (used for ALB300 data) 1) gibbs ringing correction 2) eddy_cuda 3) subcortical segmenation 4) cortical segmentation
        example: microbrain.py -i Ab300_005/study/DTI_1p5mm...45b2500_12/ -s AB300_005 -b [0,1000] --all"""

    try:
        # Note some of these options were left for testing purposes
        opts, args = getopt.getopt(argv,"hs:b:i:",["idcm=","subdir=","bvalues=", "denoise","dnnl", "gibbs", "eddy", "eddy_cuda", "no-json","no-N4","all","segall", "hcp", "cb","allxeddy", "allxn4","allxcort"])
    except getopt.GetoptError:
        print(help_string)
        sys.exit(2)
    
    if len(opts) == 0:
        print(help_string)
        sys.exit(2)
        
    for opt, arg in opts:
        if opt == '-h':
            print(help_string)
            
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
            cuda = False
            mask = True
            eddy = True
        elif opt in ("--eddy_cuda"):
            cuda = True
            mask = True
            eddy = True
        elif opt in ("--no-json"):
            json = False
        elif opt in ("--no-N4"):
            N4 = False
        elif opt in ("--segall"):
            vb_seg = True
            cort_seg = True
        elif opt in ("--all"):
            mask = True
            gibbs = True
            eddy = True
            cuda = True
            dti_model = True
            vb_seg = True
            cort_seg = True
            N4 = True
        elif opt in ("--allxcort"):
            mask = True
            gibbs = True
            eddy = True
            cuda = True
            dti_model = True
            vb_seg = True
            cort_seg = False
            N4 = True
        elif opt in ("--cb"): # CB_BRAIN Preproc/DTI Maps
            mask = True
            eddy = True
            cuda = True
            dti_model = True
            vb_seg = True
            cort_seg = True
            N4 = True
        elif opt in("--hcp"):
            mask = True
            dti_model = True
            vb_seg = True
            cort_seg = True
            N4 = True
        elif opt in ("--allxeddy"):
            mask=True
            gibbs = True
            dti_model = True
            vb_seg = True
            cort_seg = True
            N4 = True
        elif opt in ("--allxn4"):
            mask=True
            gibbs = True
            eddy = True
            cuda = True
            N4 = False
            dti_model = True
            vb_seg = True
            cort_seg = True

    outputDir, subID = os.path.split(outputDir)
    print('Processing:' + subID)
    print('OutDir: ' + outputDir) 
    
    total_t_start = time()
    print("Starting dMRI processing for: " + subID)
    
    # Given an input folder, convert to NIFTI and output into appropriate folder.
    # Function will skip this step if orig folder already exists with converted NIFTI File
    [fdwi, fbval, fbvec, fjson, stdout, returncode] = mbrain_preproc.dcm2nii(inputDir, outputDir, 'orig/', subID)
    if returncode != 0:
        print('DSurfer: dcm2niix returned an error, make sure it is installed correctly and that dicom files exist')
        sys.exit()


    origDir = outputDir + '/' + subID + '/orig/'
    
    ## Denoising/Gibbs ringing correction
    preprocDir = outputDir + '/' + subID + '/preprocessed/'
    fout = fdwi
    preproc_suffix = ''
    if denoise:
        fout = mbrain_preproc.denoiseMPPCA(fout, fbval, fbvec, preprocDir, patch_radius=2)
        preproc_suffix = preproc_suffix + '_DN'

    if gibbs:
        fout, stdout, returncode = mbrain_preproc.gibbsRingingCorrection(fout, preprocDir)
        if returncode != 0:
            print('DSurfer: unring.a64 returned an error, make sure it is installed correctly')
            sys.exit()

        preproc_suffix = preproc_suffix + '_GR'
    
    # mask data using meanB0
    bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
    if mask:
        fmask, fb0avg, stdout, returncode = mbrain_preproc.fslv6p0_brain_mask(fout,bvals)
        if returncode != 0:
            print('DSurfer: bet2 returned an error, make sure it is installed correctly')
            sys.exit()
    else:
        fmask, fb0avg = mbrain_preproc.fslv6p0_fake_brain_mask(fout,bvals)

    if dnnl:
        fout = mbrain_preproc.denoiseNONLOCAL(fout, fmask, preprocDir)
        preproc_suffix = preproc_suffix + '_DNNL'

    # Eddy current correction
    if eddy:
        # Make index file based on bvalues
        findex = 'index.txt'
        index = np.ones(bvals.shape)
        np.savetxt('index.txt', index, delimiter='', newline=' ', fmt='%01d')
        
        # Make acquisition file with incorrrect echo spacing (this is only useful if you need the eddy correction maps scaled properly).
        facq = 'acqparam.txt' 
        acq = np.array([0,-1,0,0.10064]).T
        np.savetxt('acqparam.txt', acq, delimiter='', newline=' ',fmt='%.1f')

        if not json:
            fjson = False
        
        fout, fbvec_rotated, stdout, returncode = mbrain_preproc.fslv6p0_eddy(fout, facq, findex, fmask, fbval, fbvec, fjson, cuda, preprocDir, preprocDir + 'eddy_output_files/')
        if returncode != 0:
            print("DSurfer: FSL's eddy returned an error, make sure it is installed correctly")
            sys.exit()

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
        # Output Average DWI maps for each shell, as well as N4 corrected versions. Note, N4 correction applied to mean DWI not to the raw data itself)
        if N4:
            fb0, fb0_n4, fdwi, fdwi_n4, stdout, returncode = mbrain_preproc.output_DWI_maps(fb0avg, fmask, fout, bvals, bval_list, meanDWIDir, preproc_suffix, dwi_shell = bval_list[-1])
            if returncode != 0:
                print("DSurfer: N4 returned an error, make sure it is installed correctly")
                sys.exit()
        else: 
            fb0, fb0_n4, fdwi, fdwi_n4 = mbrain_preproc.output_DWI_maps_noN4(fb0avg, fmask, fout, bvals, bval_list, meanDWIDir, preproc_suffix, dwi_shell = bval_list[-1])

        # Output Tensor and associated diffusion parametric maps
        ffa, fmd = mbrain_modelling.output_DTI_maps_multishell(fout, fmask, bvals, bvecs, tensorDir, shells = bval_list)
        
        # Will be Implementing this soon
        #if vb_seg:
        #    mbrain_seg.register_and_segment(fb0_n4, fdwi_n4, ffa, fmask, outputDir, subID, suffix)

        # Will be finishing implementing this soon
        #if cort_seg:
        #    print("DSurfing")
        #    src_freesurf_subdir = '/usr/local/freesurfer/subjects/TEMP_CB_BRAIN_CS/'
        #    freesurf_subdir = '/usr/local/freesurfer/subjects/DSURFER_' + subID + '/'
        #    os.system('cp -r ' + src_freesurf_subdir + ' ' + freesurf_subdir)
        #    mbrain_cort.generate_surfaces_from_dwi(outputDir, subID, preproc_suffix, shell_suffix, freesurf_subdir)
    
    print("Total time for processing: ", time() - total_t_start)
    print("")

    return

if __name__ == "__main__":
   main(sys.argv[1:])
