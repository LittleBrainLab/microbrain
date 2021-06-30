#!/usr/local/bin/python
import os,sys,getopt
import shutil
import glob
from os import system
from os import path
from time import time
import subprocess

import numpy as np
import nibabel as nib

from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table
import dipy.reconst.dti as dti

sys.path.append('../preprocessing/')
import mbrain_preproc as mbrain_preproc

sys.path.append('../modelling/')
import mbrain_modelling as mbrain_modelling

sys.path.append('../segmentation-v2/')
import mbrain_segment as mbrain_seg

#sys.path.append('../surfing/')
#import mbrain_cortical_segmentation as mbrain_cort
def fsl_ext():
    fsl_extension = ''
    if os.environ['FSLOUTPUTTYPE'] == 'NIFTI':
        fsl_extension = '.nii'
    elif os.environ['FSLOUTPUTTYPE'] == 'NIFTI_GZ':
        fsl_extension = '.nii.gz'
    return fsl_extension

def main(argv):
    inputDir = ''
    outputDir = ''
    bval_list =[]
    
    # Three masking options
    bet_mask = False
    microbrain_mask = False
    explicit_mask = ''
    
    fval = 0.3
    gval = 0.0
    rerun_mask = False
    stabilize = False
    dnlsam = False
    dmppca = False
    gibbs = False
    eddy = False
    topup = False
    fieldmap = False
    pe_direction_specified = False
    acq_readout_specified = False
    acq_readout = 0.05
    effective_echo = False
    pe_direction = ''
    cuda = False
    json = True
    dti_model = False
    diffusion_seg = False
    freesurfDir = False
    N4 = True # By default do N4 correction on DWI image based on the bias field from b0
    cpu_num = 0

    help_string = """usage: microbrain.py -s <Subject Directory> -b <bvaluelist> [options]
    description: microbrain is a fully automated diffusion MRI analysis pipeline for measurement of grey matter microstructure.  
    Note that options below require the installation of multiple neuroimaging/python analysis software packages (see install instructions).
    VERY BETA Version use with extreme caution. For trouble shooting help contact Graham Little (gtlittle@ualberta.ca)

    mandatory arguments:
    -s <directory>,--subdir= - specifies the directory which microbrain will place all processed files
    -b [bval_list],--bvalues= - specify the bvalues to use for diffusion modelling and segmention (for example [0,1000] to use all b0 and b1000 volumes)

    optional arguments: 
    --mask-params=[f-value,g-value], --- sets the bet2 mask parameters default [0.3,0.0]
    --rerun-mask - deletes any previously created files (post-masking step) and reruns entire analysis using specified mask parameters 
    --gibbs - perform gibbs ringing correction (Kellner et al., 2016) using included third party c program
    --stabilize - perform stabilization using algorithm provided by non-local spatial angular matching algo (NLSAM, St Jean et al., 2016)
    --dnlsam - perform denoising using non-local spatial angular matching (NLSAM, St Jean et al., 2016)
    --dnppca - performs MPPCA denoising (Veraart et al., 2016) via DIPY
    --eddy --eddy_cuda - performs eddy current correction (Anderson et al., 2016) via FSL v6.0 or higher. CUDA mode runs faster if CUDA setup on NVIDIA GPU
    
    --pe_direction=phase_encode_direction - direction of diffusion data in orig directory. phase_encode_direction can be LR for left/right or AP for anterior / posterior and their opposite pairs.  Required for distortion correction using a fieldmap or reverse phase encode data
    --AcqReadout= - Time between centre of first echo to centre of last echo as defined by FSL. Only required if using fieldmap spatial distortion correction 
    --EffectiveEcho= - Time between centre of first echo to centre of last. Only required if using fieldmap spatial distortion correction 

    --no-N4 - does not perform N4 correction (Tustison el al., 2010) via ANTS, useful flag when data is prescan normalized (Siemens)
    --no-json - if no .json image acquistion specification file available will run without this file
    --mbrain-seg - perform surface based subcortical/cortical segmentation (Little et al., 2021, ISMRM and Little et al., 2021, NeuroImage) via MIRTK
    --freesurf-dir - given a freesurfer subject directory use BBR to register T1 surfaces / segmentations to diffusion data and perform all analysis using these segmentations rather than segmenting on DTI itself. This assumes diffusion is in perfect alignment with the diffusion data (i.e. corrected for spatial distortions). Useful for lower resolution data where microbrain segmentation would fail.
   
    --cpu-num - number of cpus to use for nlsam denoising and surface-based subcortical/cortical segmentation (MIRTK) if not set will use defaults for those programs

    Examples Different Distortion Correction Methods:

    1) No spatial distortion correction, b1000
    python3 microbrain_setup.py -s path_to_subject_directory -i path_to_dicom_directory
    python3 microbrain.py -s path_to_subject_directory -b [0,1000] --all
    
    2) Spatial Distortion Correction with reverse phase encode
    python3 microbrain_setup.py -s path_to_subject_directory -i path_to_dicom_directory --idcm_reversePE=path_to_dicom_directory_with_reverse_PE_data
    python3 microbrain.py -s path_to_subject_directory -b [0,1000] --pe_direction=AP --all

    3) Spatial Distortion Correction with siemens fieldmap data
    python3 microbrain_setup.py -s path_to_subject_directory -i path_to_dicom_directory --idcm_fieldmap=[path_to_dicom_directory_with_magnitude_data, path_to_dicom_directory_with_phase_data, 2.46]
    python3 microbrain.py -s path_to_subject_directory -b [0,1000] --pe_direction=AP --AcqReadout=0.04999 --EffectiveEcho=0.00034 --all

    Examples UofAlberta Data:
    --cb - (used for CB_BRAIN data, no spatial distortion correction) 1) eddy_cuda 2) modelling tensor 3) subcortical segmentation 4) cortical segmentation 
        python3 microbrain_setup.py -s somepath/CB_BRAIN_050 -i path_to_dicom_directory_containing_diffusion
        python3 microbrain.py -s somepath/CB_BRAIN_050 -b [0,1000] --cb

    --all - (used for ALB300 data, example below combines with fieldmap spatial distortion correction) 1) gibbs ringing correction 2) eddy_cuda 3) subcortical segmenation 4) cortical segmentation
        python3 microbrain_setup.py -s somepath/AB300_005 -i Ab300_005/study/DTI_1p5mm...45b2500_12/ --idcm_fieldmap=[path_to_dicom_directory_with_magnitude_data, path_to_dicom_directory_with_phase_data, 2.46] 
        python3 microbrain.py -s somepath/AB300_005 -b [0,1000] --pe_direction=AP --AcqReadout=0.04999 --EffectiveEcho=0.00034 --all
    
    --rerun-mask, --mask-params - used when masking fails and pipeline needs to be rerun 
        python3 microbrain.py -s somepath/Ab300_0005 -b [0,1000] python3 microbrain_setup.py -s path_to_subject_directory -i path_to_dicom_directory --all --mask-params=[0.4,0.1] --rerun-mask"""

    try:
        # Note some of these options were left for testing purposes
        opts, args = getopt.getopt(argv,"hs:b:i:",["idcm=","subdir=","bvalues=", "bet-mask", "microbrain-mask", "explicit-mask=","pe_direction=","EffectiveEcho=", "AcqReadout=", "rerun-mask", "dmppca", "dnlsam", "cpu-num=","gibbs", "eddy", "eddy_cuda", "no-json", "no-N4", "all", "mbrain-seg", "freesurf-dir=", "hcp", "cb","allxeddy", "allxn4", "stabilize"])
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
        elif opt in ("-s", "--subdir"):
            outputDir = os.path.normpath(arg)
        elif opt in ("-b", "--bvalues"):
            dti_model = True
            bval_list = np.fromstring(arg.strip('[]'), np.int, sep=',')
            shell_suffix = ''
            for shell in bval_list:
                shell_suffix = shell_suffix + 'b' + str(shell)
        elif opt in ("--bet-mask"):
            bet_mask = True
        elif opt in ("--microbrain-mask"):
            microbrain_mask = True
        elif opt in ("--no-mask"):
            mask = False
        elif opt in ("--explicit-mask"):
            explicit_mask = os.path.normpath(arg)
        elif opt in ("--rerun-mask"):
            rerun_mask = True
        elif opt in ("--stabilize"):
            stabilize = True
        elif opt in ("--dmppca"):
            dmppca = True
        elif opt in ("--dnlsam"):
            dnlsam = True
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
        elif opt in ("--pe_direction"):
            pe_direction_specified = True
            pe_direction = arg
        elif opt in ("--AcqReadout"):
            acq_readout_specified = True
            acq_readout = float(arg)
        elif opt in ("--EffectiveEcho"):
            effective_echo = float(arg)
        elif opt in ("--no-json"):
            json = False
        elif opt in ("--no-N4"):
            N4 = False
        elif opt in ("--cpu-num"):
            proc_num = int(arg)
        elif opt in ("--mbrain-seg"):
            diffusion_seg = True
        elif opt in ("--freesurf-dir"):
            freesurfDir = os.path.normpath(arg)
        elif opt in ("--all"):
            mask = True
            gibbs = True
            eddy = True
            cuda = True
            dti_model = True
            diffusion_seg = True
            N4 = True
        elif opt in ("--cb"): 
            mask = True
            eddy = True
            cuda = True
            dti_model = True
            diffusion_seg = True
            N4 = True
        elif opt in("--hcp"):
            mask = True
            dti_model = True
            diffusion_seg = True
            N4 = True

    outputDir, subID = os.path.split(outputDir)
    print('Processing:' + subID)
    print('OutDir: ' + outputDir) 
    
    total_t_start = time()
    print("Starting dMRI processing for: " + subID)
   
    # Naming convention for output directories
    origDir = outputDir + '/' + subID + '/orig/'
    preprocDir = outputDir + '/' + subID + '/preprocessed/'
    meanDWIDir = outputDir + '/' + subID + '/meanDWI/'
    tensorDir = outputDir + '/' + subID + '/DTI_maps/'
    maskDir = outputDir + '/' + subID + '/microbrain_mask/'
    regDir = outputDir + '/' + subID + '/registration/'
    subcortDir = outputDir + '/' + subID + '/subcortical_segmentation/'
    cortDir = outputDir + '/' + subID + '/cortical_segmentation/'

    if not os.path.exists(preprocDir):
        process = subprocess.run(['mkdir', preprocDir], stdout=subprocess.PIPE, universal_newlines=True)

    # Decide what type of spatial distortion correction should be performed based on what filenames exist in the subject directory
    fieldmapDir = outputDir + '/' + subID + '/orig_fieldmap/'
    reverseDir = outputDir + '/' + subID + '/orig_reversePE/'
    ffieldmap = ''
    if os.path.exists(fieldmapDir):
        print("Fieldmap found: using fieldmap for spatial distortion correction")
        if not pe_direction_specified or not acq_readout_specified or not effective_echo:
            print("Error:  AcqReadout, pe_direction and EffectiveEcho arguments needed to be specified to use field map distortion correction. Exiting")
            sys.exit()
        
        fieldmap = True
        ffieldmap_hz = fieldmapDir + subID + '_fieldmap_hz' + fsl_ext()
        ffieldmap_rad = fieldmapDir + subID + '_fieldmap_rad' + fsl_ext()
        ffieldmap_mag = fieldmapDir + subID + '_emean_brain1' + fsl_ext()

    elif os.path.exists(reverseDir):
        print("Reverse Phase Encode Data Found: using FSL topup for spatial distortion correction")
        if not pe_direction_specified:
            print("Error: PE direction argument needs to be specified to use topup distortion correction. Exiting")
            sys.exit()

        topup = True
        
    fdwi = origDir + subID + fsl_ext()
    fbval = fdwi.replace(fsl_ext(), '.bval')
    fbvec = fdwi.replace(fsl_ext(), '.bvec')
    fjson = fdwi.replace(fsl_ext(), '.json')

    ## Denoising/Gibbs ringing correction
    fout = fdwi
    bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
    preproc_suffix = ''
   
    # Generate a mask of data
    if explicit_mask != '':
        fmask = explicit_mask
    elif bet_mask:
        # Extract b0s
        fout_img = nib.load(fout)
        fout_data = fout_img.get_data()
        
        b0_data = fout_data[:,:,:,np.logical_or(bvals < 50, bvals > -50)]
        firstb0_data = b0_data[:,:,:,0]
        ffirstb0 = fout.replace(fsl_ext(), '_firstb0' + fsl_ext())
        nib.save(nib.Nifti1Image(firstb0_data, fout_img.affine), ffirstb0)
    
        fbrain = ffirstb0.replace(fsl_ext(), '_brain' + fsl_ext())
        fmask = ffirstb0.replace(fsl_ext(), '_brain_mask' + fsl_ext())
        process = subprocess.run(['bet',ffirstb0, fbrain, '-m','-f','0.3'], stdout=subprocess.PIPE, universal_newlines=True)
    
    elif microbrain_mask:
        fmd = fout.replace(fsl_ext(),'_MD' + fsl_ext())
        if not os.path.exists(fmd):
            fout_img = nib.load(fout)
            fout_data = fout_img.get_data()

            tolerance = 50
            b_idx = []
            for shell in bval_list:
                b_idx = np.append(b_idx, np.squeeze(np.array(np.where(np.logical_and(bvals < shell + tolerance, bvals > shell - tolerance)))))
            b_idx = b_idx.astype(int)

            print('Modelling Tensor using ' + str(b_idx.shape) + ' volumes:')
            print(b_idx)
            print('Bvals:')
            print(bvals[b_idx])

            sub_bvals = bvals[b_idx]
            sub_bvecs = bvecs[b_idx]
            fout_data = fout_data[:,:,:,b_idx]

            gtab = gradient_table(sub_bvals, sub_bvecs)
            
            print("Fitting Tensor for Mask")
            tenmodel = dti.TensorModel(gtab)
            tenfit = tenmodel.fit(fout_data)
            nib.save(nib.Nifti1Image(tenfit.md.astype(np.float32), fout_img.affine), fmd)
        else:
            print("Already Fit Tensor for Mask: Skipping")

        fmd_brain = fmd.replace(fsl_ext(), '_brain' + fsl_ext())
        fmask = fmd.replace(fsl_ext(), '_brain_mask' + fsl_ext())
        if not os.path.exists(fmask):
            print("Masking MD Map")
            process = subprocess.run(['bet',fmd, fmd_brain, '-m','-f','0.3'], stdout=subprocess.PIPE, universal_newlines=True)
        else:
            print("Already Masked MD Map: skipping")
    
    else: # Create a mask inclusive of all voxels
        fout_img = nib.load(fout)
        fout_data = fout_img.get_data()
        fmask = fout.replace(fsl_ext(), '_fake_mask' + fsl_ext())
        fake_mask_data = np.ones(np.shape(np.squeeze(fout_data[:,:,:,0])))
        nib.save(nib.Nifti1Image(fake_mask_data, fout_img.affine), fmask)

    # Signal stabilization and/or denoising with NLSAM (automatic estimates for N and sigma via st-jean 2020, Medical Image Analysis)
    if dnlsam:
        fout = mbrain_preproc.denoiseNLSAM(fout, fmask, fbval, fbvec, preprocDir, cpu_num=proc_num)
        preproc_suffix = preproc_suffix + '_DNLSAM'
    elif stabilize:
        fout = mbrain_preproc.stabilizeNLSAM(fout, fmask, fbval, fbvec, preprocDir)
        preproc_suffix = preproc_suffix + '_STAB'

    # Denosing using MPPCA
    if dmppca:
        fout = mbrain_preproc.denoiseMPPCA(fout, fbval, fbvec, preprocDir, patch_radius=2)
        preproc_suffix = preproc_suffix + '_DMPPCA'

    if gibbs:
        fout, stdout, returncode = mbrain_preproc.gibbsRingingCorrection(fout, preprocDir)
        if returncode != 0:
            print('DSurfer: unring.a64 returned an error, make sure it is installed correctly')
            sys.exit()
        
        preproc_suffix = preproc_suffix + '_GR'

    # Run FSL Topup if required
    ftopup = ''
    if topup:
        fpe_ind = fdwi.replace(fsl_ext(), '_PEIND.txt')
        pe_direction_ind = np.loadtxt(fpe_ind, delimiter=' ')
        b0_pe_direction = pe_direction_ind[np.logical_and(bvals >= -20, bvals <= 20)]
        
        facq = preprocDir + 'acqparam.txt'
        acq=np.zeros((b0_pe_direction.shape[0], 4))
        acq[:,3] = 0.05 # Note this number will make the off-resonance field not scaled properly
        
        if pe_direction == 'LR':
            acq[:,0] = b0_pe_direction
        if pe_direction == 'RL':
            acq[:,0] = b0_pe_direction * -1
        elif pe_direction == 'AP':
            acq[:,1] = b0_pe_direction
        elif pe_direction == 'PA':
            acq[:,1] = b0_pe_direction * -1

        np.savetxt(facq, acq, delimiter=' ',fmt='%.2f')

        findex = preprocDir + 'index.txt'
        index = np.ones(bvals.shape)
        first_reverseb0 = np.where(b0_pe_direction==1)[0]
        index[pe_direction_ind == 1] = first_reverseb0[0] + 1
        np.savetxt(findex, index, delimiter='', newline=' ', fmt='%01d')

        ftopup, ftopup_unwarped, stdout, returncode = mbrain_preproc.fslv6p0_topup(fb0s, facq)
    else:
        # Make index file based on bvalues
        findex = preprocDir + 'index.txt'
        index = np.ones(bvals.shape)
        np.savetxt(findex, index, delimiter='', newline=' ', fmt='%01d')

        # Make acquisition file with echo spacing
        facq = preprocDir + 'acqparam.txt'
        acq = [0, 1, 0, acq_readout]
        if fieldmap:
            if pe_direction == 'LR':
                acq = [-1, 0, 0, acq_readout]
            if pe_direction == 'RL':
                acq = [1, 0, 0, acq_readout]
            elif pe_direction == 'AP':
                acq = [0, -1, 0, acq_readout]
            elif pe_direction == 'PA':
                acq = [0, 1, 0, acq_readout]
        acq = np.array(acq).T
        np.savetxt(facq, acq[None,:], fmt=('%01d','%01d','%01d','%.6f'))

    #if mask:
        # If rerun mask flag, delete all data analyzed post mask and rerun all analysis
    #    if rerun_mask:
    #        for thisDir in [meanDWIDir, tensorDir, regDir, subcortDir, cortDir, preprocDir + 'eddy_output_files/']:
    #            if os.path.exists(thisDir):
    #                process = subprocess.run(['rm', '-r', thisDir], stdout=subprocess.PIPE, universal_newlines=True)
    #        
    #        feddy = fout.replace(fsl_ext(), '_EDDY' + fsl_ext())
    #        if os.path.exists(feddy):
    #            process = subprocess.run(['rm', feddy], stdout=subprocess.PIPE, universal_newlines=True)
    #            process = subprocess.run(['rm', feddy + '.eddy_rotated_bvecs'], stdout=subprocess.PIPE, universal_newlines=True)
            
            #fmask = fout.replace(fsl_ext(), '_b0avg_BET_mask' + fsl_ext())
            #if os.path.exists(fmask):
            #    process = subprocess.run(['rm', fmask], stdout=subprocess.PIPE, universal_newlines=True)

        #if topup:
            #fbrain = ftopup_unwarped.replace(fsl_ext(), '_brain' + fsl_ext())
            #fmask = ftopup_unwarped.replace(fsl_ext(), '_brain_mask' + fsl_ext())
            #if os.path.exists(fmask):
            #    process = subprocess.run(['rm', fmask], stdout=subprocess.PIPE, universal_newlines=True)
           
            #process = subprocess.run(['bet',ftopup_unwarped, fbrain, '-m','-f', str(fval)], stdout=subprocess.PIPE, universal_newlines=True)
            #fmask, fb0avg, stdout, returncode = mbrain_preproc.fslv6p0_brain_mask(ftopup_unwarped,np.zeros((b0_data.shape[3],)),fval,gval)
        #else:
            #fbrain = ffirstb0.replace(fsl_ext(), '_brain' + fsl_ext())
            #fmask = ffirstb0.replace(fsl_ext(), '_brain_mask' + fsl_ext())
            #if os.path.exists(fmask):
            #    process = subprocess.run(['rm', fmask], stdout=subprocess.PIPE, universal_newlines=True)
            
            #process = subprocess.run(['bet',ffirstb0, fbrain, '-m','-f', str(fval)], stdout=subprocess.PIPE, universal_newlines=True)
            #fmask, fb0avg, stdout, returncode = mbrain_preproc.fslv6p0_brain_mask(fb0s,np.zeros((b0_data.shape[3],)),fval,gval)
        
        #if returncode != 0:
        #    print('DSurfer: bet2 returned an error, make sure it is installed correctly')
        #    sys.exit()
    #else:
    #    # TODO Test this pipeline
    #    if topup:
    #        fmask, fb0avg = mbrain_preproc.fslv6p0_fake_brain_mask(ftopup_unwarped,np.zeros((b0_data.shape[3],)))
    #    else:
    #        fmask, fb0avg = mbrain_preproc.fslv6p0_fake_brain_mask(fb0s,np.zeros((b0_data.shape[3],)))

    # Eddy current correction
    if eddy:
        if not json:
            fjson = False
       
        # If using fieldmap register/interpolate it to the firstB0 image, this is needed prior to input into eddy
        ffieldmap_hz_reg2firstb0 = ''
        if fieldmap:
            ffirstb0_brain = ffirstb0.replace(fsl_ext(),'_brain' + fsl_ext())
            process = subprocess.run(['bet',ffirstb0, ffirstb0_brain, '-m','-f', str(fval)], stdout=subprocess.PIPE, universal_newlines=True)
            
            if pe_direction == 'AP':
                unwarp_direction = 'y-'
            elif pe_direction == 'PA':
                unwarp_direction = 'y'
            elif pe_direction == 'LR':
                unwarp_direction = 'x' # TODO Test this
            elif pe_direction == 'RL':
                unwarp_direction = 'x-' # TODO Test this

            ffieldmap_mag_warped = ffieldmap_mag.replace(fsl_ext(), '_warped' + fsl_ext())
            process = subprocess.run(['fugue','-v','-i', ffieldmap_mag, '--unwarpdir=' + unwarp_direction, '--dwell=' + str(effective_echo),'--loadfmap=' + ffieldmap_rad,'-w',  ffieldmap_mag_warped], stdout=subprocess.PIPE, universal_newlines=True)
            
            ffieldmap_mag_warped_2firstb0 = ffieldmap_mag_warped.replace(fsl_ext(),'_2firstb0' + fsl_ext())
            ffieldmap_2firstb0_mat = fieldmapDir + subID + '_fieldmap2firstb0.mat'
            process = subprocess.run(['flirt','-in',ffieldmap_mag_warped,'-ref',  ffirstb0_brain, '-out', ffieldmap_mag_warped_2firstb0,'-omat',ffieldmap_2firstb0_mat, '-dof', '6'], stdout=subprocess.PIPE, universal_newlines=True)

            ffieldmap_hz_reg2firstb0 = ffieldmap_hz.replace(fsl_ext(),'_reg2firstb0')
            process = subprocess.run(['flirt','-in',ffieldmap_hz,'-ref',ffirstb0_brain,'-applyxfm','-init',ffieldmap_2firstb0_mat,'-out',ffieldmap_hz_reg2firstb0,'-interp','spline'], stdout=subprocess.PIPE, universal_newlines=True)
            
            # Use giant fieldmap mask for input into eddy
            ffieldmap_mask = ffieldmap_mag.replace('brain1' + fsl_ext(),'brain_mask' + fsl_ext())
            fmask_field = ffirstb0.replace(fsl_ext(), '_mask_fieldmap' + fsl_ext())
            process = subprocess.run(['flirt','-in', ffieldmap_mask,'-ref',ffirstb0_brain,'-applyxfm','-init',ffieldmap_2firstb0_mat,'-out',fmask_field,'-interp','nearestneighbour'], stdout=subprocess.PIPE, universal_newlines=True)
            fmask_eddy = fmask_field

            #Fieldmap lies outside the b0 mask so dilate it
            #fmask_dil = fmask.replace(fsl_ext(), '_dil' + fsl_ext())
            #process = subprocess.run(['fslmaths', fmask, '-kernel', 'box', '15', '-dilF', fmask_dil], stdout=subprocess.PIPE, universal_newlines=True)
            #fmask = fmask_dil
        else:
            fmask_eddy = fmask

        fout, fbvec_rotated, stdout, returncode = mbrain_preproc.fslv6p0_eddy(fout, facq, findex, fmask_eddy, fbval, fbvec, fjson, ftopup, ffieldmap_hz_reg2firstb0, cuda, preprocDir, preprocDir + 'eddy_output_files/')
        if returncode != 0:
            print("DSurfer: FSL's eddy returned an error, make sure it is installed correctly")
            sys.exit()

        bvals, bvecs = read_bvals_bvecs(fbval, fbvec_rotated) # update bvecs with rotated version from eddy correction
        
        preproc_suffix = preproc_suffix + '_EDDY'

    # Remove preceding _
    if preproc_suffix != '':
        preproc_suffix =  preproc_suffix[1:]
        suffix = preproc_suffix + '_' + shell_suffix
    else:
        preproc_suffix = ''
        suffix = '_' + shell_suffix

    if dti_model:
        # Output Tensor and associated diffusion parametric maps
        ffa, fmd, ffa_rgb, fad, frd, fevec, fpevec, ftensor = mbrain_modelling.output_DTI_maps_multishell(fout, fmask, bvals, bvecs, tensorDir, shells = bval_list)
        
        # Output first b0 for N4 correction
        fout_img = nib.load(fout)
        fout_data = fout_img.get_data()
    
        b0_data = fout_data[:,:,:, np.logical_and(bvals >= -20, bvals <= 20)]

        firstb0_data = b0_data[:,:,:,0]
        ffirstb0_undistort = fout.replace(fsl_ext(), '_firstb0' + fsl_ext())
        nib.save(nib.Nifti1Image(firstb0_data, fout_img.affine), ffirstb0_undistort)
    
        # Output Average DWI maps for each shell, as well as N4 corrected versions. Note, N4 correction applied to mean DWI not to the raw data itself)
        if N4:
            fb0, fb0_n4, fdwi, fdwi_n4, stdout, returncode = mbrain_preproc.output_DWI_maps(ffirstb0_undistort, fmask, fout, bvals, bval_list, meanDWIDir, preproc_suffix, dwi_shell = bval_list[-1])
            if returncode != 0:
                print("DSurfer: N4 returned an error, make sure it is installed correctly")
                sys.exit()
        else:
            fb0, fb0_n4, fdwi, fdwi_n4 = mbrain_preproc.output_DWI_maps_noN4(ffirstb0_undistort, fmask, fout, bvals, bval_list, meanDWIDir, preproc_suffix, dwi_shell = bval_list[-1])

        ## Surface-based deformation subcortical segmentation
        #if diffusion_seg:
        #    mbrain_seg.segment(fmask, outputDir, subID, preproc_suffix, shell_suffix, bval_list, cpu_num=proc_num)

        ## Surface-based deformation cortical segmentation
        #if cort_seg:
        #    print("DSurfing")
        #    src_tmp_freesurf_subdir = '/usr/local/freesurfer/subjects/TEMP_CB_BRAIN_CS/'
        #    tmp_freesurf_subdir = '/usr/local/freesurfer/subjects/DSURFER_' + subID + '/'
        #    os.system('cp -r ' + src_tmp_freesurf_subdir + ' ' + tmp_freesurf_subdir)
        #    mbrain_cort.generate_surfaces_from_dwi(outputDir, subID, preproc_suffix, shell_suffix, tmp_freesurf_subdir)
    
    print("Total time for processing: ", time() - total_t_start)
    print("")

    return

if __name__ == "__main__":
   main(sys.argv[1:])