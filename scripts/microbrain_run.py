#!/usr/local/bin/python
from microbrain.preprocessing import mbrain_preproc
from microbrain.modelling import mbrain_modelling
from microbrain.subcort_segmentation import mbrain_segment
from microbrain.surfing import mbrain_cortical_segmentation
import dipy.reconst.dti as dti
from dipy.core.gradients import gradient_table
from dipy.io import read_bvals_bvecs
import nibabel as nib
import numpy as np
import subprocess
from time import time
from os import path
from os import system
import getopt
import os
import sys
import inspect


def fsl_ext():
    fsl_extension = ''
    if os.environ['FSLOUTPUTTYPE'] == 'NIFTI':
        fsl_extension = '.nii'
    elif os.environ['FSLOUTPUTTYPE'] == 'NIFTI_GZ':
        fsl_extension = '.nii.gz'
    return fsl_extension


def get_temp_fs_dir():
    """
    Return temporary freesurf directory in microbrain repository

    Returns
    -------
    tmp_fs_dir: string
        tmp fs path
    """

    import microbrain  # ToDo. Is this the only way?
    module_path = inspect.getfile(microbrain)

    tmp_fs_dir = os.path.dirname(module_path) + "/data/TEMP_FS/"

    return tmp_fs_dir


def main(argv):
    inputDir = ''
    outputDir = ''
    bval_list = []

    # Default values for CL arguments

    # Three masking options
    bet_mask = False
    microbrain_mask = False
    explicit_mask = ''

    bet_fval = 0.3
    bet_gval = 0.0
    rerun_mask = False
    stabilize = False
    dnlsam = False
    dmppca = False
    dmppca_radius = 2
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
    cort_seg = False
    use_tensor_wm = False
    freesurfDir = False
    N4 = True  # By default do N4 correction on DWI image based on the bias field from b0
    proc_num = 0

    help_string = """usage: microbrain.py -s <Subject Directory> -b <bvaluelist> [options]
    description: microbrain is a fully automated diffusion MRI analysis pipeline for measurement of grey matter microstructure.
    Note that options below require the installation of multiple neuroimaging/python analysis software packages (see install instructions).
    VERY BETA Version use with extreme caution. For trouble shooting help contact Graham Little (graham.little.phd@gmail.com)

    mandatory arguments:
    -s <directory>,--subdir= - specifies the directory which microbrain will place all processed files
    -b [bval_list],--bvalues= - specify the bvalues to use for diffusion modelling and segmention (for example [0,1000] to use all b0 and b1000 volumes)

    optional arguments:
    --mask-params=[f-value,g-value], --- sets the bet2 mask parameters default [0.3,0.0]
    --gibbs - perform gibbs ringing correction (Kellner et al., 2016) using included third party c program
    --stabilize - perform stabilization using algorithm provided by non-local spatial angular matching algo (NLSAM, St Jean et al., 2016)
    --dnlsam - perform denoising using non-local spatial angular matching (NLSAM, St Jean et al., 2016)
    --dmppca - performs MPPCA denoising (Veraart et al., 2016) via DIPY
    --eddy --eddy_cuda - performs eddy current correction (Anderson et al., 2016) via FSL v6.0 or higher. CUDA mode runs faster if CUDA setup on NVIDIA GPU

    --pe_direction=phase_encode_direction - direction of diffusion data in orig directory. phase_encode_direction can be LR for left/right or AP for anterior / posterior and their opposite pairs.  Required for distortion correction using a fieldmap or reverse phase encode data
    --AcqReadout= - Time between centre of first echo to centre of last echo as defined by FSL. Only required if using fieldmap spatial distortion correction
    --EffectiveEcho= - Time between centre of first echo to centre of last. Only required if using fieldmap spatial distortion correction

    --no-N4 - does not perform N4 correction (Tustison el al., 2010) via ANTS, useful flag when data is prescan normalized (Siemens)
    --no-json - if no .json image acquistion specification file available will run without this file
    --mbrain-seg - perform surface based subcortical (Little et al., 2021, ISMRM) via MIRTK
    --mbrain-cort - perform surface based cortex segmentation using DTI (Little et al., 2021, NeuroImage) via MIRTK
    --use-tensor-wm - when performing wm/cortex segmentation use the tensor-based force + dwi and stop (i.e. do not refine surface on mean DWI only).  Useful when GM/WM contrast on mean DWI is poor.  Not necessary in high quality data such as HCP diffusion acquisitions.
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
    """

    try:
        # Note some of these options were left for testing purposes
        opts, args = getopt.getopt(argv, "hs:b:i:", ["idcm=", "subdir=", "bvalues=", "bet-mask", "bet-fval=", "bet-gval=", "microbrain-mask", "explicit-mask=", "pe_direction=", "EffectiveEcho=", "AcqReadout=", "rerun-mask",
                                   "dmppca", "dmppca-radius=", "dnlsam", "cpu-num=", "gibbs", "eddy", "eddy_cuda", "no-json", "no-N4", "all", "mbrain-seg", "mbrain-cort", "freesurf-dir=", "hcp", "cb", "allxeddy", "allxn4", "stabilize", "use-tensor-wm"])
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
            bval_list = np.fromstring(arg.strip('[]'), int, sep=',')
            shell_suffix = ''
            for shell in bval_list:
                shell_suffix = shell_suffix + 'b' + str(shell)
        elif opt in ("--bet-mask"):
            bet_mask = True
        elif opt in ("--bet-fval"):
            bet_fval = float(arg)
        elif opt in ("--bet-gval"):
            bet_gval = float(arg)
        elif opt in ("--microbrain-mask"):
            microbrain_mask = True
        elif opt in ("--no-mask"):
            mask = False
        elif opt in ("--explicit-mask"):
            explicit_mask = os.path.normpath(arg)
        elif opt in ("--stabilize"):
            stabilize = True
        elif opt in ("--dmppca"):
            dmppca = True
        elif opt in ("--dmppca-radius"):
            dmppca_radius = int(arg)
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
        elif opt in ("--mbrain-cort"):
            cort_seg = True
        elif opt in ("--use-tensor-wm"):
            use_tensor_wm = True
        elif opt in ("--freesurf-dir"):  # Not available currently
            freesurfDir = os.path.normpath(arg)

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
        process = subprocess.run(
            ['mkdir', preprocDir], stdout=subprocess.PIPE, universal_newlines=True)

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
            print(
                "Error: PE direction argument needs to be specified to use topup distortion correction. Exiting")
            sys.exit()

        topup = True

    fdwi = origDir + subID + fsl_ext()
    fbval = fdwi.replace(fsl_ext(), '.bval')
    fbvec = fdwi.replace(fsl_ext(), '.bvec')
    fjson = fdwi.replace(fsl_ext(), '.json')

    # Denoising/Gibbs ringing correction
    fout = fdwi
    bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
    preproc_suffix = ''

    # Generate a mask of data
    if explicit_mask != '':
        fmask = explicit_mask
    elif bet_mask:
        # Extract b0s
        fout_img = nib.load(fout)
        fout_data = fout_img.get_fdata()

        b0_data = fout_data[:, :, :, np.logical_and(bvals < 50, bvals > -50)]
        firstb0_data = b0_data[:, :, :, 0]
        ffirstb0 = fout.replace(fsl_ext(), '_firstb0' + fsl_ext())
        nib.save(nib.Nifti1Image(firstb0_data, fout_img.affine), ffirstb0)

        fbrain = ffirstb0.replace(fsl_ext(), '_brain' + fsl_ext())
        fmask = ffirstb0.replace(fsl_ext(), '_brain_mask' + fsl_ext())
        process = subprocess.run(['bet', ffirstb0, fbrain, '-m', '-f', str(bet_fval), '-g', str(bet_gval)],
                                 stdout=subprocess.PIPE, universal_newlines=True)

    elif microbrain_mask:
        fmd = fout.replace(fsl_ext(), '_MD' + fsl_ext())
        if not os.path.exists(fmd):
            fout_img = nib.load(fout)
            fout_data = fout_img.get_fdata()

            tolerance = 50
            b_idx = []
            for shell in bval_list:
                b_idx = np.append(b_idx, np.squeeze(np.array(np.where(
                    np.logical_and(bvals < shell + tolerance, bvals > shell - tolerance)))))
            b_idx = b_idx.astype(int)

            print('Modelling Tensor using ' + str(b_idx.shape) + ' volumes:')
            print(b_idx)
            print('Bvals:')
            print(bvals[b_idx])

            sub_bvals = bvals[b_idx]
            sub_bvecs = bvecs[b_idx]
            fout_data = fout_data[:, :, :, b_idx]

            gtab = gradient_table(sub_bvals, sub_bvecs)

            print("Fitting Tensor for Mask")
            tenmodel = dti.TensorModel(gtab)
            tenfit = tenmodel.fit(fout_data)
            nib.save(nib.Nifti1Image(tenfit.md.astype(
                np.float32), fout_img.affine), fmd)
        else:
            print("Already Fit Tensor for Mask: Skipping")

        fmd_brain = fmd.replace(fsl_ext(), '_brain' + fsl_ext())
        fmask = fmd.replace(fsl_ext(), '_brain_mask' + fsl_ext())
        if not os.path.exists(fmask):
            print("Masking MD Map")
            process = subprocess.run(
                ['bet', fmd, fmd_brain, '-m', '-f', '0.3'], stdout=subprocess.PIPE, universal_newlines=True)
        else:
            print("Already Masked MD Map: skipping")

    else:  # Create a mask inclusive of all voxels
        fout_img = nib.load(fout)
        fout_data = fout_img.get_fdata()
        fmask = fout.replace(fsl_ext(), '_fake_mask' + fsl_ext())
        fake_mask_data = np.ones(np.shape(np.squeeze(fout_data[:, :, :, 0])))
        nib.save(nib.Nifti1Image(fake_mask_data, fout_img.affine), fmask)

    # Signal stabilization and/or denoising with NLSAM (automatic estimates for N and sigma via st-jean 2020, Medical Image Analysis)
    if dnlsam and os.path.exists(reverseDir):
        print("Performing NLSAM denoising after topup/eddy correction")
    elif dnlsam:
        fout = mbrain_preproc.denoiseNLSAM(
            fout, fmask, fbval, fbvec, preprocDir, cpu_num=proc_num)
        preproc_suffix = preproc_suffix + '_DNLSAM'
    elif stabilize:
        fout = mbrain_preproc.stabilizeNLSAM(
            fout, fmask, fbval, fbvec, preprocDir)
        preproc_suffix = preproc_suffix + '_STAB'

    # Denosing using MPPCA
    if dmppca:
        print(dmppca_radius)
        fout = mbrain_preproc.denoiseMPPCA(
            fout, fbval, fbvec, preprocDir, patch_radius=dmppca_radius)
        preproc_suffix = preproc_suffix + '_DMPPCA'

    if gibbs:
        fout, stdout, returncode = mbrain_preproc.gibbsRingingCorrection(
            fout, preprocDir)
        if returncode != 0:
            print(
                'microbrain: unring.a64 returned an error, make sure it is installed correctly')
            sys.exit()

        preproc_suffix = preproc_suffix + '_GR'

    # Run FSL Topup if required
    ftopup = ''
    if topup:
        fpe_ind = fdwi.replace(fsl_ext(), '_PEIND.txt')
        pe_direction_ind = np.loadtxt(fpe_ind, delimiter=' ')
        b0_pe_direction = pe_direction_ind[np.logical_and(
            bvals >= -20, bvals <= 20)]

        facq = preprocDir + 'acqparam.txt'
        acq = np.zeros((b0_pe_direction.shape[0], 4))
        # Note this number will make the off-resonance field not scaled properly
        acq[:, 3] = 0.05

        if pe_direction == 'LR':
            acq[:, 0] = b0_pe_direction
        if pe_direction == 'RL':
            acq[:, 0] = b0_pe_direction * -1
        elif pe_direction == 'AP':
            acq[:, 1] = b0_pe_direction
        elif pe_direction == 'PA':
            acq[:, 1] = b0_pe_direction * -1

        np.savetxt(facq, acq, delimiter=' ', fmt='%.2f')

        findex = preprocDir + 'index.txt'
        index = np.ones(bvals.shape)
        first_reverseb0 = np.where(b0_pe_direction == 1)[0]
        index[pe_direction_ind == 1] = first_reverseb0[0] + 1
        np.savetxt(findex, index, delimiter='', newline=' ', fmt='%01d')

        # output b0s
        fb0s = fout.replace(fsl_ext(), '_b0vols' + fsl_ext())
        dwi_img = nib.load(fout)
        dwi_vol = dwi_img.get_fdata()
        b0_vols = dwi_vol[:, :, :, np.logical_and(
            bvals >= -20, bvals <= 20)]
        nib.save(nib.Nifti1Image(b0_vols, dwi_img.affine), fb0s)

        ftopup, ftopup_unwarped, stdout, returncode = mbrain_preproc.fslv6p0_topup(
            fb0s, facq)
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
        np.savetxt(facq, acq[None, :], fmt=('%01d', '%01d', '%01d', '%.6f'))

    # Eddy current correction
    if eddy:
        if not json:
            fjson = False

        # If using fieldmap register/interpolate it to the firstB0 image, this is needed prior to input into eddy
        ffieldmap_hz_reg2firstb0 = ''
        if fieldmap:
            ffirstb0_brain = ffirstb0.replace(fsl_ext(), '_brain' + fsl_ext())
            process = subprocess.run(['bet', ffirstb0, ffirstb0_brain, '-m', '-f', str(
                bet_fval), '-g', str(bet_gval)], stdout=subprocess.PIPE, universal_newlines=True)

            if pe_direction == 'AP':
                unwarp_direction = 'y-'
            elif pe_direction == 'PA':
                unwarp_direction = 'y'
            elif pe_direction == 'LR':
                unwarp_direction = 'x'  # TODO Test this
            elif pe_direction == 'RL':
                unwarp_direction = 'x-'  # TODO Test this

            ffieldmap_mag_warped = ffieldmap_mag.replace(
                fsl_ext(), '_warped' + fsl_ext())
            process = subprocess.run(['fugue', '-v', '-i', ffieldmap_mag, '--unwarpdir=' + unwarp_direction, '--dwell=' + str(
                effective_echo), '--loadfmap=' + ffieldmap_rad, '-w',  ffieldmap_mag_warped], stdout=subprocess.PIPE, universal_newlines=True)

            ffieldmap_mag_warped_2firstb0 = ffieldmap_mag_warped.replace(
                fsl_ext(), '_2firstb0' + fsl_ext())
            ffieldmap_2firstb0_mat = fieldmapDir + subID + '_fieldmap2firstb0.mat'
            process = subprocess.run(['flirt', '-in', ffieldmap_mag_warped, '-ref',  ffirstb0_brain, '-out', ffieldmap_mag_warped_2firstb0,
                                     '-omat', ffieldmap_2firstb0_mat, '-dof', '6'], stdout=subprocess.PIPE, universal_newlines=True)

            ffieldmap_hz_reg2firstb0 = ffieldmap_hz.replace(
                fsl_ext(), '_reg2firstb0')
            process = subprocess.run(['flirt', '-in', ffieldmap_hz, '-ref', ffirstb0_brain, '-applyxfm', '-init', ffieldmap_2firstb0_mat,
                                     '-out', ffieldmap_hz_reg2firstb0, '-interp', 'spline'], stdout=subprocess.PIPE, universal_newlines=True)

            # Use giant fieldmap mask for input into eddy
            ffieldmap_mask = ffieldmap_mag.replace(
                'brain1' + fsl_ext(), 'brain_mask' + fsl_ext())
            fmask_field = ffirstb0.replace(
                fsl_ext(), '_mask_fieldmap' + fsl_ext())
            process = subprocess.run(['flirt', '-in', ffieldmap_mask, '-ref', ffirstb0_brain, '-applyxfm', '-init', ffieldmap_2firstb0_mat,
                                     '-out', fmask_field, '-interp', 'nearestneighbour'], stdout=subprocess.PIPE, universal_newlines=True)
            fmask_eddy = fmask_field

            # Fieldmap lies outside the b0 mask so dilate it
            # fmask_dil = fmask.replace(fsl_ext(), '_dil' + fsl_ext())
            # process = subprocess.run(['fslmaths', fmask, '-kernel', 'box', '15', '-dilF', fmask_dil], stdout=subprocess.PIPE, universal_newlines=True)
            # fmask = fmask_dil
        else:
            fmask_eddy = fmask

        # Make sure mask has same dimensions as first b0 volume (sometimes prerproc steps mess this up)
        ffirstVol = fout.replace(fsl_ext(), '_firstVol' + fsl_ext())
        process = subprocess.run(
            ['fslroi', fout, ffirstVol, '0', '1'], stdout=subprocess.PIPE, universal_newlines=True)
        process = subprocess.run(
            ['fslcpgeom', ffirstVol, fmask_eddy], stdout=subprocess.PIPE, universal_newlines=True)

        fout, fbvec_rotated, stdout, returncode = mbrain_preproc.fslv6p0_eddy(
            fout, facq, findex, fmask_eddy, fbval, fbvec, fjson, ftopup, ffieldmap_hz_reg2firstb0, cuda, preprocDir, preprocDir + 'eddy_output_files/')
        if returncode != 0:
            print(
                "microbrain: FSL's eddy returned an error, make sure it is installed correctly")
            sys.exit()

        # update bvecs with rotated version from eddy correction
        bvals, bvecs = read_bvals_bvecs(fbval, fbvec_rotated)

        preproc_suffix = preproc_suffix + '_EDDY'

    # If reverse phase encode detected perform denoising after eddy
    if dnlsam and os.path.exists(reverseDir):
        fout = mbrain_preproc.denoiseNLSAM(
            fout, fmask, fbval, fbvec, preprocDir, cpu_num=proc_num)
        preproc_suffix = preproc_suffix + '_DNLSAM'

    # Remove preceding _
    if preproc_suffix != '':
        preproc_suffix = preproc_suffix[1:]
        suffix = preproc_suffix + '_' + shell_suffix
    else:
        preproc_suffix = ''
        suffix = '_' + shell_suffix

    if dti_model:
        # Output Tensor and associated diffusion parametric maps
        ffa, fmd, ffa_rgb, fad, frd, fevec, fpevec, ftensor = mbrain_modelling.output_DTI_maps_multishell(
            fout, fmask, bvals, bvecs, tensorDir, shells=bval_list)

        # Output first b0 for N4 correction
        ffirstb0_undistort = fout.replace(fsl_ext(), '_firstb0' + fsl_ext())
        if not os.path.exists(ffirstb0_undistort):
            fout_img = nib.load(fout)
            fout_data = fout_img.get_fdata()

            b0_data = fout_data[:, :, :, np.logical_and(
                bvals >= -20, bvals <= 20)]

            firstb0_data = b0_data[:, :, :, 0]

            nib.save(nib.Nifti1Image(firstb0_data,
                                     fout_img.affine), ffirstb0_undistort)

            fout_img.uncache()
            del fout_data

        # Output Average DWI maps for each shell, as well as N4 corrected versions. Note, N4 correction applied to mean DWI not to the raw data itself)
        if N4:
            fb0, fb0_n4, fdwi, fdwi_n4, stdout, returncode = mbrain_preproc.output_DWI_maps(
                ffirstb0_undistort, fmask, fout, bvals, bval_list, meanDWIDir, preproc_suffix, dwi_shell=bval_list[-1])
            if returncode != 0:
                print(
                    "microbrain: N4 returned an error, make sure it is installed correctly")
                sys.exit()
        else:
            fb0, fb0_n4, fdwi, fdwi_n4 = mbrain_preproc.output_DWI_maps_noN4(
                ffirstb0_undistort, fmask, fout, bvals, bval_list, meanDWIDir, preproc_suffix, dwi_shell=bval_list[-1])

        # Surface-based deformation subcortical segmentation
        if diffusion_seg:
            meshDir, voxelDir = mbrain_segment.segment(outputDir, subID, preproc_suffix,
                                                       shell_suffix, cpu_num=proc_num)

            # Surface-based deformation cortical segmentation
            if cort_seg:
                src_tmp_freesurf_subdir = get_temp_fs_dir()
                tmp_freesurf_subdir = os.environ['SUBJECTS_DIR'] + \
                    '/MBRAIN_' + subID + '/'
                os.system('cp -r ' + src_tmp_freesurf_subdir +
                          ' ' + tmp_freesurf_subdir)

                print("Source: " + src_tmp_freesurf_subdir)
                print("TempFS: " + tmp_freesurf_subdir)

                if use_tensor_wm:
                    mbrain_cortical_segmentation.generate_surfaces_from_dwi(
                        fmask, voxelDir, outputDir, subID, preproc_suffix, shell_suffix, tmp_freesurf_subdir, cpu_num=proc_num, use_tensor_wm=True)
                else:
                    mbrain_cortical_segmentation.generate_surfaces_from_dwi(
                        fmask, voxelDir, outputDir, subID, preproc_suffix, shell_suffix, tmp_freesurf_subdir, cpu_num=proc_num)
                os.system('rm -r ' + tmp_freesurf_subdir)

    print("Total time for processing: ", time() - total_t_start)
    print("")

    return


if __name__ == "__main__":
    main(sys.argv[1:])
