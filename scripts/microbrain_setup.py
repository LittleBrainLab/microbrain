#!/usr/local/bin/python
import os
import sys
import getopt
from os import system
from os import path
from time import time
import subprocess
from shutil import which


import numpy as np
import nibabel as nib

from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table


def is_tool(name):
    """
    Check to see if program is installed to path and executable before running subprocess

    Parameters
    ----------
    name : str
        Name of program to check if installed

    Returns
    -------
    bool
        True if program is installed, False otherwise"""
    return which(name) is not None


def fsl_ext():
    """
    Check to see if FSL is outputting nifti or nifti_gz files

    Returns
    -------
    str : fsl_extension
        Returns the FSL output extension for nifti or nifti_gz files
    """

    fsl_extension = ''
    if os.environ['FSLOUTPUTTYPE'] == 'NIFTI':
        fsl_extension = '.nii'
    elif os.environ['FSLOUTPUTTYPE'] == 'NIFTI_GZ':
        fsl_extension = '.nii.gz'
    return fsl_extension


def dcm2nii(dataDir, outDir, rawDir, thisSub, mag_image=False, phase_image=False):
    """
    Convert dicom files to nifti files using dcm2niix

    Parameters
    ----------
    dataDir : str
        Directory with dicom files to convert
    outDir : str
        Output directory for converted nifti files
    rawDir : str
        Directory to place converted nifti files
    thisSub : str
        Subject ID
    mag_image : bool
        Flag to indicate if magnitude image
    phase_image : bool
        Flag to indicate if phase image

    Returns
    -------
    niiOutFile : str
        Path to nifti file
    bvalOut : str
        Path to bval file
    bvecOut : str
        Path to bvec file
    niiOutJson : str
        Path to json file
    stdout : str
        Standard output from dcm2niix
    return_code : int
        Return code from dcm2niix
    """

    subDir = outDir + '/' + thisSub + '/'
    if not os.path.exists(subDir):
        subprocess.run(['mkdir', subDir], stdout=subprocess.PIPE,
                       universal_newlines=True)

    convertDir = subDir + rawDir + '/'
    if not os.path.exists(convertDir):
        subprocess.run(['mkdir', convertDir],
                       stdout=subprocess.PIPE, universal_newlines=True)

    niiOutFile = convertDir + thisSub + fsl_ext()

    if mag_image:
        niiOutFile = convertDir + thisSub + '_e1' + fsl_ext()

    if phase_image:
        niiOutFile = convertDir + thisSub + '_e2_ph' + fsl_ext()

    niiOutJson = convertDir + thisSub + '.json'
    bvalOut = convertDir + thisSub + '.bval'
    bvecOut = convertDir + thisSub + '.bvec'
    if not os.path.exists(niiOutFile):
        print("Converting Subject: " + thisSub)

        if os.path.exists(dataDir):
            if is_tool('dcm2niix'):
                dcm2nii_cmd = ['dcm2niix',
                               '-f', thisSub,
                               '-o', convertDir]
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
                print(
                    "microbrain Setup: Could not find dcm2niix, make sure it is installed to your path")
                stdout = ''
                return_code = 1
        else:
            print("microbrain Setup: Could not find srcDir: " + dataDir)
            stdout = ''
            return_code = 1
    else:
        print("microbrain Setup: Dicom already converted skipping subject: " + thisSub)
        stdout = ''
        return_code = 0

    return niiOutFile, bvalOut, bvecOut, niiOutJson, stdout, return_code


def main(argv):
    inputDir = ''
    outputDir = ''

    topup = False
    fieldmap = False
    smooth_radius = 0
    input_fieldmap = ''
    input_reverse_pe = ''
    make_bfiles = False

    help_string = """usage: microbrain_setup.py -s <Subject Directory> [options]
    description: microbrain_setup.py creates the subject data directory for downstream processing from a microbrain_run.py.
    Three options exist for setting up the subject folder
    1) diffusion data no spatial distortion correction
    2) diffusion data spatial distortion correction with an additional dataset acquired with reverse phase encode
    3) diffusion data spatial distortion correction with a field map
    Note that options below require the installation of multiple neuroimaging/python analysis software packages.
    VERY BETA version use with extreme caution. For trouble shooting help contact Graham Little (grahamn.little.phd@gmail.com)

    mandatory arguments:
    -s <directory>,--subdir= - specifies the directory which microbrain will place all processed files

    -i <dicom_directory>,--idcm=dicom_directory - uses dcm2niix to convert dicom files to NIFTI which are then placed in a newly created subject directory in the "orig" folder

    optional arguments:
    --idcm_reversePE=dicom_directory - uses dcm2niix to convert reverse phase encode dicom to NIFTI, stores in "orig_reverse" folder

    --idcm_fieldmap=[dicom_magnitude, dicom_phase, TE_difference] - uses dcm2niix to convert reverse phase encode dicom to NIFTI, calculates fieldmap (in hz) from magnitude/phase images and stores in "orig_fieldmap" folder. TE_difference is the difference in TE between first and second images, in default siemens sequence this value is usually 2.46 ms

    --make_bfiles - if bval/bvec files are not generated in conversion step, assume only b0s and make bval/bvec files

    Examples Different Distortion Correction:

    1) No spatial distortion correction, b1000
    microbrain_setup.py -s path_to_subject_directory -i path_to_dicom_directory
    microbrain_run.py -s path_to_subject_directory -b [0,1000] --gibbs --eddy

    2) Spatial Distortion Correction with reverse phase encode
    microbrain_setup.py -s path_to_subject_directory -i path_to_dicom_directory --idcm_reversePE=path_to_dicom_directory_with_reverse_PE_data
    microbrain_run.py -s path_to_subject_directory -b [0,1000] --pe_direction=AP --gibbs --eddy

    3) Spatial Distortion Correction with siemens fieldmap data
    microbrain_setup.py -s path_to_subject_directory -i path_to_dicom_directory --idcm_fieldmap=[path_to_dicom_directory_with_magnitude_data, path_to_dicom_directory_with_phase_data, 2.46]
    microbrain_run.py -s path_to_subject_directory -b [0,1000] --pe_direction=AP --AcqReadout=0.04999 --EffectiveEcho=0.00034 --gibbs --eddy
    """

    try:
        # Note some of these options were left for testing purposes
        opts, args = getopt.getopt(argv, "hs:i:", [
                                   "idcm=", "idcm_reversePE=", "idcm_fieldmap=", "idcm_fieldmap_smooth=", "make_bfiles"])
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
        elif opt in ("--idcm_reversePE"):
            topup = True
            inputDir_reversePE = os.path.normpath(arg)
        elif opt in ("--idcm_fieldmap"):
            fieldmap = True
            fieldmap_list = arg.strip('[]').split(',')
            fieldmapDir_mag = fieldmap_list[0]
            fieldmapDir_phase = fieldmap_list[1]
            fieldmap_echo_diff = fieldmap_list[2]
        elif opt in ("--idcm_fieldmap_smooth"):
            smooth_radius = int(arg)
        elif opt in ("--make_bfiles"):
            make_bfiles = True

    outputDir, subID = os.path.split(outputDir)
    print('Setting Up:' + subID)
    print('OutDir: ' + outputDir)

    # Make subject directory converted dicoms
    subDir = outputDir + '/' + subID + '/'
    if not os.path.exists(subDir):
        process = subprocess.run(
            ['mkdir', subDir], stdout=subprocess.PIPE, universal_newlines=True)

    origDir = outputDir + '/' + subID + '/orig/'
    if not os.path.exists(origDir):
        process = subprocess.run(
            ['mkdir', origDir], stdout=subprocess.PIPE, universal_newlines=True)

    # if reverse phase encode data is acquired convert that to nifti and merge to input file else just convert the file and move on
    if topup:
        # Given an input folder, convert to NIFTI and output into appropriate folder.
        # Function will skip this step if orig folder already exists with converted NIFTI File
        [fdwi, fbval, fbvec, fjson, stdout, returncode] = dcm2nii(
            inputDir, outputDir, 'orig_normalPE/', subID)
        if returncode != 0:
            print('microbrain: dcm2niix returned an error, make sure it is installed correctly and that dicom files exist')
            sys.exit()

        [fdwi_reversePE, fbval_reversePE, fbvec_reversePE, fjson_reversePE, stdout,
            returncode] = dcm2nii(inputDir_reversePE, outputDir, 'orig_reversePE/', subID)
        if returncode != 0:
            print('microbrain: dcm2niix returned an error, make sure it is installed correctly and that dicom files exist')
            sys.exit()
        origDir_reversePE = outputDir + '/' + subID + '/orig_reversePE/'

        # generate reverse PE bval/bvec files if only b0s
        if make_bfiles:
            reversePE_data = nib.load(fdwi_reversePE).get_fdata()
            if not os.path.exists(fbval_reversePE):
                print(
                    'microbrain: no bval file found, assuming reverse PE file are only b0 volumes')
                if len(reversePE_data.shape) == 3:
                    reversePE_bvals = np.zeros((1, 1))
                else:
                    reversePE_bvals = np.zeros((1, reversePE_data.shape[3]))
                np.savetxt(fbval_reversePE, reversePE_bvals, fmt="%d")

            if make_bfiles and not os.path.exists(fbvec_reversePE):
                print(
                    'microbrain: no bval file found, assuming reverse PE file are only b0 volumes')

                if len(reversePE_data.shape) == 3:
                    reversePE_bvec = np.zeros((3, 1))
                else:
                    reversePE_bvec = np.zeros((3, reversePE_data.shape[3]))
                np.savetxt(fbvec_reversePE, reversePE_bvec, fmt="%d")

        # Merge nifti
        fdwi_merge = outputDir + '/' + subID + '/orig/' + subID + fsl_ext()
        process = subprocess.run(['fslmerge', '-t', fdwi_merge, fdwi,
                                 fdwi_reversePE], stdout=subprocess.PIPE, universal_newlines=True)

        # Merge bvals / bvecs
        fbval_merge = fdwi_merge.replace(fsl_ext(), '.bval')
        fbvec_merge = fdwi_merge.replace(fsl_ext(), '.bvec')
        fjson_merge = fdwi_merge.replace(fsl_ext(), '.json')
        fpe_ind = fdwi_merge.replace(fsl_ext(), '_PEIND.txt')

        bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
        bvals_reversePE, bvecs_reversePE = read_bvals_bvecs(
            fbval_reversePE, fbvec_reversePE)

        bvals_merge = np.concatenate((bvals, bvals_reversePE))
        bvecs_merge = np.concatenate((bvecs, bvecs_reversePE)).T
        pe_direction_ind = np.ones(bvals_merge.shape)*-1
        pe_direction_ind[bvals.shape[0]:] = 1

        np.savetxt(
            fbval_merge, bvals_merge[None, :], delimiter=' ', fmt='%01d')
        np.savetxt(fbvec_merge, bvecs_merge, delimiter=' ', fmt='%05f')
        np.savetxt(
            fpe_ind, pe_direction_ind[None, :], delimiter=' ', fmt='%01d')
        if os.path.exists(fjson):
            process = subprocess.run(
                ['cp', fjson, fjson_merge], stdout=subprocess.PIPE, universal_newlines=True)

    elif fieldmap:
        # Given an input folder, convert to NIFTI and output into appropriate folder.
        # Function will skip this step if orig folder already exists with converted NIFTI File
        [fdwi, fbval, fbvec, fjson, stdout, returncode] = dcm2nii(
            inputDir, outputDir, 'orig/', subID)
        if returncode != 0:
            print('microbrain: dcm2niix returned an error, make sure it is installed correctly and that dicom files exist')
            sys.exit()

        # Convert magnitude image
        [ffieldmap_mag1, fnotbval, fnotbvec, ffieldmap_mag_json, stdout, returncode] = dcm2nii(
            fieldmapDir_mag, outputDir, 'orig_fieldmap/', subID, True, False)
        if returncode != 0:
            print('microbrain: dcm2niix returned an error, make sure it is installed correctly and that dicom files exist')
            sys.exit()

        # Convert phase image
        [ffieldmap_phase, fnotbval, fnotbvec, ffieldmap_phase_json, stdout, returncode] = dcm2nii(
            fieldmapDir_phase, outputDir, 'orig_fieldmap/', subID, False, True)
        if returncode != 0:
            print('microbrain: dcm2niix returned an error, make sure it is installed correctly and that dicom files exist')
            sys.exit()

        # After converting Siemens fieldmap, prepare it for use with microbrain Pipeline (i.e. calculate fieldmap from magnitude/phase images and convert to hz)

        # Average Magnitude images
        ffieldmap_mag2 = ffieldmap_mag1.replace(
            'e1' + fsl_ext(), 'e2' + fsl_ext())
        ffieldmap_mag = ffieldmap_mag1.replace(
            'e1' + fsl_ext(), 'emean' + fsl_ext())
        process = subprocess.run(['fslmaths', ffieldmap_mag1, '-add', ffieldmap_mag2,
                                 '-div', '2', ffieldmap_mag], stdout=subprocess.PIPE, universal_newlines=True)

        ffieldmap_mag_brain1 = ffieldmap_mag.replace(fsl_ext(), '_brain1')
        process = subprocess.run(['bet', ffieldmap_mag, ffieldmap_mag_brain1,
                                 '-m', '-B'], stdout=subprocess.PIPE, universal_newlines=True)
        # ffieldmap_mag_brain = ffieldmap_mag_brain + fsl_ext()

        ffieldmap_mag_brain1_mask = ffieldmap_mag_brain1 + '_mask' + fsl_ext()
        ffieldmap_mag_brain_mask = ffieldmap_mag_brain1_mask.replace(
            'brain1_mask' + fsl_ext(), 'brain_mask' + fsl_ext())
        process = subprocess.run(['fslmaths', ffieldmap_mag_brain1_mask, '-kernel', 'box', '20',
                                 '-dilF', ffieldmap_mag_brain_mask], stdout=subprocess.PIPE, universal_newlines=True)
        # process = subprocess.run(['fslmaths',ffieldmap_mag_brain1_mask,'-ero',ffieldmap_mag_brain_mask], stdout=subprocess.PIPE, universal_newlines=True)

        ffieldmap_mag_brain = ffieldmap_mag_brain_mask.replace(
            '_mask' + fsl_ext(), fsl_ext())
        process = subprocess.run(['fslmaths', ffieldmap_mag, '-mul', ffieldmap_mag_brain_mask,
                                 ffieldmap_mag_brain], stdout=subprocess.PIPE, universal_newlines=True)

        ffieldmap_rad = ffieldmap_mag.replace(
            'emean' + fsl_ext(), 'fieldmap_rad' + fsl_ext())
        process = subprocess.run(['fsl_prepare_fieldmap', 'SIEMENS', ffieldmap_phase, ffieldmap_mag_brain,
                                 ffieldmap_rad, fieldmap_echo_diff], stdout=subprocess.PIPE, universal_newlines=True)

        if smooth_radius != 0:
            process = subprocess.run(['fugue', '--loadfmap=' + ffieldmap_rad, '-s', str(
                smooth_radius), '--savefmap=' + ffieldmap_rad], stdout=subprocess.PIPE, universal_newlines=True)

        ffieldmap_hz = ffieldmap_rad.replace(
            'rad' + fsl_ext(), 'hz' + fsl_ext())
        process = subprocess.run(['fslmaths', ffieldmap_rad, '-div', '6.28318531',
                                 ffieldmap_hz], stdout=subprocess.PIPE, universal_newlines=True)
    else:
        # Given an input folder, convert to NIFTI and output into appropriate folder.
        # Function will skip this step if orig folder already exists with converted NIFTI File
        [fdwi, fbval, fbvec, fjson, stdout, returncode] = dcm2nii(
            inputDir, outputDir, 'orig/', subID)
        if returncode != 0:
            print('microbrain: dcm2niix returned an error, make sure it is installed correctly and that dicom files exist')
            sys.exit()


if __name__ == "__main__":
    main(sys.argv[1:])
