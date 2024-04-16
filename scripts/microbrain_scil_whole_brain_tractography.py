#!/usr/local/bin/python
from scipy.ndimage import binary_fill_holes
from microbrain.utils import surf_util as sutil
from dipy.core.gradients import gradient_table
from dipy.io import read_bvals_bvecs
import nibabel as nib
import numpy as np
from shutil import which
from os import path
import getopt
import os
import sys


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


def run_wb_tractography(subDir, subID, outDir):

    mask_dir = outDir + '/tracking_mask'
    if not os.path.exists(mask_dir):
        os.makedirs(mask_dir)

    # Create tracking mask from mbrain segmentation
    fmask_tracking_float = mask_dir + '/whole_brain_tracking_mask.nii.gz'
    if not os.path.exists(fmask_tracking_float):
        os.system('microbrain_export_wm_tracking_mask.py -s ' +
                  subDir + ' -o ' + fmask_tracking_float)

    # Convert mask to int
    fmask_tracking = fmask_tracking_float.replace('.nii.gz', '_int.nii.gz')
    if not os.path.exists(fmask_tracking):
        os.system('fslmaths ' + fmask_tracking_float +
                  ' ' + fmask_tracking + ' -odt int')

    # Generate fiber response function
    # Note these definitions only work for HCP data right now
    fdwi = subDir + '/orig/' + subID + '.nii.gz'
    fbval = subDir + '/orig/' + subID + '.bval'
    fbvec = subDir + '/orig/' + subID + '.bvec'

    fodf_dir = outDir + '/fodf'
    if not os.path.exists(fodf_dir):
        os.makedirs(fodf_dir)

    fresponse = fodf_dir + '/' + subID + '_frf.txt'
    if not os.path.exists(fresponse):
        print('Generating fiber response function')
        os.system('scil_compute_ssst_frf.py ' + fdwi + ' ' + fbval +
                  ' ' + fbvec + ' ' + fresponse + ' --mask ' + fmask_tracking)

    # convert brain mask to int
    fmask_brain_float = subDir + '/orig/' + subID + '_mask.nii.gz'
    fmask_brain = subDir + '/orig/' + subID + '_mask_int.nii.gz'
    if not os.path.exists(fmask_brain):
        os.system('fslmaths ' + fmask_brain_float +
                  ' ' + fmask_brain + ' -odt int')

    ffodf = fodf_dir + '/' + subID + '_fodf.nii.gz'
    if not os.path.exists(ffodf):
        print('Generating fodf')
        os.system('scil_compute_ssst_fodf.py ' + fdwi + ' ' + fbval + ' ' +
                  fbvec + ' ' + fresponse + ' ' + ffodf + ' --mask ' + fmask_brain)

    # Generate whole brain tractography
    tractogram_dir = outDir + '/tractogram'
    if not os.path.exists(tractogram_dir):
        os.makedirs(tractogram_dir)

    fwb_tract = tractogram_dir + '/' + subID + '_wb_tractogram.trk'
    if not os.path.exists(fwb_tract):
        print('Generating whole brain tractography')
        os.system('scil_compute_local_tracking_dev.py ' + ffodf + ' ' + fmask_tracking + ' ' +
                  fmask_tracking + ' ' + fwb_tract + ' --nt 1000000 --save_seeds --min_length 0 -v --sfthres_init 0.1')

    return


def main(argv):
    subList = []
    outDir = ''

    help_string = """usage: microbrain_export_wm_tracking_mask.py -s <subject_directory> -o <output_directory>
    description: microbrain_scil_whole_brain_tractography.py outputs

    mandatory arguments:
    -s, --subDir <subject directory> - microbrain subject output directory

    optional arguments:
    -o, --outDir <output_directory> - directory for output if doesn't exist will make it (default: <subject_directory>/tracking))
    """

    try:
        # Note some of these options were left for testing purposes
        opts, args = getopt.getopt(argv, "hs:o", ["subDir", "outDir"])
    except getopt.GetoptError:
        print(help_string)
        sys.exit(2)

    if len(opts) == 0:
        print(help_string)
        sys.exit(2)

    outDir = ''
    for opt, arg in opts:
        if opt == '-h':
            print(help_string)
            sys.exit()
        elif opt in ("-s", "--subDir"):
            subDir = os.path.normpath(arg)
        elif opt in ("-o", "--outDir"):
            outDir = os.path.normpath(arg)

    # Get FA, MD and subcortical GM segmentation directory
    baseDir, subID = os.path.split(os.path.normpath(subDir))

    if outDir == '':
        outDir = baseDir + '/' + subID + '/tracking'

    print('Running whole brain tractography using scilpy tools: ' + subID)

    # Create output directory if it doesn't exist
    if not os.path.exists(outDir):
        os.makedirs(outDir)

    run_wb_tractography(subDir, subID, outDir)


if __name__ == "__main__":
    main(sys.argv[1:])
