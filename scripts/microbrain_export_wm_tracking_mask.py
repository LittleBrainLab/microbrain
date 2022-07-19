#!/usr/local/bin/python
import os
import sys
sys.path.append('../../MicroBrain/')

import getopt
from os import path
from shutil import which


import numpy as np
import nibabel as nib

from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table

from MicroBrain.utils import surf_util as sutil

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


def export_wm_mask(subID, segDir, surfDir, dtiDir, outFile):

    # Convert LH and RH wm mesh to mask
    fdwi = segDir + '/voxel_output/' + subID + '_refined_LEFT_STRIATUM' + fsl_ext()  
    
    # LH WM to mask
    flh_wmsurf = surfDir + '/' + subID + '_b0b1000_wm_final_lh.vtk'
    flh_out = flh_wmsurf.replace('.vtk', '_mask.nii.gz')
    sutil.surf_to_volume_mask(fdwi, flh_wmsurf, 1, flh_out)

    # RH WM to mask
    frh_wmsurf = surfDir + '/' + subID + '_b0b1000_wm_final_rh.vtk'
    frh_out = frh_wmsurf.replace('.vtk', '_mask.nii.gz')
    sutil.surf_to_volume_mask(fdwi, frh_wmsurf, 1, frh_out)

    # Fill in midline
    lh_wm_img = nib.load(flh_out)
    lh_wm_mask = lh_wm_img.get_data() == 1
    lh_shifted = np.roll(lh_wm_mask, -2, axis=0)

    rh_wm_img = nib.load(frh_out)
    rh_wm_mask = rh_wm_img.get_data() == 1
    rh_shifted = np.roll(rh_wm_mask, 2, axis=0)

    wm_intersect = np.logical_and(lh_shifted, rh_shifted)

    wm_data = np.zeros(lh_wm_mask.shape)
    wm_data[lh_wm_mask] = 1
    wm_data[rh_wm_mask] = 1
    wm_data[wm_intersect] = 1

    exclude_label = ['LEFT_THALAMUS', 'RIGHT_THALAMUS', 
                    'LEFT_STRIATUM', 'RIGHT_STRIATUM',  
                    'LEFT_GLOBUS', 'RIGHT_GLOBUS',
                    'LEFT_HIPPO', 'RIGHT_HIPPO',
                    'LEFT_AMYGDALA', 'RIGHT_AMYGDALA']

    for slabel in exclude_label:
        fseg_vol = segDir + 'voxel_output/' + subID + '_refined_' + str(slabel) + fsl_ext()
        seg_img = nib.load(fseg_vol)
        seg_data = seg_img.get_data()

        wm_data[seg_data == 1] =  0

    # Remove CSF (ventricles) from mask
    fmd = dtiDir + '/' + subID + '_b0b1000_MD.nii.gz'
    md_data = nib.load(fmd).get_data()

    wm_data[md_data > 0.0015] =  0

    nib.save(nib.Nifti1Image(wm_data, lh_wm_img.affine), outFile)

    return

def main(argv):
    subList = []
    outFile = ''

    help_string = """usage: microbrain_export_wm_tracking_mask.py -s <subject_directory> -o outputFile 
    description: microbrain_export_wm_tracking_mask.py outputs  

    mandatory arguments:
    -s <subject directory> - subject directory
    -o <outFile> - file for output 
    """

    try:
        # Note some of these options were left for testing purposes
        opts, args = getopt.getopt(argv, "hs:o:", ["subList=", "outFile="])
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
        elif opt in ("-s", "--sublist"):
            subDir = os.path.normpath(arg)
        elif opt in ("-o", "--outfile"):
            outfile = os.path.normpath(arg)

    
    # Get FA, MD and subcortical GM segmentation directory
    baseDir, subID = os.path.split(os.path.normpath(subDir))

    print('Extracting WM mask for subject: ' + subID)

    segDir = baseDir + '/' + subID + '/subcortical_segmentation/'
    surfDir = baseDir + '/' + subID + '/surf/'
    dtiDir = baseDir + '/' + subID + '/DTI_maps/'
    export_wm_mask(subID, segDir, surfDir, dtiDir, outfile)
    
if __name__ == "__main__":
    main(sys.argv[1:])
