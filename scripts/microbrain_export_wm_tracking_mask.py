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
import glob
sys.path.append('../../microbrain/')


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


def export_wm_mask(subID, segDir, surfDir, dtiDir, regDir, outFile):

    # Convert LH and RH wm mesh to mask
    fdwi = segDir + '/voxel_output/' + subID + '_refined_LEFT_STRIATUM' + fsl_ext()

    # LH WM to mask
    flh_wmsurf = glob.glob(
        surfDir + '/mesh_segmentation/' + subID + '*_wm_final_lh.vtk')[0]
    print(flh_wmsurf)
    flh_out = flh_wmsurf.replace('.vtk', '_mask.nii.gz')
    sutil.surf_to_volume_mask(fdwi, flh_wmsurf, 1, flh_out)

    # RH WM to mask
    frh_wmsurf = glob.glob(
        surfDir + '/mesh_segmentation/' + subID + '*_wm_final_rh.vtk')[0]
    frh_out = frh_wmsurf.replace('.vtk', '_mask.nii.gz')
    sutil.surf_to_volume_mask(fdwi, frh_wmsurf, 1, frh_out)

    # Fill in midline
    lh_wm_img = nib.load(flh_out)
    lh_wm_mask = lh_wm_img.get_fdata() == 1
    lh_shifted = np.roll(lh_wm_mask, -2, axis=0)

    rh_wm_img = nib.load(frh_out)
    rh_wm_mask = rh_wm_img.get_fdata() == 1
    rh_shifted = np.roll(rh_wm_mask, 2, axis=0)

    wm_intersect = np.logical_and(lh_shifted, rh_shifted)

    wm_data = np.zeros(lh_wm_mask.shape)
    wm_data[lh_wm_mask] = 1
    wm_data[rh_wm_mask] = 1
    wm_data[wm_intersect] = 1

    # Include brainstem WM and cerrebellum WM
    ftissue = surfDir + '/tissue_classification/tissue_prob_' + subID + '_out_seg.nii.gz'
    tissue_seg = nib.load(ftissue).get_fdata()
    fharvard_reg = regDir + '/HarvardOxford-sub-prob-1mm_ANTsReg_native.nii.gz'
    harvard_prob = nib.load(fharvard_reg).get_fdata()
    fmni_reg = regDir + '/MNI-prob-1mm_ANTsReg_native.nii.gz'
    mni_prob = nib.load(fmni_reg).get_fdata()

    stem_ind = 7
    cb_ind = 1
    cb_stem_mask = np.logical_or(
        harvard_prob[:, :, :, stem_ind] > 0, mni_prob[:, :, :, cb_ind] > 0)
    wm_cb_stem_mask = np.logical_and(tissue_seg == 2, cb_stem_mask)
    wm_data[binary_fill_holes(wm_cb_stem_mask)] = 1

    # Exclude subcortical GM
    exclude_label = ['LEFT_THALAMUS', 'RIGHT_THALAMUS',
                     'LEFT_STRIATUM', 'RIGHT_STRIATUM',
                     'LEFT_GLOBUS', 'RIGHT_GLOBUS',
                     'LEFT_HIPPO', 'RIGHT_HIPPO',
                     'LEFT_AMYGDALA', 'RIGHT_AMYGDALA']

    for slabel in exclude_label:
        fseg_vol = segDir + 'voxel_output/' + subID + \
            '_refined_' + str(slabel) + fsl_ext()
        seg_img = nib.load(fseg_vol)
        seg_data = seg_img.get_fdata()

        wm_data[seg_data == 1] = 0

    # Remove CSF (ventricles) from mask
    fmd = glob.glob(dtiDir + '/' + subID + '*_MD.nii.gz')[0]
    md_data = nib.load(fmd).get_fdata()

    wm_data[md_data > 0.0015] = 0

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
    regDir = baseDir + '/' + subID + '/registration/'
    export_wm_mask(subID, segDir, surfDir, dtiDir, regDir, outfile)


if __name__ == "__main__":
    main(sys.argv[1:])
