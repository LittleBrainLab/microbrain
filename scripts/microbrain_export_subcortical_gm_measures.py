#!/usr/local/bin/python
import os
import sys
import getopt
from os import path
from shutil import which


import numpy as np
import nibabel as nib


def is_tool(name):
    """
    Check whether `name` is on PATH and marked as executable.

    Parameters
    ----------
    name : str
        The name of the program to check.

    Returns
    -------
    bool
        `True` if `name` is executable, `False` otherwise.
    """
    return which(name) is not None


def fsl_ext():
    """
    Returns the FSL output type extension

    Returns
    -------
    fsl_extension : str
        The FSL output type extension
    """

    fsl_extension = ''
    if os.environ['FSLOUTPUTTYPE'] == 'NIFTI':
        fsl_extension = '.nii'
    elif os.environ['FSLOUTPUTTYPE'] == 'NIFTI_GZ':
        fsl_extension = '.nii.gz'
    return fsl_extension


def extract_FA_MD_from_subcortical_segmentations(subID, segDir, ffa, fmd):
    """
    Extracts FA and MD values from subcortical segmentations created with mbrain-seg

    Parameters
    ----------
    subID : str
        The subject ID
    segDir : str
        The directory containing the subcortical segmentations
    ffa : str
        The FA image file
    fmd : str
        The MD image file

    Returns
    -------
    thisFA : np.array
        The mean FA values for the subcortical segmentations
    thisFAstd : np.array
        The standard deviation of the FA values for the subcortical segmentations
    thisMD : np.array
        The mean MD values for the subcortical segmentations
    thisMDstd : np.array
        The standard deviation of the MD values for the subcortical segmentations
    subcort_label : list
        The subcortical segmentation labels
    """

    md_img = nib.load(fmd)
    md_data = md_img.get_fdata()

    fa_img = nib.load(ffa)
    fa_data = fa_img.get_fdata()

    subcort_label = ['LEFT_THALAMUS', 'RIGHT_THALAMUS', 'LEFT_CAUDATE',
                     'RIGHT_CAUDATE', 'LEFT_PUTAMEN', 'RIGHT_PUTAMEN', 'LEFT_GLOBUS2', 'RIGHT_GLOBUS2']
    thisFA = np.zeros((1, 8))
    thisFAstd = np.zeros((1, 8))
    thisMD = np.zeros((1, 8))
    thisMDstd = np.zeros((1, 8))
    sind = 0
    for slabel, sind in zip(subcort_label, range(0, len(subcort_label))):
        fseg_vol = segDir + subID + '_refined_' + str(slabel) + fsl_ext()
        seg_img = nib.load(fseg_vol)
        seg_data = seg_img.get_fdata()

        thisFA[0, sind] = np.mean(fa_data[seg_data == 1])
        thisFAstd[0, sind] = np.std(fa_data[seg_data == 1])
        thisMD[0, sind] = np.mean(md_data[seg_data == 1])
        thisMDstd[0, sind] = np.std(md_data[seg_data == 1])
        sind = sind + 1

    return thisFA, thisFAstd, thisMD, thisMDstd, subcort_label


def main(argv):
    subList = []
    outFile = ''

    help_string = """usage: microbrain_export_subcortical_gm_measures.py -s <subject_directory_list> -o outputFile
    description: This script extracts FA and MD values from subcortical segmentations created with mbrain-seg

    mandatory arguments:
    -s <directory>,--subList= - specifies the list of directories (e.g. [SubDir1,SubDir2,SubDir3]) for which subcortical diffusion measurements will be taken
    -o <outFile>, --outFile= - file for output
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
            subList = arg.replace('[', '').replace(']', '').split(',')
        elif opt in ("-o", "--outfile"):
            outfile = os.path.normpath(arg)

    sublist_fa = np.zeros((len(subList), 8))
    sublist_fa_std = np.zeros((len(subList), 8))
    sublist_md = np.zeros((len(subList), 8))
    sublist_md_std = np.zeros((len(subList), 8))
    sub_ind = 0

    subID_list = []
    for subDir in subList:
        # Get FA, MD and subcortical GM segmentation directory
        baseDir, subID = os.path.split(os.path.normpath(subDir))
        subID_list += [subID]
        print('Extracting Subject: ' + subID)

        dtiDir = baseDir + '/' + subID + '/DTI_maps/'
        for file in os.listdir(dtiDir):
            if file.endswith('_FA' + fsl_ext()):
                ffa = dtiDir + file
            elif file.endswith('_MD' + fsl_ext()):
                fmd = dtiDir + file

        segDir = baseDir + '/' + subID + '/subcortical_segmentation/'

        FA, FAstd, MD, MDstd, subcort_label = extract_FA_MD_from_subcortical_segmentations(
            subID, segDir, ffa, fmd)

        sublist_fa[sub_ind, :] = FA
        sublist_fa_std[sub_ind, :] = FAstd
        sublist_md[sub_ind, :] = MD
        sublist_md_std[sub_ind, :] = MDstd
        sub_ind = sub_ind + 1

    subList_labels = np.array(subID_list, dtype='|S40')[:, np.newaxis]
    subcort_label = [''] + subcort_label
    subcort_label = str(subcort_label).replace(
        "'", "").replace('[', '').replace(']', '')

    np.savetxt(outfile + '_fa_mean.csv', np.hstack((subList_labels,
               sublist_fa.astype(np.str_))), delimiter=',', fmt='%s', header=subcort_label)
    np.savetxt(outfile + '_fa_std.csv', np.hstack((subList_labels,
               sublist_fa_std.astype(np.str_))), delimiter=',', fmt='%s', header=subcort_label)
    np.savetxt(outfile + '_md_mean.csv', np.hstack((subList_labels,
               sublist_md.astype(np.str_))), delimiter=',', fmt='%s', header=subcort_label)
    np.savetxt(outfile + '_md_std.txt', np.hstack((subList_labels,
               sublist_md_std.astype(np.str_))), delimiter=',', fmt='%s', header=subcort_label)


if __name__ == "__main__":
    main(sys.argv[1:])
