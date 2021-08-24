#!/usr/local/bin/python
import os,sys,getopt
import shutil
import glob
from os import system
from os import path
from time import time
import subprocess
from shutil import which



import numpy as np
import nibabel as nib

from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table

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

def extract_FA_MD_from_subcortical_segmentations(subID, segDir, ffa, fmd):
    md_img = nib.load(fmd)
    md_data = md_img.get_data()

    fa_img = nib.load(ffa)
    fa_data = fa_img.get_data()

    subcort_label = ['LEFT_THALAMUS','RIGHT_THALAMUS','LEFT_CAUDATE','RIGHT_CAUDATE','LEFT_PUTAMEN','RIGHT_PUTAMEN','LEFT_GLOBUS2','RIGHT_GLOBUS2']
    thisFA = np.zeros((1,8))
    thisFAstd = np.zeros((1,8))
    thisMD = np.zeros((1,8))
    thisMDstd = np.zeros((1,8))
    sind = 0
    for slabel,sind in zip(subcort_label,range(0,len(subcort_label))):
        fseg_vol = segDir + subID + '_refined_' + str(slabel) + fsl_ext()
        seg_img = nib.load(fseg_vol)
        seg_data = seg_img.get_data()

        thisFA[0,sind] = np.mean(fa_data[seg_data==1])
        thisFAstd[0,sind] = np.std(fa_data[seg_data==1])
        thisMD[0,sind] = np.mean(md_data[seg_data==1])
        thisMDstd[0,sind] = np.std(md_data[seg_data==1])
        sind = sind + 1

    return thisFA, thisFAstd, thisMD, thisMDstd, subcort_label

def main(argv):
    subList = []
    outFile = ''
    
    help_string = """usage: microbrain_MeasureSubcortical.py -s <subject_directory_list> -o outputFile 
    description: microbrain_measureSubcortical.py outputs  

    mandatory arguments:
    -s <directory>,--subList= - specifies the list of directories (e.g. [SubDir1,SubDir2,SubDir3]) for which subcortical diffusion measurements will be taken
    -o <outFile>, --outFile= - file for output 
    """

    try:
        # Note some of these options were left for testing purposes
        opts, args = getopt.getopt(argv,"hs:o:",["subList=","outFile="])
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
            subList = arg.replace('[','').replace(']','').split(',')
        elif opt in ("-o", "--outfile"):
            outfile = os.path.normpath(arg)
    

    sublist_fa = np.zeros((len(subList),8))
    sublist_fa_std = np.zeros((len(subList),8))
    sublist_md = np.zeros((len(subList),8))
    sublist_md_std = np.zeros((len(subList),8))
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

        FA, FAstd, MD, MDstd, subcort_label = extract_FA_MD_from_subcortical_segmentations(subID, segDir, ffa, fmd)

        sublist_fa[sub_ind,:] = FA
        sublist_fa_std[sub_ind,:] = FAstd
        sublist_md[sub_ind,:] = MD
        sublist_md_std[sub_ind,:] = MDstd
        sub_ind = sub_ind + 1

    subList_labels = np.array(subID_list, dtype='|S40')[:,np.newaxis]
    subcort_label = [''] + subcort_label
    subcort_label = str(subcort_label).replace("'","").replace('[','').replace(']','')
    
    print(subcort_label) 
    
    np.savetxt(outfile + '_fa_mean.csv',np.hstack((subList_labels,sublist_fa.astype(np.str_))),delimiter=',', fmt='%s', header=subcort_label)
    np.savetxt(outfile + '_fa_std.csv',np.hstack((subList_labels,sublist_fa_std.astype(np.str_))),delimiter=',', fmt='%s', header=subcort_label)
    np.savetxt(outfile + '_md_mean.csv',np.hstack((subList_labels,sublist_md.astype(np.str_))),delimiter=',', fmt='%s', header=subcort_label)
    np.savetxt(outfile + '_md_std.txt',np.hstack((subList_labels,sublist_md_std.astype(np.str_))),delimiter=',', fmt='%s', header=subcort_label)

if __name__ == "__main__":
   main(sys.argv[1:])
