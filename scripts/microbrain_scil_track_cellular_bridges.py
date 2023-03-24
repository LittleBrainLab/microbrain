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

from scipy.ndimage import binary_fill_holes

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


def run_cell_bridge_tractography(subDir, subID, outDir, algo='prob', nbr_seeds=10, sfthres_init=0.1, sfthres=0.15, min_length=20, min_mesh_length=0.10, max_mesh_length=0.15):

    mask_dir = outDir + '/tracking_mask'
    if not os.path.exists(mask_dir):
        os.makedirs(mask_dir)

    # Create tracking mask from mbrain segmentation
    fmask_tracking_float = mask_dir + '/whole_brain_tracking_mask.nii.gz'
    if not os.path.exists(fmask_tracking_float):
        os.system('python microbrain_export_wm_tracking_mask.py -s ' + subDir + ' -o ' + fmask_tracking_float)

    # Convert mask to int
    fmask_tracking = fmask_tracking_float.replace('.nii.gz', '_int.nii.gz')
    if not os.path.exists(fmask_tracking):
        os.system('fslmaths ' + fmask_tracking_float + ' ' + fmask_tracking + ' -odt int')

    # Generate fiber response function
    ## Note these definitions only work for HCP data right now
    fdwi = subDir + '/orig/' + subID + '.nii.gz'
    fbval = subDir + '/orig/' + subID + '.bval'
    fbvec = subDir + '/orig/' + subID + '.bvec'
    
    fodf_dir = outDir + '/fodf'
    if not os.path.exists(fodf_dir):
        os.makedirs(fodf_dir)
    
    fresponse = fodf_dir + '/' + subID + '_frf.txt'
    if not os.path.exists(fresponse):
        print('Generating fiber response function')
        os.system('scil_compute_ssst_frf.py ' + fdwi + ' ' + fbval + ' ' + fbvec + ' ' + fresponse + ' --mask ' + fmask_tracking)

    # convert brain mask to int
    fmask_brain_float = subDir + '/orig/' + subID + '_mask.nii.gz'
    fmask_brain = subDir + '/orig/' + subID + '_mask_int.nii.gz'
    if not os.path.exists(fmask_brain):
        os.system('fslmaths ' + fmask_brain_float + ' ' + fmask_brain + ' -odt int')

    ffodf = fodf_dir + '/' + subID + '_fodf.nii.gz'
    if not os.path.exists(ffodf):
        print('Generating fodf')
        os.system('scil_compute_ssst_fodf.py ' + fdwi + ' ' + fbval + ' ' + fbvec + ' ' + fresponse + ' ' + ffodf + ' --mask ' + fmask_brain)
    
    # Generate seed based tractography
    tracking_dir = outDir + '/gm_cellular_bridge_tractography'
    if not os.path.exists(tracking_dir):
        os.makedirs(tracking_dir)

    mesh_dir = tracking_dir + '/subcortical_segmentation_ply'
    if not os.path.exists(mesh_dir):
        os.makedirs(mesh_dir)

    seed_dir = tracking_dir + '/seeds'
    if not os.path.exists(seed_dir):
        os.makedirs(seed_dir)

    tractogram_dir = tracking_dir + '/tractograms'
    if not os.path.exists(tractogram_dir):
        os.makedirs(tractogram_dir)

    # for each hemisphere left and right create a tractogram for the striatum and globus pallidus
    for hemi in ['LEFT', 'RIGHT']:
        for struct in ['STRIATUM', 'GLOBUS']:
            StructVTK=subDir + '/subcortical_segmentation/mesh_output/' + subID + '_refined_' + hemi + '_' + struct + '.vtk' 
            
            StructPLY=mesh_dir + '/' + subID + '_refined_' + hemi + '_' + struct + '.ply'
            if not os.path.exists(StructPLY):
                os.system('scil_convert_surface.py ' + StructVTK + ' ' +  StructPLY)

            StructPLY_remesh = StructPLY.replace('.ply','_remesh.ply')
            if not os.path.exists(StructPLY_remesh):
                os.system('mirtk remesh-surface ' + StructPLY + ' ' + StructPLY_remesh + ' -min_edgelength ' + str(min_mesh_length) + ' -max_edgelength ' + str(max_mesh_length))
            
            # Output txt files containing mesh vertex coordinates and norms
            inCoords_remesh = seed_dir + '/' + subID + '_refined_' + hemi + '_' + struct + '_remesh_coords.txt'
            inNorms_remesh = seed_dir + '/' + subID + '_refined_' + hemi + '_' + struct + '_remesh_norms.txt'
            StructPLY_remesh_reorient=mesh_dir + '/' + subID + '_refined_' + hemi + '_' + struct + '_remesh_reorient.ply'
            if not os.path.exists(inCoords_remesh):
                os.system('python scil_convert_mesh_to_coords_norms.py ' + StructPLY_remesh + ' ' + inCoords_remesh + ' ' + inNorms_remesh + ' --apply_transform ' + fmask_brain + ' --ras --output_mesh ' + StructPLY_remesh_reorient)

            # Generate tractogram for structure
            out_tractogram_remesh = tractogram_dir + '/' + subID + '_refined_' + hemi + '_' + struct + '_tractogram.trk'
            if not os.path.exists(out_tractogram_remesh):
                os.system('scil_compute_mesh_based_local_tracking.py ' + ffodf + ' ' + inCoords_remesh + ' ' +
                        fmask_brain + ' ' + out_tractogram_remesh + ' --in_norm_list ' + inNorms_remesh +
                        ' --save_seeds --min_length ' + str(min_length) + ' --sfthres_init ' + str(sfthres_init) + ' --sfthres ' + str(sfthres) + ' --algo ' + str(algo) + ' --nbr_sps ' + str(nbr_seeds) + ' -v ')
    
    # Filter tractograms to create a single cellular bridge tractogram per hemisphere
    for hemi in ['LEFT', 'RIGHT']:
        # STRIATUM Both Ends
        in_tractogram_STRIATUM = tractogram_dir + '/' + subID + '_refined_' + hemi + '_STRIATUM_tractogram.trk'
        in_mesh_STRIATUM = mesh_dir + '/' + subID + '_refined_' + hemi + '_STRIATUM_remesh.ply'
        out_tractogram_STRIATUM_STRIATUM = tractogram_dir + '/' + subID + '_refined_' + hemi + '_STRIATUM_tractogram_filter_STRIATUM_Both_Ends.trk'
        if not os.path.exists(out_tractogram_STRIATUM_STRIATUM):
            os.system('scil_filter_tractogram_based_on_mesh.py ' + in_tractogram_STRIATUM + ' ' + out_tractogram_STRIATUM_STRIATUM + ' --mesh_roi ' + in_mesh_STRIATUM + ' both_ends include')
        
        # STRIATUM orientation
        out_tractogram_STRIATUM_STRIATUM_orientation = out_tractogram_STRIATUM_STRIATUM.replace('.trk', '_orientationMaxY3.trk')
        if not os.path.exists(out_tractogram_STRIATUM_STRIATUM_orientation):
            os.system('scil_filter_streamlines_by_orientation.py ' + out_tractogram_STRIATUM_STRIATUM + ' ' + out_tractogram_STRIATUM_STRIATUM_orientation + ' --max_y 3 ')

        # STRIATUM length
        out_tractogram_STRIATUM_STRIATUM_orientation_length = out_tractogram_STRIATUM_STRIATUM_orientation.replace('.trk', '_maxlen20.trk')
        if not os.path.exists(out_tractogram_STRIATUM_STRIATUM_orientation_length):
            os.system('scil_filter_streamlines_by_length.py ' + out_tractogram_STRIATUM_STRIATUM_orientation + ' ' + out_tractogram_STRIATUM_STRIATUM_orientation_length + ' --maxL 20')
        
        # GLOBUS end in STRIATUM


        # STRIATUM end in GLOBUS


    # Output quantative metrics for each tractogram



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
