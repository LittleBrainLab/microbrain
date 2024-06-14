#!/usr/local/bin/python
import nibabel as nib
import numpy as np
from shutil import which
from os import path
import getopt
import os
import sys
from glob import glob
import inspect


def get_fsl_atlas_dir():
    """
    Return tissue directory in microbrain repository

    Returns
    -------
    tissue_dir: string
        tissue path
    """

    import microbrain  # ToDo. Is this the only way?
    module_path = inspect.getfile(microbrain)

    fsl_atlas_dir = os.path.dirname(module_path) + "/data/fsl/atlases/"

    return fsl_atlas_dir


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


def run_cell_bridge_tractography(subDir, subID, outDir, algo='prob', nbr_seeds=10, sfthres_init=0.1, sfthres=0.15, min_length_tractogram=0, min_length=2, max_length=20, min_mesh_length=0.10, max_mesh_length=0.15):
    """
    Run cellular bridge tractography for a subject (Little 2024, in prep)

    Parameters
    ----------
    subDir : str
        The subject directory
    subID : str
        The subject ID
    outDir : str
        The output directory

    Optional Parameters
    ----------
    algo : str
        The tracking algorithm prob or det (default: prob)
    nbr_seeds : int
        The number of seeds per vertex (default: 10)
    sfthres_init : float
        The initial fODF threshold for streamline tractography (default: 0.1)
    sfthres : float
        The fODF threshold for streamline tractography (default: 0.15)
    min_length_tractogram : int
        The minimum length of a streamline when generating tractogram(default: 0)
    min_length : int
        The minimum length of the streamline when filtering for CLGBs (default: 2)
    max_length : int
        The maximum length of the streamline when filtering for CLGBs (default: 20)
    min_mesh_length : float
        The minimum edge length of the mesh (default: 0.10)
    max_mesh_length : float
        The maximum edge length of the mesh (default: 0.15)

    Returns
    -------
    none
    """

    mask_dir = outDir + '/tracking_mask'
    if not os.path.exists(mask_dir):
        os.makedirs(mask_dir)

    # Create tracking mask from microbrain segmentation
    fmask_tracking_float = mask_dir + '/whole_brain_tracking_mask.nii.gz'
    if not os.path.exists(fmask_tracking_float):
        os.system('microbrain_export_wm_tracking_mask.py -s ' +
                  subDir + ' -o ' + fmask_tracking_float)

    # Convert mask to int
    fmask_tracking = fmask_tracking_float.replace('.nii.gz', '_int.nii.gz')
    if not os.path.exists(fmask_tracking):
        os.system('fslmaths ' + fmask_tracking_float +
                  ' ' + fmask_tracking + ' -odt int')

    # Get preprocessing suffix from mean b0
    mean_b0 = glob(subDir + '/meanDWI/' + subID + '*_mean_b0.nii.gz')[0]
    preproc_suffix = mean_b0.split(subID)[-1].replace('_mean_b0.nii.gz', '')[1:]

    if preproc_suffix == '':
        fdwi = subDir + '/orig/' + subID + '.nii.gz'
        fbval = subDir + '/orig/' + subID + '.bval'
        fbvec = subDir + '/orig/' + subID + '.bvec'
    else:
        # if preproc_suffix contains 'EDDY' then use rotated bvecs
        if 'EDDY' in preproc_suffix:
            fbvec = subDir + '/preprocessed/' + subID + '_' + \
                preproc_suffix + '.nii.gz.eddy_rotated_bvecs'
        else:
            fbvec = subDir + '/orig/' + subID + '.bvec'
        fbval = subDir + '/orig/' + subID + '.bval'
        fdwi = subDir + '/preprocessed/' + subID + '_' + preproc_suffix + '.nii.gz'

    fodf_dir = outDir + '/fodf'
    if not os.path.exists(fodf_dir):
         os.makedirs(fodf_dir)

    # Generate fiber response function
    fresponse = fodf_dir + '/' + subID + '_frf.txt'
    if not os.path.exists(fresponse):
        print('Generating fiber response function')
        os.system('scil_frf_ssst.py ' + fdwi + ' ' + fbval +
                  ' ' + fbvec + ' ' + fresponse + ' --mask_wm ' + fmask_tracking)

    # convert brain mask to int
    fmask_brain_float = glob(subDir + '/orig/' + subID + '*_mask.nii.gz')[0]
    fmask_brain = fmask_brain_float.replace('_mask.nii.gz', '_mask_int.nii.gz')
    if not os.path.exists(fmask_brain):
         os.system('fslmaths ' + fmask_brain_float +
                   ' ' + fmask_brain + ' -odt int')

    ffodf = fodf_dir + '/' + subID + '_fodf.nii.gz'
    if not os.path.exists(ffodf):
         print('Generating fodf')
         os.system('scil_fodf_ssst.py ' + fdwi + ' ' + fbval + ' ' +
                   fbvec + ' ' + fresponse + ' ' + ffodf + ' --mask ' + fmask_brain)

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

    tractogram_gm_cellular_bridge_dir = tracking_dir + \
        '/tractograms_gm_cellular_bridges'
    if not os.path.exists(tractogram_gm_cellular_bridge_dir):
        os.makedirs(tractogram_gm_cellular_bridge_dir)

    metric_dir = tracking_dir + '/metric_maps'
    if not os.path.exists(metric_dir):
         os.makedirs(metric_dir)

    ic_roi_dir = tracking_dir + '/ic_roi'
    if not os.path.exists(ic_roi_dir):
        os.makedirs(ic_roi_dir)

    # for each hemisphere left and right create a tractogram for the striatum and globus pallidus
    # Generating globus tractogram here just for QA purposes (is not included in gm cell bridge reconstruction)
    for hemi in ['LEFT', 'RIGHT']:
        # register JHU atlas to subject space
        atlas = get_fsl_atlas_dir() + 'JHU/JHU-ICBM-labels-1mm.nii.gz'
        regDir = subDir + '/registration/'
        affine = regDir + '/ants_native2mni_0GenericAffine.mat'
        warp = regDir + '/ants_native2mni_1InverseWarp.nii.gz'
        regAtlas = ic_roi_dir + '/JHU-ICBM-labels-1mm_reg2native.nii.gz'

        if not os.path.exists(regAtlas):
            os.system('antsApplyTransforms -d 3 -i ' + atlas + ' -r ' + fmask_brain +
                      ' -o ' + regAtlas + ' -n NearestNeighbor -t [' + affine + ',1] -t ' + warp)

        ic_roi = ic_roi_dir + '/' + subID + '_' + hemi + '_ic_roi.nii.gz'
        ic_roi_seeding_mask = ic_roi.replace('.nii.gz', '_seeding_mask.nii.gz')
        if not os.path.exists(ic_roi):
            if hemi == 'LEFT':
                os.system('fslmaths ' + regAtlas + ' -thr 18 -uthr 18 -bin ' +
                          ic_roi.replace('.nii.gz', '_anterior.nii.gz'))
                os.system('fslmaths ' + regAtlas + ' -thr 20 -uthr 20 -bin ' +
                          ic_roi.replace('.nii.gz', '_posterior.nii.gz'))
                os.system('fslmaths ' + regAtlas + ' -thr 44 -uthr 44 -bin ' +
                          ic_roi.replace('.nii.gz', '_slf.nii.gz'))
                os.system('fslmaths ' + regAtlas + ' -thr 26 -uthr 26 -bin ' +
                          ic_roi.replace('.nii.gz', '_scorrad.nii.gz'))
            else:
                os.system('fslmaths ' + regAtlas + ' -thr 17 -uthr 17 -bin ' +
                          ic_roi.replace('.nii.gz', '_anterior.nii.gz'))
                os.system('fslmaths ' + regAtlas + ' -thr 19 -uthr 19 -bin ' +
                          ic_roi.replace('.nii.gz', '_posterior.nii.gz'))
                os.system('fslmaths ' + regAtlas + ' -thr 43 -uthr 43 -bin ' +
                          ic_roi.replace('.nii.gz', '_slf.nii.gz'))
                os.system('fslmaths ' + regAtlas + ' -thr 25 -uthr 25 -bin ' +
                          ic_roi.replace('.nii.gz', '_scorrad.nii.gz'))

            os.system('fslmaths ' + ic_roi.replace('.nii.gz', '_anterior.nii.gz') + ' -add ' + ic_roi.replace('.nii.gz', '_posterior.nii.gz') +
                      ' -add ' + ic_roi.replace('.nii.gz', '_slf.nii.gz') + ' -add ' + ic_roi.replace('.nii.gz', '_scorrad.nii.gz') + ' ' + ic_roi)
            os.system('fslmaths ' + ic_roi.replace('.nii.gz', '_anterior.nii.gz') + ' -add ' +
                      ic_roi.replace('.nii.gz', '_posterior.nii.gz') + ' ' + ic_roi_seeding_mask)

        # dilate IC ROI
        ic_roi_dilated = ic_roi.replace('.nii.gz', '_dilated.nii.gz')
        if not os.path.exists(ic_roi_dilated):
            os.system('fslmaths ' + ic_roi +
                      ' -kernel sphere 3 -dilM ' + ic_roi_dilated)

        for struct in ['STRIATUM', 'GLOBUS']:
            StructVTK = subDir + '/subcortical_segmentation/mesh_output/' + \
                subID + '_refined_' + hemi + '_' + struct + '.vtk'

            StructPLY = mesh_dir + '/' + subID + '_refined_' + hemi + '_' + struct + '.ply'
            if not os.path.exists(StructPLY):
                os.system('scil_surface_convert.py ' +
                          StructVTK + ' ' + StructPLY)

            StructPLY_remesh = StructPLY.replace('.ply', '_remesh.ply')
            if not os.path.exists(StructPLY_remesh):
                os.system('mirtk remesh-surface ' + StructPLY + ' ' + StructPLY_remesh +
                          ' -min-edgelength ' + str(min_mesh_length) + ' -max-edgelength ' + str(max_mesh_length))

            # Output txt files containing mesh vertex coordinates and norms
            inCoords_remesh = seed_dir + '/' + subID + '_refined_' + \
                hemi + '_' + struct + '_remesh_coords.txt'
            inNorms_remesh = seed_dir + '/' + subID + '_refined_' + \
                hemi + '_' + struct + '_remesh_norms.txt'
            seeds_trk = seed_dir + '/' + subID + '_refined_' + \
                hemi + '_' + struct + '_seeds.trk'
            StructPLY_remesh_reorient = mesh_dir + '/' + subID + \
                '_refined_' + hemi + '_' + struct + '_remesh_reorient.ply'
            if not os.path.exists(inCoords_remesh):
                os.system('meshtrack_surface_convert_to_coordinates.py ' + StructPLY_remesh + ' ' + inCoords_remesh + ' ' + inNorms_remesh + ' --apply_transform ' +
                          fmask_brain + ' --ras --output_mesh ' + StructPLY_remesh_reorient + ' --within_mask ' + ic_roi_dilated + ' --output_trk ' + seeds_trk)

            # Generate tractogram for structure
            out_tractogram_remesh = tractogram_dir + '/' + subID + \
                '_refined_' + hemi + '_' + struct + '_tractogram.trk'
            if not os.path.exists(out_tractogram_remesh) and struct != 'GLOBUS':
                print('sfthres: ' + str(sfthres))
                os.system('meshtrack_tracking_based_on_surface.py ' + ffodf + ' ' + inCoords_remesh + ' ' +
                          fmask_tracking + ' ' + out_tractogram_remesh + ' --in_norm_list ' + inNorms_remesh +
                          ' --save_seeds --min_length ' + str(min_length_tractogram) + ' --sfthres_init ' + str(sfthres_init) + ' --sfthres ' + str(sfthres) + ' --algo ' + str(algo) + ' --nbr_sps ' + str(nbr_seeds) + ' -v')

    # Filter tractograms to create a single cellular bridge tractogram per hemisphere
    streamline_file_list = []
    for hemi in ['LEFT', 'RIGHT']:
        # tractograms
        in_tractogram_STRIATUM = tractogram_dir + '/' + subID + \
            '_refined_' + hemi + '_STRIATUM_tractogram.trk'

        # Meshes
        in_mesh_STRIATUM = mesh_dir + '/' + subID + \
            '_refined_' + hemi + '_STRIATUM_remesh.ply'

        # STRIATUM Both Ends
        out_tractogram_STRIATUM_STRIATUM = tractogram_gm_cellular_bridge_dir + '/' + \
            subID + '_refined_' + hemi + '_STRIATUM_tractogram_filter_STRIATUM_Both_Ends.trk'
        if not os.path.exists(out_tractogram_STRIATUM_STRIATUM):
            os.system('meshtrack_tractogram_filter_by_surface.py ' + in_tractogram_STRIATUM + ' ' +
                      out_tractogram_STRIATUM_STRIATUM + ' --mesh_roi ' + in_mesh_STRIATUM + ' both_ends include --dist_thr 0.1')

        # STRIATUM orientation
        out_tractogram_STRIATUM_STRIATUM_orientation = out_tractogram_STRIATUM_STRIATUM.replace(
           '.trk', '_orientationMaxY6.trk')
        if not os.path.exists(out_tractogram_STRIATUM_STRIATUM_orientation):
            os.system('scil_tractogram_filter_by_orientation.py ' + out_tractogram_STRIATUM_STRIATUM +
                      ' ' + out_tractogram_STRIATUM_STRIATUM_orientation + ' --max_y 6 ')

        # STRIATUM length
        out_tractogram_STRIATUM_STRIATUM_orientation_length = tractogram_gm_cellular_bridge_dir + \
            '/' + subID + '_refined_' + hemi + '_gm_celluar_bridge_tractogram.trk'
        if not os.path.exists(out_tractogram_STRIATUM_STRIATUM_orientation_length):
            os.system('scil_tractogram_filter_by_length.py ' + out_tractogram_STRIATUM_STRIATUM_orientation + ' ' +
                      out_tractogram_STRIATUM_STRIATUM_orientation_length + ' --maxL ' + str(max_length) + ' --minL ' + str(min_length))
        streamline_file_list.append(
            out_tractogram_STRIATUM_STRIATUM_orientation_length)

    # Output quantative metrics for each tractogram
    for streamline_file in streamline_file_list:
        streamline_file_name = streamline_file.split('/')[-1]
        out_streamline_density = metric_dir + '/' + \
            streamline_file_name.replace('.trk', '_density.nii.gz')
        if not os.path.exists(out_streamline_density):
            os.system('scil_tractogram_compute_density_map.py ' +
                      streamline_file + ' ' + out_streamline_density)

        out_streamline_density_norm = metric_dir + '/' + \
            streamline_file_name.replace('.trk', '_density_norm.nii.gz')
        if not os.path.exists(out_streamline_density_norm):
            os.system('scil_volume_math.py normalize_max ' + out_streamline_density +
                      ' ' + out_streamline_density_norm + ' --data float32')

        out_streamline_density_mask = metric_dir + '/' + \
            streamline_file_name.replace('.trk', '_density_mask.nii.gz')
        if not os.path.exists(out_streamline_density_mask):
            os.system('scil_volume_math.py ceil ' + out_streamline_density_norm +
                      ' ' + out_streamline_density_mask + ' --data uint8')

        out_afd_map = metric_dir + '/' + \
            streamline_file_name.replace('.trk', '_afd.nii.gz')
        if not os.path.exists(out_afd_map):
            os.system('scil_bundle_mean_fixel_afd.py ' +
                      streamline_file + ' ' + ffodf + ' ' + out_afd_map)

        out_bingham_map = metric_dir + '/' + \
            streamline_file_name.replace('.trk', '_bingham.nii.gz')
        if not os.path.exists(out_bingham_map):
            os.system('scil_fodf_to_bingham.py ' + ffodf + ' ' +
                      out_bingham_map + ' --mask ' + out_streamline_density_mask)

        out_fd_map = metric_dir + '/' + \
            streamline_file_name.replace('.trk', '_fd.nii.gz')
        if not os.path.exists(out_fd_map):
            os.system('scil_bingham_metrics.py ' + out_bingham_map +
                      ' --out_fd ' + out_fd_map + ' --mask ' + out_streamline_density_mask + ' --not_all')

        out_fd_fixel_map = metric_dir + '/' + \
            streamline_file_name.replace('.trk', '_fd_fixel.nii.gz')
        if not os.path.exists(out_fd_fixel_map):
            os.system('scil_bundle_mean_fixel_bingham_metric.py ' + streamline_file +
                      ' ' + out_bingham_map + ' ' + out_fd_map + ' ' + out_fd_fixel_map)

        out_metric_file = metric_dir + '/' + \
            streamline_file_name.replace('.trk', '_metrics.csv')
        out_weighted_metric_file = metric_dir + '/' + \
            streamline_file_name.replace('.trk', '_weighted_metrics.csv')
        if not os.path.exists(out_metric_file) or not os.path.exists(out_weighted_metric_file):
            # extract mean FA, AFD_fixel and FA_fixel for each tractogram
            fa_map = glob(subDir + '/DTI_maps/*FA.nii.gz')[0]
            metric_list = [fa_map, out_afd_map, out_fd_fixel_map]
            metric_name_list = ['FA', 'AFD', 'FD']

            # read in normalized density mask
            density_norm = nib.load(out_streamline_density_norm).get_fdata()

            metric_array = np.zeros((len(metric_list), 1))
            weighted_metric_array = np.zeros((len(metric_list), 1))
            for metric, metric_name in zip(metric_list, metric_name_list):
                metric_data = nib.load(metric).get_fdata()
                metric_array[metric_name_list.index(metric_name)] = np.mean(
                    metric_data[density_norm > 0])
                weighted_metric_array[metric_name_list.index(metric_name)] = np.mean(
                    metric_data[density_norm > 0] * density_norm[density_norm > 0])

            if not os.path.exists(out_metric_file):
                np.savetxt(out_metric_file, metric_array.T, delimiter=',',
                           header=','.join(metric_name_list), comments='')

            if not os.path.exists(out_weighted_metric_file):
                np.savetxt(out_weighted_metric_file, weighted_metric_array.T,
                           delimiter=',', header=','.join(metric_name_list), comments='')

    return


def main(argv):

    help_string = """usage: microbrain_export_wm_tracking_mask.py -s <subject_directory> -o <output_directory>
    description: Reconstructs caudalentricular gray matter bridges using mesh-based custom seeding and filtering tractography using methods from Little et al. 2024 (in prep)

    mandatory arguments:
    -s, --subDir <subject directory> - microbrain subject output directory

    optional arguments:
    -o, --outDir <output_directory> - directory for output if doesn't exist will make it (default: <subject_directory>/tracking))

    optional tracking parameters:
    --sfthres_init <float> - initial seed fODF threshold (default: 0.1)
    --sfthres <float> - seed fODF threshold (default: 0.15)
    """

    try:
        # Note some of these options were left for testing purposes
        opts, args = getopt.getopt(
            argv, "hs:o", ["subDir", "outDir", "sfthres=", "sfthres_init="])
    except getopt.GetoptError:
        print(help_string)
        sys.exit(2)

    if len(opts) == 0:
        print(help_string)
        sys.exit(2)

    # Defaults
    subList = []
    outDir = ''
    sfthres_init = 0.1
    sfthres = 0.15

    for opt, arg in opts:
        if opt == '-h':
            print(help_string)
            sys.exit()
        elif opt in ("-s", "--subDir"):
            subDir = os.path.normpath(arg)
        elif opt in ("-o", "--outDir"):
            outDir = os.path.normpath(arg)
        elif opt == "--sfthres_init":
            sfthres_init = float(arg)
        elif opt == "--sfthres":
            sfthres = float(arg)

    # Get FA, MD and subcortical GM segmentation directory
    baseDir, subID = os.path.split(os.path.normpath(subDir))

    if outDir == '':
        outDir = baseDir + '/' + subID + '/tracking'

    print('Running caudalentricular GM tractography using scilpy tools: ' + subID)

    # Create output directory if it doesn't exist
    if not os.path.exists(outDir):
        os.makedirs(outDir)

    run_cell_bridge_tractography(
        subDir, subID, outDir, sfthres=sfthres, sfthres_init=sfthres_init)


if __name__ == "__main__":
    main(sys.argv[1:])
