#!/usr/local/bin/python
import microbrain.utils.surf_util as sutil

import vtk
from shutil import which

from os import system, environ, path
import numpy as np
import nibabel as nib

from skimage.morphology import binary_erosion

# FSL Harvard Atlas label indices
LEFT_WHITE_IDX = 0
LEFT_CORTEX_IDX = 1
LEFT_VENT_IDX = 2
LEFT_THAL_IDX = 3
LEFT_CAUDATE_IDX = 4
LEFT_PUT_IDX = 5
LEFT_GLOB_IDX = 6
BRAIN_STEM_IDX = 7
LEFT_HIPPO_IDX = 8
LEFT_AMYG_IDX = 9
LEFT_ACCUM_IDX = 10

RIGHT_WHITE_IDX = 11
RIGHT_CORTEX_IDX = 12
RIGHT_VENT_IDX = 13
RIGHT_THAL_IDX = 14
RIGHT_CAUDATE_IDX = 15
RIGHT_PUT_IDX = 16
RIGHT_GLOB_IDX = 17
RIGHT_HIPPO_IDX = 18
RIGHT_AMYG_IDX = 19
RIGHT_ACCUM_IDX = 20

# indices specific for microbrain ROIs
LEFT_HIPPOAMYG_IDX = 88
LEFT_STRIATUM_IDX = 66
RIGHT_HIPPOAMYG_IDX = 188
RIGHT_STRIATUM_IDX = 166
VENT_IDX = 200


# move these functions to mbrain_io.py
def is_tool(name):
    """
    Checks to see if third party binary is installed and accessible by command line

    Parameters
    ----------
    name: string
        the name of the command to look for

    Returns
    -------
    Boolean: true if command exists, false if not in path
    """

    return which(name) is not None


def fsl_ext():
    """
    Gets the preferred image extension used by FSL

    Parameters
    ----------
    none

    Returns
    -------
    extension: a string containing the preferred output
    """

    fsl_extension = ''
    if environ['FSLOUTPUTTYPE'] == 'NIFTI':
        fsl_extension = '.nii'
    elif environ['FSLOUTPUTTYPE'] == 'NIFTI_GZ':
        fsl_extension = '.nii.gz'
    return fsl_extension


def register_probatlas_to_native(fsource, ftemplate, fatlas, regDir, cpu_num=0):
    """
    Given an image will register this image to the template (ANTs SyN)
    Then applies this transform to an atlas

    Parameters
    ----------
    fsource: string
        filename of image in native space
    ftemplate: string
        filename of image template in MNI space (or normal space)
    fatlas: string
        filename of 4D nifti file in MNI space to be registered
    regDir: string
        directory to store output

    Optional Parameters
    -------------------
    cpu_num: integer
        number of cpu threads to use for registration

    Returns
    -------
    fatlas_out: filename for atlas registered to native space
    """
    if not path.exists(regDir):
        system('mkdir ' + regDir)

    if cpu_num > 0:
        cpu_str = ' -n ' + str(cpu_num)
    else:
        cpu_str = ''

    method_suffix = '_ANTsReg_native'
    ftempbasename = path.basename(ftemplate)

    fatlas_basename = path.basename(fatlas)
    fatlas_out = regDir + \
        fatlas_basename.replace('.nii.gz', method_suffix + fsl_ext())

    ftransform_out = regDir + 'ants_native2mni_'

    print('Running Ants Registration')
    # Make template mask
    ftemplate_mask = regDir + \
        ftempbasename.replace('.nii.gz', '_mask' + fsl_ext())
    if not path.exists(ftemplate_mask):
        system('fslmaths ' + ftemplate + ' -bin ' + ftemplate_mask)

    if not path.exists(ftransform_out + '1InverseWarp.nii.gz'):
        system('antsRegistrationSyN.sh' +
               ' -d 3' +
               ' -f ' + ftemplate +
               ' -m ' + fsource +
               ' -o ' + ftransform_out +
               ' -x ' + ftemplate_mask +
               cpu_str)

    print('Warping probabilistic atlas to native space')
    if not path.exists(fatlas_out):
        system('antsApplyTransforms' +
               ' -t [' + ftransform_out + '0GenericAffine.mat,1] ' +
               ' -t ' + ftransform_out + '1InverseWarp.nii.gz ' +
               ' -r ' + fsource +
               ' -e 3 ' +
               ' -i ' + fatlas +
               ' -o ' + fatlas_out)
    else:
        print("ANTs nonlinear registeration of atlases already performed")

    return fatlas_out


def initial_voxel_labels_from_harvard(fharvard, output_prefix, outDir, md_file=None, md_thresh=0.0015, fa_file=None, fa_thresh=0.5):
    """
    Generates 3D meshes from the FSL harvard brain atlas.

    Parameters
    ----------
    fharvard: string
        filename for the harvard probabilistic atlas to be used (usually registered to native space)
    output_prefix: string
        prefix used for filenames for microbrain subject
    outDir: string
        output directory to store all files

    Optional Parameters:
    --------------------
    md_file: string
        filename for mean diffusivity map used to remove voxels from labels (useful in case of poor registration to native space)
    md_thresh: float
        removes voxels in lables with MD > md_thresh

    fa_file: string
        filename for fractional anisotropy map used to remove voxels from labels (useful in case of poor registration to native space)
    fa_thresh: string
        removes voxels in lables with FA > fa_thresh

    Returns
    -------
    None
    """

    if md_file:
        md_data = nib.load(md_file).get_fdata()

    if fa_file:
        fa_data = nib.load(fa_file).get_fdata()

    finitlabels_prefix = outDir + output_prefix + '_initialization'

    harvard_img = nib.load(fharvard)
    harvard_data = harvard_img.get_fdata()

    subcort_ind = [LEFT_THAL_IDX,
                   LEFT_CAUDATE_IDX,
                   LEFT_PUT_IDX,
                   LEFT_GLOB_IDX,
                   LEFT_HIPPO_IDX,
                   LEFT_AMYG_IDX,
                   LEFT_ACCUM_IDX,
                   LEFT_HIPPOAMYG_IDX,
                   LEFT_STRIATUM_IDX,
                   RIGHT_THAL_IDX,
                   RIGHT_CAUDATE_IDX,
                   RIGHT_PUT_IDX,
                   RIGHT_GLOB_IDX,
                   RIGHT_HIPPO_IDX,
                   RIGHT_AMYG_IDX,
                   RIGHT_ACCUM_IDX,
                   RIGHT_HIPPOAMYG_IDX,
                   RIGHT_STRIATUM_IDX,
                   VENT_IDX]

    subcort_label = ['LEFT_THALAMUS',
                     'LEFT_CAUDATE',
                     'LEFT_PUTAMEN',
                     'LEFT_GLOBUS',
                     'LEFT_HIPPO',
                     'LEFT_AMYGDALA',
                     'LEFT_ACCUMBENS',
                     'LEFT_HIPPOAMYG',
                     'LEFT_STRIATUM',
                     'RIGHT_THALAMUS',
                     'RIGHT_CAUDATE',
                     'RIGHT_PUTAMEN',
                     'RIGHT_GLOBUS',
                     'RIGHT_HIPPO',
                     'RIGHT_AMYGDALA',
                     'RIGHT_ACCUMBENS',
                     'RIGHT_HIPPOAMYG',
                     'RIGHT_STRIATUM',
                     'VENTRICLES']

    for sind, slabel in zip(subcort_ind, subcort_label):
        tmplabel = np.zeros(harvard_data.shape[0:3])
        if sind == RIGHT_ACCUM_IDX or sind == LEFT_ACCUM_IDX:
            tmplabel[binary_erosion(harvard_data[:, :, :, sind] > 25)] = 1
        elif sind == RIGHT_HIPPOAMYG_IDX:
            tmplabel[binary_erosion(np.logical_or(
                harvard_data[:, :, :, RIGHT_HIPPO_IDX] > 35, harvard_data[:, :, :, RIGHT_AMYG_IDX] > 35))] = 1
        elif sind == LEFT_HIPPOAMYG_IDX:
            tmplabel[binary_erosion(np.logical_or(
                harvard_data[:, :, :, LEFT_HIPPO_IDX] > 35, harvard_data[:, :, :, LEFT_AMYG_IDX] > 35))] = 1
        elif sind == RIGHT_STRIATUM_IDX:
            tmplabel[binary_erosion(np.logical_or(np.logical_or(harvard_data[:, :, :, RIGHT_PUT_IDX] > 15,
                                    harvard_data[:, :, :, RIGHT_CAUDATE_IDX] > 15), harvard_data[:, :, :, RIGHT_ACCUM_IDX] > 15))] = 1
        elif sind == LEFT_STRIATUM_IDX:
            tmplabel[binary_erosion(np.logical_or(np.logical_or(harvard_data[:, :, :, LEFT_PUT_IDX] > 15,
                                    harvard_data[:, :, :, LEFT_CAUDATE_IDX] > 15), harvard_data[:, :, :, LEFT_ACCUM_IDX] > 15))] = 1
        elif sind == RIGHT_AMYG_IDX or sind == LEFT_AMYG_IDX:
            tmplabel[binary_erosion(harvard_data[:, :, :, sind] > 35)] = 1
        elif sind == RIGHT_HIPPO_IDX or sind == LEFT_HIPPO_IDX:
            tmplabel[binary_erosion(harvard_data[:, :, :, sind] > 35)] = 1
        elif sind == VENT_IDX:
            tmplabel[harvard_data[:, :, :, LEFT_VENT_IDX] > 25] = 1
            tmplabel[harvard_data[:, :, :, RIGHT_VENT_IDX] > 25] = 1
        else:
            tmplabel[binary_erosion(harvard_data[:, :, :, sind] > 50)] = 1

        # # remove large MD voxels in GM if argument provided
        if sind != VENT_IDX:
            if md_file:
                tmplabel[md_data > md_thresh] = 0
            if fa_file:
                tmplabel[fa_data > fa_thresh] = 0

        finit_tmplabel = finitlabels_prefix + '_' + slabel + fsl_ext()
        nib.save(nib.Nifti1Image(tmplabel, harvard_img.affine), finit_tmplabel)
        system('mirtk extract-connected-components ' +
               finit_tmplabel + ' ' + finit_tmplabel)

        # Extract surfaces
        finit_surf = finitlabels_prefix + '_' + slabel + '.vtk'
        system('mirtk extract-surface ' + finit_tmplabel +
               ' ' + finit_surf + ' -isovalue 0.5')
        system('mirtk extract-connected-points ' +
               finit_surf + ' ' + finit_surf)
        system('mirtk smooth-surface ' + finit_surf + ' ' +
               finit_surf + ' -iterations 50 -lambda 0.05')

        # Label Surfaces
        system('mirtk project-onto-surface ' + finit_surf + ' ' + finit_surf +
               ' -constant ' + str(sind) + ' -pointdata -name struct_label')

    # Combine left and right thalamus
    lh_thalamus = sutil.read_surf_vtk(
        finitlabels_prefix + '_LEFT_THALAMUS.vtk')
    rh_thalamus = sutil.read_surf_vtk(
        finitlabels_prefix + '_RIGHT_THALAMUS.vtk')
    appender = vtk.vtkAppendPolyData()
    appender.AddInputData(lh_thalamus)
    appender.AddInputData(rh_thalamus)
    appender.Update()
    thalamus_surf = appender.GetOutput()
    sutil.write_surf_vtk(thalamus_surf, finitlabels_prefix + '_THALAMUS.vtk')

    return

def extract_surfaces_from_labels(flabels, label_list, outDir, fout_prefix):
    """
    Given a 3D nifti file containing a voxelwise labeling of structures, output a smoothed 3D surface representation of the labels

    Parameters
    ----------
    flabels: string
        filename for 3D nifti file containing labels
    label_list: string
        list of label integers indicating which labels to convert to surfaces (e.g. [1, 3])
    outDir: string
        directory to output the meshes
    fout_prefix: string
        prefix for output files

    Returns
    -------
    list of outputed vtk files
    """

    # Read in labels output nifti images
    label_img = nib.load(flabels)
    label_data = label_img.get_fdata()

    fout_list = []
    for label_ind in label_list:
        # Write tmp nifti
        ftmpNII = outDir + 'tmplabel' + str(label_ind) + fsl_ext()
        tmplabel = np.zeros(label_data.shape)
        tmplabel[label_data == label_ind] = 1
        nib.save(nib.Nifti1Image(tmplabel,
                 label_img.affine), ftmpNII)

        # Extract surface around label
        fout_surf = fout_prefix + '_' + str(label_ind) + '.vtk'
        system('mirtk extract-surface ' + ftmpNII +
               ' ' + fout_surf + ' -isovalue 0.5')
        system('rm ' + ftmpNII)

        # Smooth surface for the label
        system('mirtk smooth-surface ' + fout_surf + ' ' +
               fout_surf + ' -iterations 50 -lambda 0.2')

        # Add label to surface mesh
        system('mirtk project-onto-surface ' + fout_surf + ' ' + fout_surf +
               ' -constant ' + str(label_ind) + ' -pointdata -name struct_label')
        fout_list = fout_list + [fout_surf]

    return fout_list


def deform_subcortical_surfaces(fdwi, ffa, fmd, fharvard_native, segDir, initSegDir, subID, cpu_num=0):
    """
    Script to segment subcortical brain regions with 3D surface based deformation using DTI images/maps

    Parameters
    ----------
    fdwi: string
        filename for mean diffusion weigthed image (suggested b1000) used for globus pallidus segmentation
    ffa: string
        filename for fa map
    ffmd: string
        filename for md map
    fharvard_native: string
        FSL harvard probabilistic atlas transformed to native space
    segDir: string
        parent directory to store pipeline output
    initSegDir: string
        directory containing initial structure surfaces generated using initial_voxel_labels_from_harvard
    subID: string
        microbrain subject ID

    Optional Parameters
    -------------------
    cpu_num: integer
        number of cpu threads for MIRTK deform-mesh to use (default is mirtk default of all available threads)

    Returns
    -------
    none
    """

    min_edgelength = 0.6
    max_edgelength = 1.1

    min_hippo_edgelength = 0.6
    max_hippo_edgelength = 1.1

    curv_w = 8.0
    gcurv_w = 2.0

    step_size = 0.1
    step_num = 200

    averages = '4 2 1'

    initial_seg_prefix = '_initialization'
    seg_prefix = '_refined'

    if cpu_num > 0:
        cpu_str = ' -threads ' + str(cpu_num) + ' '
    else:
        cpu_str = ' '

    # Make output folders
    MDFA_Dir = segDir + 'md_plus_fa_maps/'
    if not path.exists(MDFA_Dir):
        system('mkdir ' + MDFA_Dir)

    probForceDir = segDir + 'probability_force_maps/'
    if not path.exists(probForceDir):
        system('mkdir ' + probForceDir)

    meshDir = segDir + 'mesh_output/'
    if not path.exists(meshDir):
        system('mkdir ' + meshDir)

    voxelDir = segDir + 'voxel_output/'
    if not path.exists(voxelDir):
        system('mkdir ' + voxelDir)

    atroposDir = segDir + 'atropos_hippoamyg_seg/'
    if not path.exists(atroposDir):
        system('mkdir ' + atroposDir)

    fsubcortseg_vtk = meshDir + subID + seg_prefix + '_subcortGM.vtk'
    if not path.exists(fsubcortseg_vtk):
        print('Subcortical Deformation Segmentation')

        # Make composite map of FA + csf probabilities. (This map defines the borders of the caudate and the thalamus)
        fmd_plusFA = MDFA_Dir + \
            path.basename(fmd.replace(fsl_ext(), '_plusFA' + fsl_ext()))
        system('fslmaths ' + fmd + ' -mul 1000 -add ' + ffa + ' ' + fmd_plusFA)
        MD_plusFA_data = nib.load(fmd_plusFA).get_fdata()

        fa_img = nib.load(ffa)

        harvard_img = nib.load(fharvard_native)
        harvard_data = harvard_img.get_fdata()

        dwi_img = nib.load(fdwi)
        dwi_data = dwi_img.get_fdata()

        # Probabilistic based force for subcortical structures
        glob_prob_force = np.zeros(dwi_data.shape)
        glob_pos_ind = [LEFT_WHITE_IDX, LEFT_PUT_IDX, LEFT_CAUDATE_IDX, LEFT_ACCUM_IDX,
                        RIGHT_WHITE_IDX, RIGHT_PUT_IDX, RIGHT_CAUDATE_IDX, RIGHT_ACCUM_IDX]
        glob_neg_ind = [LEFT_GLOB_IDX, RIGHT_GLOB_IDX]
        for pos_ind in glob_pos_ind:
            glob_prob_force = glob_prob_force + harvard_data[:, :, :, pos_ind]

        for neg_ind in glob_neg_ind:
            glob_prob_force = glob_prob_force - harvard_data[:, :, :, neg_ind]

        fglob_prob_force = probForceDir + subID + '_glob_prob_force' + fsl_ext()
        nib.save(nib.Nifti1Image(glob_prob_force,
                 fa_img.affine), fglob_prob_force)

        # Deform globus pallidus based on meanDWI and resticting movement into high FA regions
        fglobus_lh = initSegDir + subID + initial_seg_prefix + '_LEFT_GLOBUS.vtk'
        fglobus_lh_refined = meshDir + subID + seg_prefix + '_LEFT_GLOBUS.vtk'
        if not path.exists(fglobus_lh_refined):
            system('mirtk deform-mesh ' + fglobus_lh + ' ' + fglobus_lh_refined + ' -image ' + fdwi + ' -edge-distance 1.0 -edge-distance-averaging ' + averages + ' -edge-distance-smoothing 1 -edge-distance-median 1 -distance-image ' + fglob_prob_force + ' -distance 0.5 -distance-smoothing 1 -distance-averaging ' + averages + ' -distance-measure normal -optimizer EulerMethod -step ' + str(step_size) + ' -steps ' + str(
                step_num) + ' -epsilon 1e-6 -delta 0.001 -min-active 1% -reset-status -nointersection -fast-collision-test -min-width 0.01 -min-distance 0.01 -repulsion 4.0 -repulsion-distance 0.5 -repulsion-width 2.0 -curvature ' + str(curv_w) + ' -gauss-curvature ' + str(gcurv_w) + ' -edge-distance-type ClosestMaximum' + cpu_str + '-ascii -remesh 1 -min-edge-length ' + str(min_edgelength) + ' -max-edge-length ' + str(max_edgelength))
            sutil.surf_to_volume_mask(fdwi, fglobus_lh_refined, 1,
                                      voxelDir + subID + seg_prefix + '_LEFT_GLOBUS' + fsl_ext())

        fglobus_rh = initSegDir + subID + initial_seg_prefix + '_RIGHT_GLOBUS.vtk'
        fglobus_rh_refined = meshDir + subID + seg_prefix + '_RIGHT_GLOBUS.vtk'
        if not path.exists(fglobus_rh_refined):
            system('mirtk deform-mesh ' + fglobus_rh + ' ' + fglobus_rh_refined + ' -image ' + fdwi + ' -edge-distance 1.0 -edge-distance-averaging ' + averages + ' -edge-distance-smoothing 1 -edge-distance-median 1 -distance-image ' + fglob_prob_force + ' -distance 0.5 -distance-smoothing 1 -distance-averaging ' + averages + ' -distance-measure normal -optimizer EulerMethod -step ' + str(step_size) + ' -steps ' + str(step_num) +
                   ' -epsilon 1e-6 -delta 0.001 -min-active 1% -reset-status -nointersection -fast-collision-test -min-width 0.01 -min-distance 0.01 -repulsion 4.0 -repulsion-distance 0.5 -repulsion-width 2.0 -curvature ' + str(curv_w) + ' -gauss-curvature ' + str(gcurv_w) + ' -edge-distance-type ClosestMaximum' + cpu_str + '-ascii -remesh 1 -min-edge-length ' + str(min_edgelength) + ' -max-edge-length ' + str(max_edgelength))
            sutil.surf_to_volume_mask(fdwi, fglobus_rh_refined, 1,
                                      voxelDir + subID + seg_prefix + '_RIGHT_GLOBUS' + fsl_ext())

        # Generate map for Striatum deformation
        fa_globus = np.zeros(MD_plusFA_data.shape)
        fa_globus[:] = MD_plusFA_data[:]

        lh_globus_data = nib.load(
            voxelDir + subID + seg_prefix + '_LEFT_GLOBUS' + fsl_ext()).get_fdata()
        rh_globus_data = nib.load(
            voxelDir + subID + seg_prefix + '_RIGHT_GLOBUS' + fsl_ext()).get_fdata()
        fa_globus[lh_globus_data == 1] = 2
        fa_globus[rh_globus_data == 1] = 2
        fmd_plusFA_globus = MDFA_Dir + \
            path.basename(fmd_plusFA.replace(fsl_ext(), '_globus' + fsl_ext()))
        if not path.exists(fmd_plusFA_globus):
            nib.save(nib.Nifti1Image(
                fa_globus, fa_img.affine), fmd_plusFA_globus)

        # Probabilistic atlas based force for striatum
        striatum_prob_force = np.zeros(dwi_data.shape)
        striatum_pos_ind = [LEFT_CORTEX_IDX, LEFT_WHITE_IDX,
                            LEFT_GLOB_IDX, RIGHT_CORTEX_IDX, RIGHT_WHITE_IDX, RIGHT_GLOB_IDX]
        striatum_neg_ind = [LEFT_PUT_IDX, LEFT_CAUDATE_IDX, LEFT_ACCUM_IDX,
                            RIGHT_PUT_IDX, RIGHT_CAUDATE_IDX, RIGHT_ACCUM_IDX]
        for pos_ind in striatum_pos_ind:
            striatum_prob_force = striatum_prob_force + \
                harvard_data[:, :, :, pos_ind]

        for neg_ind in striatum_neg_ind:
            striatum_prob_force = striatum_prob_force - \
                harvard_data[:, :, :, neg_ind]

        fstriatum_prob_force = probForceDir + subID + '_striatum_prob_force' + fsl_ext()
        if not path.exists(fstriatum_prob_force):
            nib.save(nib.Nifti1Image(striatum_prob_force,
                                     fa_img.affine), fstriatum_prob_force)

        fstriatum_lh = initSegDir + subID + initial_seg_prefix + '_LEFT_STRIATUM.vtk'
        fstriatum_lh_refined = meshDir + subID + seg_prefix + '_LEFT_STRIATUM.vtk'
        if not path.exists(fstriatum_lh_refined):
            system('mirtk deform-mesh ' + fstriatum_lh + ' ' + fstriatum_lh_refined + ' -image ' + fmd_plusFA_globus + ' -edge-distance 1.0 -edge-distance-averaging ' + averages + ' -edge-distance-smoothing 1 -edge-distance-median 1  -distance-image ' + fstriatum_prob_force + ' -distance 0.25 -distance-smoothing 1 -distance-averaging ' + averages + ' -distance-measure normal -optimizer EulerMethod -step ' + str(step_size) + ' -steps ' + str(step_num) +
                   ' -epsilon 1e-6 -delta 0.001 -min-active 1% -reset-status -nointersection -fast-collision-test -min-width 0.01 -min-distance 0.01 -repulsion 4.0 -repulsion-distance 0.5 -repulsion-width 2.0 -curvature ' + str(curv_w) + ' -gauss-curvature ' + str(gcurv_w) + ' -edge-distance-type ClosestMaximum' + cpu_str + '-ascii -remesh 1 -min-edge-length ' + str(min_edgelength) + ' -max-edge-length ' + str(max_edgelength) + ' -edge-distance-min-intensity 0.3')
            sutil.surf_to_volume_mask(fdwi, fstriatum_lh_refined, 1,
                                      voxelDir + subID + seg_prefix + '_LEFT_STRIATUM' + fsl_ext())

        fstriatum_rh = initSegDir + subID + initial_seg_prefix + '_RIGHT_STRIATUM.vtk'
        fstriatum_rh_refined = meshDir + subID + seg_prefix + '_RIGHT_STRIATUM.vtk'
        if not path.exists(fstriatum_rh_refined):
            system('mirtk deform-mesh ' + fstriatum_rh + ' ' + fstriatum_rh_refined + ' -image ' + fmd_plusFA_globus + ' -edge-distance 1.0 -edge-distance-averaging ' + averages + ' -edge-distance-smoothing 1 -edge-distance-median 1  -distance-image ' + fstriatum_prob_force + ' -distance 0.25 -distance-smoothing 1 -distance-averaging ' + averages + ' -distance-measure normal -optimizer EulerMethod -step ' + str(step_size) + ' -steps ' + str(step_num) +
                   ' -epsilon 1e-6 -delta 0.001 -min-active 1% -reset-status -nointersection -fast-collision-test -min-width 0.01 -min-distance 0.01 -repulsion 4.0 -repulsion-distance 0.5 -repulsion-width 2.0 -curvature ' + str(curv_w) + ' -gauss-curvature ' + str(gcurv_w) + ' -edge-distance-type ClosestMaximum' + cpu_str + '-ascii -remesh 1 -min-edge-length ' + str(min_edgelength) + ' -max-edge-length ' + str(max_edgelength) + ' -edge-distance-min-intensity 0.3')
            sutil.surf_to_volume_mask(fdwi, fstriatum_rh_refined, 1,
                                      voxelDir + subID + seg_prefix + '_RIGHT_STRIATUM' + fsl_ext())

        fa_md_striatum = np.zeros(MD_plusFA_data.shape)
        fa_md_striatum[:] = MD_plusFA_data[:]

        lh_striatum_data = nib.load(
            voxelDir + subID + seg_prefix + '_LEFT_STRIATUM' + fsl_ext()).get_fdata()
        rh_striatum_data = nib.load(
            voxelDir + subID + seg_prefix + '_RIGHT_STRIATUM' + fsl_ext()).get_fdata()
        fa_md_striatum[lh_striatum_data == 1] = 2
        fa_md_striatum[rh_striatum_data == 1] = 2
        fmd_plusFA_striatum = MDFA_Dir + \
            path.basename(fmd_plusFA.replace(
                fsl_ext(), '_striatum' + fsl_ext()))

        if not path.exists(fmd_plusFA_striatum):
            nib.save(nib.Nifti1Image(fa_md_striatum,
                                     fa_img.affine), fmd_plusFA_striatum)

        # Probabilistic atlas based force for thalamus
        thal_prob_force = np.zeros(dwi_data.shape)
        thal_pos_ind = [LEFT_CORTEX_IDX, LEFT_WHITE_IDX, LEFT_HIPPO_IDX,
                        RIGHT_CORTEX_IDX, RIGHT_WHITE_IDX, RIGHT_HIPPO_IDX]
        thal_neg_ind = [LEFT_THAL_IDX, RIGHT_THAL_IDX]
        for pos_ind in thal_pos_ind:
            thal_prob_force = thal_prob_force + harvard_data[:, :, :, pos_ind]

        for neg_ind in thal_neg_ind:
            thal_prob_force = thal_prob_force - harvard_data[:, :, :, neg_ind]

        fthal_prob_force = probForceDir + subID + '_thalamus_prob_force' + fsl_ext()
        if not path.exists(fthal_prob_force):
            nib.save(nib.Nifti1Image(thal_prob_force,
                                     fa_img.affine), fthal_prob_force)

        fthalamus = initSegDir + subID + initial_seg_prefix + '_THALAMUS.vtk'
        fthalamus_refined = meshDir + subID + seg_prefix + '_THALAMUS.vtk'
        if not path.exists(fthalamus_refined):
            system('mirtk deform-mesh ' + fthalamus + ' ' + fthalamus_refined + ' -image ' + fmd_plusFA_striatum + ' -edge-distance 1.0 -edge-distance-averaging ' + averages + ' -edge-distance-smoothing 1 -edge-distance-median 1 -distance-image ' + fthal_prob_force + ' -distance 0.25 -distance-smoothing 1 -distance-averaging ' + averages + ' -distance-measure normal -optimizer EulerMethod -step ' + str(step_size) + ' -steps ' + str(step_num) +
                   ' -epsilon 1e-6 -delta 0.001 -min-active 1% -reset-status -nointersection -fast-collision-test -min-width 0.01 -min-distance 0.01 -repulsion 4.0 -repulsion-distance 0.5 -repulsion-width 2.0 -curvature ' + str(curv_w) + ' -gauss-curvature ' + str(gcurv_w) + ' -edge-distance-type ClosestMaximum' + cpu_str + '-ascii -remesh 1 -min-edge-length ' + str(min_edgelength) + ' -max-edge-length ' + str(max_edgelength) + ' -edge-distance-min-intensity 1.25')

        thalamus_surf = sutil.read_surf_vtk(fthalamus_refined)
        [lh_thalamus, rh_thalamus] = sutil.split_surface_by_label(
            thalamus_surf, label=[LEFT_THAL_IDX, RIGHT_THAL_IDX], label_name='struct_label')

        fthalamus_lh_refined = meshDir + subID + seg_prefix + '_LEFT_THALAMUS.vtk'
        if not path.exists(fthalamus_lh_refined):
            sutil.write_surf_vtk(lh_thalamus, fthalamus_lh_refined)
            sutil.surf_to_volume_mask(fdwi, fthalamus_lh_refined, 1,
                                      voxelDir + subID + seg_prefix + '_LEFT_THALAMUS' + fsl_ext())

        fthalamus_rh_refined = meshDir + subID + seg_prefix + '_RIGHT_THALAMUS.vtk'
        if not path.exists(fthalamus_rh_refined):
            sutil.write_surf_vtk(rh_thalamus, fthalamus_rh_refined)
            sutil.surf_to_volume_mask(fdwi, fthalamus_rh_refined, 1,
                                      voxelDir + subID + seg_prefix + '_RIGHT_THALAMUS' + fsl_ext())

        # Hippocampus/Amygdala segmentation
        hipamyg_prob_force = np.zeros(dwi_data.shape)
        hipamyg_pos_ind = [LEFT_CORTEX_IDX, LEFT_WHITE_IDX, LEFT_THAL_IDX,
                           RIGHT_CORTEX_IDX, RIGHT_WHITE_IDX, RIGHT_THAL_IDX, BRAIN_STEM_IDX]
        hipamyg_neg_ind = [LEFT_HIPPO_IDX, LEFT_AMYG_IDX,
                           RIGHT_HIPPO_IDX, RIGHT_AMYG_IDX]
        for pos_ind in hipamyg_pos_ind:
            hipamyg_prob_force = hipamyg_prob_force + \
                harvard_data[:, :, :, pos_ind]

        for neg_ind in hipamyg_neg_ind:
            hipamyg_prob_force = hipamyg_prob_force - \
                harvard_data[:, :, :, neg_ind]

        fhipamyg_prob_force = probForceDir + subID + '_hipamyg_prob_force' + fsl_ext()
        nib.save(nib.Nifti1Image(hipamyg_prob_force,
                                 fa_img.affine), fhipamyg_prob_force)

        fhippoamyg_lh = initSegDir + subID + initial_seg_prefix + '_LEFT_HIPPOAMYG.vtk'
        fhippoamyg_lh_refined = atroposDir + subID + seg_prefix + '_LEFT_HIPPOAMYG.vtk'
        if not path.exists(fhippoamyg_lh_refined):
            system('mirtk deform-mesh ' + fhippoamyg_lh + ' ' + fhippoamyg_lh_refined + ' -image ' + fmd_plusFA + ' -edge-distance 1.0 -edge-distance-averaging ' + averages + ' -edge-distance-smoothing 1 -edge-distance-median 1 -distance-image ' + fhipamyg_prob_force + ' -distance 0.25 -distance-smoothing 1 -distance-averaging ' + averages + ' -distance-measure normal -optimizer EulerMethod -step ' + str(step_size) + ' -steps ' + str(step_num) +
                   ' -epsilon 1e-6 -delta 0.001 -min-active 1% -reset-status -nointersection -fast-collision-test -min-width 0.01 -min-distance 0.01 -repulsion 4.0 -repulsion-distance 0.5 -repulsion-width 2.0 -curvature ' + str(curv_w) + ' -gauss-curvature ' + str(gcurv_w) + ' -edge-distance-type ClosestMaximum' + cpu_str + '-ascii -remesh 1 -min-edge-length ' + str(min_hippo_edgelength) + ' -max-edge-length ' + str(max_hippo_edgelength) + ' -edge-distance-min-intensity 1.0')
            sutil.surf_to_volume_mask(fdwi, fhippoamyg_lh_refined, 1,
                                      atroposDir + subID + seg_prefix + '_LEFT_HIPPOAMYG' + fsl_ext())

        fhippoamyg_rh = initSegDir + subID + initial_seg_prefix + '_RIGHT_HIPPOAMYG.vtk'
        fhippoamyg_rh_refined = atroposDir + subID + seg_prefix + '_RIGHT_HIPPOAMYG.vtk'
        if not path.exists(fhippoamyg_rh_refined):
            system('mirtk deform-mesh ' + fhippoamyg_rh + ' ' + fhippoamyg_rh_refined + ' -image ' + fmd_plusFA + ' -edge-distance 1.0 -edge-distance-averaging ' + averages + ' -edge-distance-smoothing 1 -edge-distance-median 1 -distance-image ' + fhipamyg_prob_force + ' -distance 0.25 -distance-smoothing 1 -distance-averaging ' + averages + ' -distance-measure normal -optimizer EulerMethod -step ' + str(step_size) + ' -steps ' + str(step_num) +
                   ' -epsilon 1e-6 -delta 0.001 -min-active 1% -reset-status -nointersection -fast-collision-test -min-width 0.01 -min-distance 0.01 -repulsion 4.0 -repulsion-distance 0.5 -repulsion-width 2.0 -curvature ' + str(curv_w) + ' -gauss-curvature ' + str(gcurv_w) + ' -edge-distance-type ClosestMaximum' + cpu_str + '-ascii -remesh 1 -min-edge-length ' + str(min_hippo_edgelength) + ' -max-edge-length ' + str(max_hippo_edgelength) + ' -edge-distance-min-intensity 1.0')
            sutil.surf_to_volume_mask(fdwi, fhippoamyg_rh_refined, 1,
                                      atroposDir + subID + seg_prefix + '_RIGHT_HIPPOAMYG' + fsl_ext())

        # Run 3 channel tissue segmentation to separate amygdala from hippocampus
        # prepare prob maps and separate primary eigenvector into separate nifti file
        atropos_prefix_lh = atroposDir + subID + \
            initial_seg_prefix + '_INITAMYGHIPPO_LH'
        atropos_prefix_lh_label = atroposDir + subID + \
            initial_seg_prefix + '_INITAMYGHIPPO_LH_LABELS' + fsl_ext()
        atropos_prefix_lh_probout = atroposDir + subID + \
            initial_seg_prefix + '_INITAMYGHIPPO_LH_PROBOUT_'

        famyg_lh_prob = atropos_prefix_lh + '_01' + fsl_ext()
        if not path.exists(famyg_lh_prob):
            nib.save(nib.Nifti1Image(
                harvard_data[:, :, :, LEFT_AMYG_IDX]/100, harvard_img.affine), famyg_lh_prob)

        fhippo_lh_prob = atropos_prefix_lh + '_02' + fsl_ext()
        if not path.exists(fhippo_lh_prob):
            nib.save(nib.Nifti1Image(
                harvard_data[:, :, :, LEFT_HIPPO_IDX]/100, harvard_img.affine), fhippo_lh_prob)

        fcortex_lh_prob = atropos_prefix_lh + '_03' + fsl_ext()
        if not path.exists(fcortex_lh_prob):
            nib.save(nib.Nifti1Image(
                harvard_data[:, :, :, LEFT_CORTEX_IDX]/100, harvard_img.affine), fcortex_lh_prob)

        atropos_prefix_rh = atroposDir + subID + \
            initial_seg_prefix + '_INITAMYGHIPPO_RH'
        atropos_prefix_rh_label = atroposDir + subID + \
            initial_seg_prefix + '_INITAMYGHIPPO_RH_LABELS' + fsl_ext()
        atropos_prefix_rh_probout = atroposDir + subID + \
            initial_seg_prefix + '_INITAMYGHIPPO_RH_PROBOUT_'

        famyg_rh_prob = atropos_prefix_rh + '_01' + fsl_ext()
        if not path.exists(famyg_rh_prob):
            nib.save(nib.Nifti1Image(
                harvard_data[:, :, :, RIGHT_AMYG_IDX]/100, harvard_img.affine), famyg_rh_prob)

        fhippo_rh_prob = atropos_prefix_rh + '_02' + fsl_ext()
        if not path.exists(fhippo_rh_prob):
            nib.save(nib.Nifti1Image(
                harvard_data[:, :, :, RIGHT_HIPPO_IDX]/100, harvard_img.affine), fhippo_rh_prob)

        fcortex_rh_prob = atropos_prefix_rh + '_03' + fsl_ext()
        if not path.exists(fcortex_rh_prob):
            nib.save(nib.Nifti1Image(
                harvard_data[:, :, :, RIGHT_CORTEX_IDX]/100, harvard_img.affine), fcortex_rh_prob)

        PriorWeight = 0.3
        if not path.exists(atropos_prefix_lh_label):
            system('Atropos' +
                   ' -a [' + fmd_plusFA + ']' +
                   ' -x ' + atroposDir + subID + seg_prefix + '_LEFT_HIPPOAMYG' + fsl_ext() +
                   ' -i PriorProbabilityImages[3, ' + atropos_prefix_lh + '_%02d' + fsl_ext() + ',' + str(PriorWeight) + ',0.0001]' +
                   ' -m [0.3, 2x2x2] ' +
                   ' --use-partial-volume-likelihoods false ' +
                   ' -s 1x3 -s 1x2 ' +
                   ' -o [' + atropos_prefix_lh_label + ',' + atropos_prefix_lh_probout + '%02d' + fsl_ext() + ']' +
                   ' -k HistogramParzenWindows[1.0,32]' +
                   ' -v 1')

        if not path.exists(atropos_prefix_rh_label):
            system('Atropos' +
                   ' -a [' + fmd_plusFA + ']' +
                   ' -x ' + atroposDir + subID + seg_prefix + '_RIGHT_HIPPOAMYG' + fsl_ext() +
                   ' -i PriorProbabilityImages[3, ' + atropos_prefix_rh + '_%02d' + fsl_ext() + ',' + str(PriorWeight) + ',0.0001]' +
                   ' -m [0.3, 2x2x2] ' +
                   ' --use-partial-volume-likelihoods false ' +
                   ' -s 1x3 -s 1x2 ' +
                   ' -o [' + atropos_prefix_rh_label + ',' + atropos_prefix_rh_probout + '%02d' + fsl_ext() + ']' +
                   ' -k HistogramParzenWindows[1.0,32]' +
                   ' -v 1')

        lh_labels = nib.load(atropos_prefix_lh_label).get_fdata()
        rh_labels = nib.load(atropos_prefix_rh_label).get_fdata()

        # Generate new surfaces for hippocampus and amygdala
        fhippoamyg_lh_step1 = atroposDir + subID + \
            initial_seg_prefix + '_LEFT_HIPPOAMYG_STEP1'
        fsurf_list = extract_surfaces_from_labels(
            atropos_prefix_lh_label, [1, 2], atroposDir, fhippoamyg_lh_step1)
        famyg_lh_step1 = fsurf_list[0]
        fhippo_lh_step1 = fsurf_list[1]

        fhippoamyg_rh_step1 = atroposDir + subID + \
            initial_seg_prefix + '_RIGHT_HIPPOAMYG_STEP1'
        fsurf_list = extract_surfaces_from_labels(
            atropos_prefix_rh_label, [1, 2], atroposDir, fhippoamyg_rh_step1)
        famyg_rh_step1 = fsurf_list[0]
        fhippo_rh_step1 = fsurf_list[1]

        # Build amygdala prob force
        amyg_prob_force = np.zeros(dwi_data.shape)
        amyg_pos_ind = [LEFT_CORTEX_IDX, LEFT_WHITE_IDX, LEFT_THAL_IDX, LEFT_HIPPO_IDX, RIGHT_HIPPO_IDX,
                        RIGHT_CORTEX_IDX, RIGHT_WHITE_IDX, RIGHT_THAL_IDX, BRAIN_STEM_IDX]
        amyg_neg_ind = [LEFT_AMYG_IDX, RIGHT_AMYG_IDX]
        for pos_ind in amyg_pos_ind:
            amyg_prob_force = amyg_prob_force + \
                harvard_data[:, :, :, pos_ind]

        for neg_ind in amyg_neg_ind:
            amyg_prob_force = amyg_prob_force - \
                harvard_data[:, :, :, neg_ind]

        famyg_prob_force = probForceDir + subID + '_amyg_prob_force' + fsl_ext()
        nib.save(nib.Nifti1Image(amyg_prob_force,
                                 fa_img.affine), famyg_prob_force)

        # Add hippocampus/cortex labels to md+fa map
        fa_md_hippocortex = np.zeros(MD_plusFA_data.shape)
        fa_md_hippocortex[:] = MD_plusFA_data[:]

        fa_md_hippocortex[lh_labels == 2] = 2
        fa_md_hippocortex[rh_labels == 2] = 2
        fa_md_hippocortex[lh_labels == 3] = 2
        fa_md_hippocortex[rh_labels == 3] = 2

        fmd_plusFA_hippocortex = MDFA_Dir + \
            path.basename(fmd_plusFA.replace(
                fsl_ext(), '_hippocortex' + fsl_ext()))
        if not path.exists(fmd_plusFA_hippocortex):
            nib.save(nib.Nifti1Image(fa_md_hippocortex,
                                     fa_img.affine), fmd_plusFA_hippocortex)

        famyg_lh_refined = meshDir + subID + seg_prefix + '_LEFT_AMYGDALA.vtk'
        if not path.exists(famyg_lh_refined):
            system('mirtk deform-mesh ' + famyg_lh_step1 + ' ' + famyg_lh_refined + ' -image ' + fmd_plusFA_hippocortex + ' -edge-distance 1.0 -edge-distance-averaging ' + averages + ' -edge-distance-smoothing 1 -edge-distance-median 1 -distance-image ' + famyg_prob_force + ' -distance 0.25 -distance-smoothing 1 -distance-averaging ' + averages + ' -distance-measure normal -optimizer EulerMethod -step ' + str(step_size) + ' -steps ' + str(step_num) +
                   ' -epsilon 1e-6 -delta 0.001 -min-active 1% -reset-status -nointersection -fast-collision-test -min-width 0.01 -min-distance 0.01 -repulsion 4.0 -repulsion-distance 0.5 -repulsion-width 2.0 -curvature ' + str(curv_w) + ' -gauss-curvature ' + str(gcurv_w) + ' -edge-distance-type ClosestMaximum' + cpu_str + '-ascii -remesh 1 -min-edge-length ' + str(min_hippo_edgelength) + ' -max-edge-length ' + str(max_hippo_edgelength) + ' -edge-distance-min-intensity 1.0')
            system('mirtk project-onto-surface ' + famyg_lh_refined + ' ' + famyg_lh_refined +
                   ' -constant ' + str(LEFT_AMYG_IDX) + ' -pointdata -name struct_label')
            sutil.surf_to_volume_mask(fdwi, famyg_lh_refined, 1,
                                      voxelDir + subID + seg_prefix + '_LEFT_AMYGDALA' + fsl_ext())

        famyg_rh_refined = meshDir + subID + seg_prefix + '_RIGHT_AMYGDALA.vtk'
        if not path.exists(famyg_rh_refined):
            system('mirtk deform-mesh ' + famyg_rh_step1 + ' ' + famyg_rh_refined + ' -image ' + fmd_plusFA_hippocortex + ' -edge-distance 1.0 -edge-distance-averaging ' + averages + ' -edge-distance-smoothing 1 -edge-distance-median 1 -distance-image ' + famyg_prob_force + ' -distance 0.25 -distance-smoothing 1 -distance-averaging ' + averages + ' -distance-measure normal -optimizer EulerMethod -step ' + str(step_size) + ' -steps ' + str(step_num) +
                   ' -epsilon 1e-6 -delta 0.001 -min-active 1% -reset-status -nointersection -fast-collision-test -min-width 0.01 -min-distance 0.01 -repulsion 4.0 -repulsion-distance 0.5 -repulsion-width 2.0 -curvature ' + str(curv_w) + ' -gauss-curvature ' + str(gcurv_w) + ' -edge-distance-type ClosestMaximum' + cpu_str + '-ascii -remesh 1 -min-edge-length ' + str(min_hippo_edgelength) + ' -max-edge-length ' + str(max_hippo_edgelength) + ' -edge-distance-min-intensity 1.0')
            system('mirtk project-onto-surface ' + famyg_rh_refined + ' ' + famyg_rh_refined +
                   ' -constant ' + str(RIGHT_AMYG_IDX) + ' -pointdata -name struct_label')
            sutil.surf_to_volume_mask(fdwi, famyg_rh_refined, 1,
                                      voxelDir + subID + seg_prefix + '_RIGHT_AMYGDALA' + fsl_ext())

        # Build amygdala prob force
        hippo_prob_force = np.zeros(dwi_data.shape)
        hippo_pos_ind = [LEFT_CORTEX_IDX, LEFT_WHITE_IDX, LEFT_THAL_IDX, LEFT_AMYG_IDX, RIGHT_AMYG_IDX,
                         RIGHT_CORTEX_IDX, RIGHT_WHITE_IDX, RIGHT_THAL_IDX, BRAIN_STEM_IDX]
        hippo_neg_ind = [LEFT_HIPPO_IDX, RIGHT_HIPPO_IDX]
        for pos_ind in hippo_pos_ind:
            hippo_prob_force = hippo_prob_force + \
                harvard_data[:, :, :, pos_ind]

        for neg_ind in hippo_neg_ind:
            hippo_prob_force = hippo_prob_force - \
                harvard_data[:, :, :, neg_ind]

        fhippo_prob_force = probForceDir + subID + '_hippo_prob_force' + fsl_ext()
        nib.save(nib.Nifti1Image(hippo_prob_force,
                                 fa_img.affine), fhippo_prob_force)

        # Add hippocampus/cortex labels to md+fa map
        fa_md_amyg = np.zeros(MD_plusFA_data.shape)
        fa_md_amyg[:] = MD_plusFA_data[:]

        fa_md_amyg[lh_labels == 1] = 2
        fa_md_amyg[rh_labels == 1] = 2
        fa_md_amyg[lh_labels == 3] = 2
        fa_md_amyg[rh_labels == 3] = 2

        fmd_plusFA_amyg = MDFA_Dir + \
            path.basename(fmd_plusFA.replace(
                fsl_ext(), '_amyg' + fsl_ext()))
        if not path.exists(fmd_plusFA_amyg):
            nib.save(nib.Nifti1Image(fa_md_amyg,
                                     fa_img.affine), fmd_plusFA_amyg)

        fhippo_lh_refined = meshDir + subID + seg_prefix + '_LEFT_HIPPO.vtk'
        if not path.exists(fhippo_lh_refined):
            system('mirtk deform-mesh ' + fhippo_lh_step1 + ' ' + fhippo_lh_refined + ' -image ' + fmd_plusFA_amyg + ' -edge-distance 1.0 -edge-distance-averaging ' + averages + ' -edge-distance-smoothing 1 -edge-distance-median 1 -distance-image ' + fhippo_prob_force + ' -distance 0.25 -distance-smoothing 1 -distance-averaging ' + averages + ' -distance-measure normal -optimizer EulerMethod -step ' + str(step_size) + ' -steps ' + str(step_num) +
                   ' -epsilon 1e-6 -delta 0.001 -min-active 1% -reset-status -nointersection -fast-collision-test -min-width 0.01 -min-distance 0.01 -repulsion 4.0 -repulsion-distance 0.5 -repulsion-width 2.0 -curvature ' + str(curv_w) + ' -gauss-curvature ' + str(gcurv_w) + ' -edge-distance-type ClosestMaximum' + cpu_str + '-ascii -remesh 1 -min-edge-length ' + str(min_hippo_edgelength) + ' -max-edge-length ' + str(max_hippo_edgelength) + ' -edge-distance-min-intensity 1.0')
            system('mirtk project-onto-surface ' + fhippo_lh_refined + ' ' + fhippo_lh_refined +
                   ' -constant ' + str(LEFT_HIPPO_IDX) + ' -pointdata -name struct_label')
            sutil.surf_to_volume_mask(fdwi, fhippo_lh_refined, 1,
                                      voxelDir + subID + seg_prefix + '_LEFT_HIPPO' + fsl_ext())

        fhippo_rh_refined = meshDir + subID + seg_prefix + '_RIGHT_HIPPO.vtk'
        if not path.exists(fhippo_rh_refined):
            system('mirtk deform-mesh ' + fhippo_rh_step1 + ' ' + fhippo_rh_refined + ' -image ' + fmd_plusFA_amyg + ' -edge-distance 1.0 -edge-distance-averaging ' + averages + ' -edge-distance-smoothing 1 -edge-distance-median 1 -distance-image ' + fhippo_prob_force + ' -distance 0.25 -distance-smoothing 1 -distance-averaging ' + averages + ' -distance-measure normal -optimizer EulerMethod -step ' + str(step_size) + ' -steps ' + str(step_num) +
                   ' -epsilon 1e-6 -delta 0.001 -min-active 1% -reset-status -nointersection -fast-collision-test -min-width 0.01 -min-distance 0.01 -repulsion 4.0 -repulsion-distance 0.5 -repulsion-width 2.0 -curvature ' + str(curv_w) + ' -gauss-curvature ' + str(gcurv_w) + ' -edge-distance-type ClosestMaximum' + cpu_str + '-ascii -remesh 1 -min-edge-length ' + str(min_hippo_edgelength) + ' -max-edge-length ' + str(max_hippo_edgelength) + ' -edge-distance-min-intensity 1.0')
            system('mirtk project-onto-surface ' + fhippo_rh_refined + ' ' + fhippo_rh_refined +
                   ' -constant ' + str(RIGHT_HIPPO_IDX) + ' -pointdata -name struct_label')
            sutil.surf_to_volume_mask(fdwi, fhippo_rh_refined, 1,
                                      voxelDir + subID + seg_prefix + '_RIGHT_HIPPO' + fsl_ext())

        # Ventricle segmentation
        # Probabilistic based force for subcortical structures
        vent_prob_force = np.zeros(dwi_data.shape)
        vent_pos_ind = [LEFT_WHITE_IDX, LEFT_PUT_IDX, LEFT_CAUDATE_IDX, LEFT_ACCUM_IDX, LEFT_GLOB_IDX, LEFT_THAL_IDX,
                        RIGHT_WHITE_IDX, RIGHT_PUT_IDX, RIGHT_CAUDATE_IDX, RIGHT_ACCUM_IDX, RIGHT_GLOB_IDX, RIGHT_THAL_IDX]

        vent_neg_ind = [LEFT_VENT_IDX, RIGHT_VENT_IDX]

        for pos_ind in vent_pos_ind:
            vent_prob_force = vent_prob_force + harvard_data[:, :, :, pos_ind]

        for neg_ind in vent_neg_ind:
            vent_prob_force = vent_prob_force - harvard_data[:, :, :, neg_ind]

        fvent_prob_force = probForceDir + subID + '_vent_prob_force' + fsl_ext()
        nib.save(nib.Nifti1Image(vent_prob_force,
                 fa_img.affine), fvent_prob_force)

        # Deform ventricles on mean DWI
        fvent = initSegDir + subID + initial_seg_prefix + '_VENTRICLES.vtk'
        fvent_refined = meshDir + subID + seg_prefix + '_VENTRICLES.vtk'
        if not path.exists(fvent_refined):
            system('mirtk deform-mesh ' + fvent + ' ' + fvent_refined + ' -image ' + fdwi + ' -edge-distance 1.0 -edge-distance-averaging ' + averages + ' -edge-distance-smoothing 1 -edge-distance-median 1 -distance-image ' + fvent_prob_force + ' -distance 0.5 -distance-smoothing 1 -distance-averaging ' + averages + ' -distance-measure normal -optimizer EulerMethod -step ' + str(step_size) + ' -steps ' + str(
                step_num) + ' -epsilon 1e-6 -delta 0.001 -min-active 1% -reset-status -nointersection -fast-collision-test -min-width 0.01 -min-distance 0.01 -repulsion 4.0 -repulsion-distance 0.5 -repulsion-width 2.0 -curvature ' + str(curv_w) + ' -gauss-curvature ' + str(gcurv_w) + ' -edge-distance-type ClosestMaximum' + cpu_str + '-ascii -remesh 1 -min-edge-length ' + str(min_edgelength) + ' -max-edge-length ' + str(max_edgelength))
            system('mirtk project-onto-surface ' + fvent_refined + ' ' + fvent_refined +
                   ' -constant ' + str(VENT_IDX) + ' -pointdata -name struct_label')
            sutil.surf_to_volume_mask(fdwi, fvent_refined, 1,
                                      voxelDir + subID + seg_prefix + '_VENTRICLES' + fsl_ext())

        # Append subcort surfs together into single vtk file
        fsubcortseg_vtk = meshDir + subID + seg_prefix + '_subcortGM.vtk'
        if not path.exists(fsubcortseg_vtk):
            appender = vtk.vtkAppendPolyData()
            appender.AddInputData(sutil.read_surf_vtk(fglobus_lh_refined))
            appender.AddInputData(sutil.read_surf_vtk(fglobus_rh_refined))
            appender.AddInputData(sutil.read_surf_vtk(fstriatum_lh_refined))
            appender.AddInputData(sutil.read_surf_vtk(fstriatum_rh_refined))
            appender.AddInputData(sutil.read_surf_vtk(fthalamus_lh_refined))
            appender.AddInputData(sutil.read_surf_vtk(fthalamus_rh_refined))
            appender.AddInputData(sutil.read_surf_vtk(famyg_rh_refined))
            appender.AddInputData(sutil.read_surf_vtk(famyg_lh_refined))
            appender.AddInputData(sutil.read_surf_vtk(fhippo_rh_refined))
            appender.AddInputData(sutil.read_surf_vtk(fhippo_lh_refined))
            appender.AddInputData(sutil.read_surf_vtk(fvent_refined))
            appender.Update()

            deepGM_surf = appender.GetOutput()
            sutil.write_surf_vtk(deepGM_surf, fsubcortseg_vtk)
    else:
        print('Subcortical Segmentation already performed: Skipping')

    return meshDir, voxelDir


def segment(procDir, subID, preproc_suffix, shell_suffix, cpu_num=0):
    """
    Subcortical brain segmentation with 3D surface based deformation using DTI images/maps

    Parameters
    ----------
    procDir: string
        parent director containing subject
    subID: string
        the microbrain subject to be processed
    preproc_suffix: string
        suffix detailing what preprocessing has been performed
    shell_suffix: string
        suffix detailing what shells were used when running microbrain

    Optional Parameters
    -------------------
    cpu_num: integer
        number of threads to use for computationally intense tasks

    Returns
    -------
    none
    """

    subDir = procDir + '/' + subID
    segDir = subDir + '/subcortical_segmentation/'
    regDir = subDir + '/registration/'

    if preproc_suffix == '':
        suffix = '_' + shell_suffix
        fdwi = subDir + '/meanDWI/' + subID + '_mean_b' + \
            shell_suffix.split('b')[-1] + '_n4' + fsl_ext()
    else:
        suffix = '_' + preproc_suffix + '_' + shell_suffix
        fdwi = subDir + '/meanDWI/' + subID + '_' + \
            preproc_suffix + '_mean_b' + \
            shell_suffix.split('b')[-1] + '_n4' + fsl_ext()

    ffa = subDir + '/DTI_maps/' + subID + suffix + '_FA' + fsl_ext()
    fmd = subDir + '/DTI_maps/' + subID + suffix + '_MD' + fsl_ext()

    # Register probability maps
    ftemplate = environ['FSLDIR'] + \
        '/data/standard/FSL_HCP1065_FA_1mm.nii.gz'

    fharvard = environ['FSLDIR'] + \
        '/data/atlases/HarvardOxford/HarvardOxford-sub-prob-1mm.nii.gz'
    fharvard_native = register_probatlas_to_native(
        ffa, ftemplate, fharvard, regDir, cpu_num=cpu_num)

    # Parent Directory to store all output
    if not path.exists(segDir):
        system('mkdir ' + segDir)

    # Directory to store initial meshes
    initSegDir = segDir + 'structure_initialization/'
    if not path.exists(initSegDir):
        system('mkdir ' + initSegDir)

    initial_voxel_labels_from_harvard(
        fharvard_native, subID, initSegDir, md_file=fmd, fa_file=ffa)

    meshDir, voxelDir, = deform_subcortical_surfaces(
        fdwi, ffa, fmd, fharvard_native, segDir, initSegDir, subID, cpu_num=cpu_num)

    return meshDir, voxelDir
