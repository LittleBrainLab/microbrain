#!/usr/local/bin/python
import itk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
import vtk
from shutil import which
import mbrain_preproc as mbrain_preproc
import surf_util as sutil
import os
import sys
import shutil
import glob
from os import system
from os import path
from time import time
import subprocess
import numpy as np
import nibabel as nib

from skimage.morphology import binary_erosion, binary_dilation
from dipy.align.reslice import reslice
from dipy.io import read_bvals_bvecs

sys.path.append('../utils/')

sys.path.append('../preprocessing/')


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

# Combined indices for MicroBrain
LEFT_HIPPOAMYG_IDX = 88
LEFT_STRIATUM_IDX = 66
RIGHT_HIPPOAMYG_IDX = 188
RIGHT_STRIATUM_IDX = 166

# mirtk docker command
mirtk_cmd = 'mirtk'


# Check to see if program is installed to path and executable before running subprocess
# Ditto as below
def is_tool(name):
    return which(name) is not None

# Define this as a common util (I think I need a mbrain_io.py)


def fsl_ext():
    fsl_extension = ''
    if os.environ['FSLOUTPUTTYPE'] == 'NIFTI':
        fsl_extension = '.nii'
    elif os.environ['FSLOUTPUTTYPE'] == 'NIFTI_GZ':
        fsl_extension = '.nii.gz'
    return fsl_extension

# Legacy (originally used to segment hippo/amygdala) Move/delete this


def vertex_PCA(A):
    # define a matrix
    # calculate the mean of each column
    M = np.mean(A.T, axis=1)

    # center columns by subtracting column means
    C = A - M

    # calculate covariance matrix of centered matrix
    V = np.cov(C.T)

    # eigendecomposition of covariance matrix
    values, vectors = np.linalg.eig(V)

    # project data
    proj_A = vectors.T.dot(C.T)

    return values, vectors, proj_A

# TODO: This should return a list of transformed files and be more general (probably should move to its own file)



def register_prob_maps_ants(fsource, ftemplate, fmask, fgm, fwm, fcsf, fmni, fharvard, regDir):

    if not os.path.exists(regDir):
        os.system('mkdir ' + regDir)

    method_suffix = '_ANTsReg_native'
    ftempbasename = os.path.basename(ftemplate)
    ftemplate_out = regDir + \
        ftempbasename.replace('.nii.gz', method_suffix + fsl_ext())

    fgm_basename = os.path.basename(fgm)
    fgm_out = regDir + fgm_basename.replace('.nii', method_suffix + fsl_ext())

    fwm_basename = os.path.basename(fwm)
    fwm_out = regDir + fwm_basename.replace('.nii', method_suffix + fsl_ext())

    fcsf_basename = os.path.basename(fcsf)
    fcsf_out = regDir + \
        fcsf_basename.replace('.nii', method_suffix + fsl_ext())

    fmni_basename = os.path.basename(fmni)
    fmni_out = regDir + \
        fmni_basename.replace('.nii.gz', method_suffix + fsl_ext())

    fharvard_basename = os.path.basename(fharvard)
    fharvard_out = regDir + \
        fharvard_basename.replace('.nii.gz', method_suffix + fsl_ext())

    ftransform_out = regDir + 'ants_native2mni_'
    if not os.path.exists(fharvard_out):
        print('Running Ants Registration')

        # Make template mask
        ftemplate_mask = regDir + \
            ftempbasename.replace('.nii.gz', '_mask' + fsl_ext())
        system('fslmaths ' + ftemplate + ' -bin ' + ftemplate_mask)

        system('antsRegistrationSyN.sh' +
               ' -d 3' +
               ' -f ' + ftemplate +
               ' -m ' + fsource +
               ' -o ' + ftransform_out +
               ' -x ' + ftemplate_mask +
               ' -n 8 ')

        print('Warping probabilistic atlases to native space')

        system('antsApplyTransforms' +
               ' -t [' + ftransform_out + '0GenericAffine.mat,1] ' +
               ' -t ' + ftransform_out + '1InverseWarp.nii.gz ' +
               ' -r ' + fsource +
               ' -i ' + fgm +
               ' -o ' + fgm_out)

        system('antsApplyTransforms' +
               ' -t [' + ftransform_out + '0GenericAffine.mat,1] ' +
               ' -t ' + ftransform_out + '1InverseWarp.nii.gz ' +
               ' -r ' + fsource +
               ' -i ' + fwm +
               ' -o ' + fwm_out)

        system('antsApplyTransforms' +
               ' -t [' + ftransform_out + '0GenericAffine.mat,1] ' +
               ' -t ' + ftransform_out + '1InverseWarp.nii.gz ' +
               ' -r ' + fsource +
               ' -i ' + fcsf +
               ' -o ' + fcsf_out)

        system('antsApplyTransforms' +
               ' -t [' + ftransform_out + '0GenericAffine.mat,1] ' +
               ' -t ' + ftransform_out + '1InverseWarp.nii.gz ' +
               ' -r ' + fsource +
               ' -e 3 ' +
               ' -i ' + fmni +
               ' -o ' + fmni_out)

        system('antsApplyTransforms' +
               ' -t [' + ftransform_out + '0GenericAffine.mat,1] ' +
               ' -t ' + ftransform_out + '1InverseWarp.nii.gz ' +
               ' -r ' + fsource +
               ' -e 3 ' +
               ' -i ' + fharvard +
               ' -o ' + fharvard_out)
    else:
        print("ANTs nonlinear registeration of atlases already performed")

    return fgm_out, fwm_out, fcsf_out, fmni_out, fharvard_out


def initial_voxel_labels(subID, segDir, fharvard, md_file=None, md_thresh=0.0015, fa_file=None, fa_thresh=0.5):

    # Make output directory
    initSegDir = segDir + 'structure_initialization/'
    if not os.path.exists(initSegDir):
        os.system('mkdir ' + initSegDir)

    if md_file:
        md_data = nib.load(md_file).get_data()

    if fa_file:
        fa_data = nib.load(fa_file).get_data()

    finitlabels_prefix = initSegDir + subID + '_initialization'

    harvard_img = nib.load(fharvard)
    harvard_data = harvard_img.get_data()

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
                   RIGHT_STRIATUM_IDX]

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
                     'RIGHT_STRIATUM']

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
        else:
            tmplabel[binary_erosion(harvard_data[:, :, :, sind] > 50)] = 1

        # remove large MD voxels if argument provided
        if md_file:
            tmplabel[md_data > md_thresh] = 0

        if fa_file:
            tmplabel[fa_data > fa_thresh] = 0

        finit_tmplabel = finitlabels_prefix + '_' + slabel + fsl_ext()
        nib.save(nib.Nifti1Image(tmplabel, harvard_img.affine), finit_tmplabel)
        os.system('mirtk extract-connected-components ' +
                  finit_tmplabel + ' ' + finit_tmplabel)

        # Extract surfaces
        finit_surf = finitlabels_prefix + '_' + slabel + '.vtk'
        os.system('mirtk extract-surface ' + finit_tmplabel +
                  ' ' + finit_surf + ' -isovalue 0.5')
        os.system('mirtk extract-connected-points ' +
                  finit_surf + ' ' + finit_surf)
        os.system('mirtk smooth-surface ' + finit_surf + ' ' +
                  finit_surf + ' -iterations 50 -lambda 0.05')

        # Label Surfaces
        os.system('mirtk project-onto-surface ' + finit_surf + ' ' + finit_surf +
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

    return initSegDir

# move to surf_util.py


def surf_to_volume_mask(fdwi, fmesh, inside_val, fout):
    # Transform the vtk mesh to native space
    surfVTK = sutil.read_surf_vtk(fmesh)
    vtkImage, vtkHeader, sformMat = sutil.read_image(fdwi)
    sformInv = vtk.vtkTransform()
    sformInv.DeepCopy(sformMat)
    sformInv.Inverse()
    sformInv.Update()

    transformPD = vtk.vtkTransformPolyDataFilter()
    transformPD.SetTransform(sformInv)
    transformPD.SetInputData(surfVTK)
    transformPD.Update()
    surfVTK_voxel = transformPD.GetOutput()

    # Fill image with inside_val
    vtkImage.GetPointData().GetScalars().Fill(inside_val)

    pol2stenc = vtk.vtkPolyDataToImageStencil()
    pol2stenc.SetInputData(surfVTK_voxel)
    pol2stenc.SetOutputOrigin(vtkImage.GetOrigin())
    pol2stenc.SetOutputSpacing(vtkImage.GetSpacing())
    pol2stenc.SetOutputWholeExtent(vtkImage.GetExtent())
    pol2stenc.Update()

    imgstenc = vtk.vtkImageStencil()
    imgstenc.SetInputData(vtkImage)
    imgstenc.SetStencilConnection(pol2stenc.GetOutputPort())
    imgstenc.ReverseStencilOff()
    imgstenc.SetBackgroundValue(0)
    imgstenc.Update()

    out_image = imgstenc.GetOutput()

    sutil.write_image(out_image, vtkHeader, sformMat, fout)

    return


def extract_surfaces_from_labels(flabels, label_list, outDir, fout_prefix):

    # Read in labels output nifti images
    label_img = nib.load(flabels)
    label_data = label_img.get_data()

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
        os.system('mirtk extract-surface ' + ftmpNII +
                  ' ' + fout_surf + ' -isovalue 0.5')

        # Smooth surface for the label
        os.system('mirtk smooth-surface ' + fout_surf + ' ' +
                  fout_surf + ' -iterations 50 -lambda 0.2')

        # Add label to surface mesh
        os.system('mirtk project-onto-surface ' + fout_surf + ' ' + fout_surf +
                  ' -constant ' + str(label_ind) + ' -pointdata -name struct_label')
        fout_list = fout_list + [fout_surf]

    return fout_list


def deform_subcortical_surfaces(fdwi, ffa, fmd, fharvard_native, segDir, initSegDir, subID, cpu_num=0):
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
    if not os.path.exists(MDFA_Dir):
        os.system('mkdir ' + MDFA_Dir)

    probForceDir = segDir + 'probability_force_maps/'
    if not os.path.exists(probForceDir):
        os.system('mkdir ' + probForceDir)

    meshDir = segDir + 'mesh_output/'
    if not os.path.exists(meshDir):
        os.system('mkdir ' + meshDir)

    voxelDir = segDir + 'voxel_output/'
    if not os.path.exists(voxelDir):
        os.system('mkdir ' + voxelDir)

    atroposDir = segDir + 'atropos_hippoamyg_seg/'
    if not os.path.exists(atroposDir):
        os.system('mkdir ' + atroposDir)

    # Make composite map of FA + csf probabilities. (This map defines the borders of the caudate and the thalamus)
    fmd_plusFA = MDFA_Dir + \
        os.path.basename(fmd.replace(fsl_ext(), '_plusFA' + fsl_ext()))
    os.system('fslmaths ' + fmd + ' -mul 1000 -add ' + ffa + ' ' + fmd_plusFA)
    MD_plusFA_data = nib.load(fmd_plusFA).get_data()

    fa_img = nib.load(ffa)

    harvard_img = nib.load(fharvard_native)
    harvard_data = harvard_img.get_data()

    dwi_img = nib.load(fdwi)
    dwi_data = dwi_img.get_data()

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
    nib.save(nib.Nifti1Image(glob_prob_force, fa_img.affine), fglob_prob_force)

    # Deform globus pallidus based on meanDWI and resticting movement into high FA regions
    fglobus_lh = initSegDir + subID + initial_seg_prefix + '_LEFT_GLOBUS.vtk'
    fglobus_lh_refined = meshDir + subID + seg_prefix + '_LEFT_GLOBUS.vtk'
    if not os.path.exists(fglobus_lh_refined):
        os.system('mirtk deform-mesh ' + fglobus_lh + ' ' + fglobus_lh_refined + ' -image ' + fdwi + ' -edge-distance 1.0 -edge-distance-averaging ' + averages + ' -edge-distance-smoothing 1 -edge-distance-median 1 -distance-image ' + fglob_prob_force + ' -distance 0.5 -distance-smoothing 1 -distance-averaging ' + averages + ' -distance-measure normal -optimizer EulerMethod -step ' + str(step_size) + ' -steps ' + str(
            step_num) + ' -epsilon 1e-6 -delta 0.001 -min-active 1% -reset-status -nointersection -fast-collision-test -min-width 0.01 -min-distance 0.01 -repulsion 4.0 -repulsion-distance 0.5 -repulsion-width 2.0 -curvature ' + str(curv_w) + ' -gauss-curvature ' + str(gcurv_w) + ' -edge-distance-type ClosestMaximum' + cpu_str + '-ascii -remesh 1 -min-edge-length ' + str(min_edgelength) + ' -max-edge-length ' + str(max_edgelength))
        surf_to_volume_mask(fdwi, fglobus_lh_refined, 1,
                            voxelDir + subID + seg_prefix + '_LEFT_GLOBUS' + fsl_ext())

    fglobus_rh = initSegDir + subID + initial_seg_prefix + '_RIGHT_GLOBUS.vtk'
    fglobus_rh_refined = meshDir + subID + seg_prefix + '_RIGHT_GLOBUS.vtk'
    if not os.path.exists(fglobus_rh_refined):
        os.system('mirtk deform-mesh ' + fglobus_rh + ' ' + fglobus_rh_refined + ' -image ' + fdwi + ' -edge-distance 1.0 -edge-distance-averaging ' + averages + ' -edge-distance-smoothing 1 -edge-distance-median 1 -distance-image ' + fglob_prob_force + ' -distance 0.5 -distance-smoothing 1 -distance-averaging ' + averages + ' -distance-measure normal -optimizer EulerMethod -step ' + str(step_size) + ' -steps ' + str(step_num) +
                  ' -epsilon 1e-6 -delta 0.001 -min-active 1% -reset-status -nointersection -fast-collision-test -min-width 0.01 -min-distance 0.01 -repulsion 4.0 -repulsion-distance 0.5 -repulsion-width 2.0 -curvature ' + str(curv_w) + ' -gauss-curvature ' + str(gcurv_w) + ' -edge-distance-type ClosestMaximum' + cpu_str + '-ascii -remesh 1 -min-edge-length ' + str(min_edgelength) + ' -max-edge-length ' + str(max_edgelength))
        surf_to_volume_mask(fdwi, fglobus_rh_refined, 1,
                            voxelDir + subID + seg_prefix + '_RIGHT_GLOBUS' + fsl_ext())

    # Generate map for Striatum deformation
    fa_globus = np.zeros(MD_plusFA_data.shape)
    fa_globus[:] = MD_plusFA_data[:]

    lh_globus_data = nib.load(
        voxelDir + subID + seg_prefix + '_LEFT_GLOBUS' + fsl_ext()).get_data()
    rh_globus_data = nib.load(
        voxelDir + subID + seg_prefix + '_RIGHT_GLOBUS' + fsl_ext()).get_data()
    fa_globus[lh_globus_data == 1] = 2
    fa_globus[rh_globus_data == 1] = 2
    fmd_plusFA_globus = MDFA_Dir + \
        os.path.basename(fmd_plusFA.replace(fsl_ext(), '_globus' + fsl_ext()))
    if not os.path.exists(fmd_plusFA_globus):
        nib.save(nib.Nifti1Image(fa_globus, fa_img.affine), fmd_plusFA_globus)

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
    if not os.path.exists(fstriatum_prob_force):
        nib.save(nib.Nifti1Image(striatum_prob_force,
                                 fa_img.affine), fstriatum_prob_force)

    fstriatum_lh = initSegDir + subID + initial_seg_prefix + '_LEFT_STRIATUM.vtk'
    fstriatum_lh_refined = meshDir + subID + seg_prefix + '_LEFT_STRIATUM.vtk'
    if not os.path.exists(fstriatum_lh_refined):
        os.system('mirtk deform-mesh ' + fstriatum_lh + ' ' + fstriatum_lh_refined + ' -image ' + fmd_plusFA_globus + ' -edge-distance 1.0 -edge-distance-averaging ' + averages + ' -edge-distance-smoothing 1 -edge-distance-median 1  -distance-image ' + fstriatum_prob_force + ' -distance 0.25 -distance-smoothing 1 -distance-averaging ' + averages + ' -distance-measure normal -optimizer EulerMethod -step ' + str(step_size) + ' -steps ' + str(step_num) +
                  ' -epsilon 1e-6 -delta 0.001 -min-active 1% -reset-status -nointersection -fast-collision-test -min-width 0.01 -min-distance 0.01 -repulsion 4.0 -repulsion-distance 0.5 -repulsion-width 2.0 -curvature ' + str(curv_w) + ' -gauss-curvature ' + str(gcurv_w) + ' -edge-distance-type ClosestMaximum' + cpu_str + '-ascii -remesh 1 -min-edge-length ' + str(min_edgelength) + ' -max-edge-length ' + str(max_edgelength) + ' -edge-distance-min-intensity 0.3')
        surf_to_volume_mask(fdwi, fstriatum_lh_refined, 1,
                            voxelDir + subID + seg_prefix + '_LEFT_STRIATUM' + fsl_ext())

    fstriatum_rh = initSegDir + subID + initial_seg_prefix + '_RIGHT_STRIATUM.vtk'
    fstriatum_rh_refined = meshDir + subID + seg_prefix + '_RIGHT_STRIATUM.vtk'
    if not os.path.exists(fstriatum_rh_refined):
        os.system('mirtk deform-mesh ' + fstriatum_rh + ' ' + fstriatum_rh_refined + ' -image ' + fmd_plusFA_globus + ' -edge-distance 1.0 -edge-distance-averaging ' + averages + ' -edge-distance-smoothing 1 -edge-distance-median 1  -distance-image ' + fstriatum_prob_force + ' -distance 0.25 -distance-smoothing 1 -distance-averaging ' + averages + ' -distance-measure normal -optimizer EulerMethod -step ' + str(step_size) + ' -steps ' + str(step_num) +
                  ' -epsilon 1e-6 -delta 0.001 -min-active 1% -reset-status -nointersection -fast-collision-test -min-width 0.01 -min-distance 0.01 -repulsion 4.0 -repulsion-distance 0.5 -repulsion-width 2.0 -curvature ' + str(curv_w) + ' -gauss-curvature ' + str(gcurv_w) + ' -edge-distance-type ClosestMaximum' + cpu_str + '-ascii -remesh 1 -min-edge-length ' + str(min_edgelength) + ' -max-edge-length ' + str(max_edgelength) + ' -edge-distance-min-intensity 0.3')
        surf_to_volume_mask(fdwi, fstriatum_rh_refined, 1,
                            voxelDir + subID + seg_prefix + '_RIGHT_STRIATUM' + fsl_ext())

    fa_md_striatum = np.zeros(MD_plusFA_data.shape)
    fa_md_striatum[:] = MD_plusFA_data[:]

    lh_striatum_data = nib.load(
        voxelDir + subID + seg_prefix + '_LEFT_STRIATUM' + fsl_ext()).get_data()
    rh_striatum_data = nib.load(
        voxelDir + subID + seg_prefix + '_RIGHT_STRIATUM' + fsl_ext()).get_data()
    fa_md_striatum[lh_striatum_data == 1] = 2
    fa_md_striatum[rh_striatum_data == 1] = 2
    fmd_plusFA_striatum = MDFA_Dir + \
        os.path.basename(fmd_plusFA.replace(
            fsl_ext(), '_striatum' + fsl_ext()))

    if not os.path.exists(fmd_plusFA_striatum):
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
    if not os.path.exists(fthal_prob_force):
        nib.save(nib.Nifti1Image(thal_prob_force,
                 fa_img.affine), fthal_prob_force)

    fthalamus = initSegDir + subID + initial_seg_prefix + '_THALAMUS.vtk'
    fthalamus_refined = meshDir + subID + seg_prefix + '_THALAMUS.vtk'
    if not os.path.exists(fthalamus_refined):
        os.system('mirtk deform-mesh ' + fthalamus + ' ' + fthalamus_refined + ' -image ' + fmd_plusFA_striatum + ' -edge-distance 1.0 -edge-distance-averaging ' + averages + ' -edge-distance-smoothing 1 -edge-distance-median 1 -distance-image ' + fthal_prob_force + ' -distance 0.25 -distance-smoothing 1 -distance-averaging ' + averages + ' -distance-measure normal -optimizer EulerMethod -step ' + str(step_size) + ' -steps ' + str(step_num) +
                  ' -epsilon 1e-6 -delta 0.001 -min-active 1% -reset-status -nointersection -fast-collision-test -min-width 0.01 -min-distance 0.01 -repulsion 4.0 -repulsion-distance 0.5 -repulsion-width 2.0 -curvature ' + str(curv_w) + ' -gauss-curvature ' + str(gcurv_w) + ' -edge-distance-type ClosestMaximum' + cpu_str + '-ascii -remesh 1 -min-edge-length ' + str(min_edgelength) + ' -max-edge-length ' + str(max_edgelength) + ' -edge-distance-min-intensity 1.25')

    thalamus_surf = sutil.read_surf_vtk(fthalamus_refined)
    [lh_thalamus, rh_thalamus] = sutil.split_surface_by_label(
        thalamus_surf, label=[LEFT_THAL_IDX, RIGHT_THAL_IDX], label_name='struct_label')

    fthalamus_lh_refined = meshDir + subID + seg_prefix + '_LEFT_THALAMUS.vtk'
    if not os.path.exists(fthalamus_lh_refined):
        sutil.write_surf_vtk(lh_thalamus, fthalamus_lh_refined)
        surf_to_volume_mask(fdwi, fthalamus_lh_refined, 1,
                            voxelDir + subID + seg_prefix + '_LEFT_THALAMUS' + fsl_ext())

    fthalamus_rh_refined = meshDir + subID + seg_prefix + '_RIGHT_THALAMUS.vtk'
    if not os.path.exists(fthalamus_rh_refined):
        sutil.write_surf_vtk(rh_thalamus, fthalamus_rh_refined)
        surf_to_volume_mask(fdwi, fthalamus_rh_refined, 1,
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
    if not os.path.exists(fhippoamyg_lh_refined):
        os.system('mirtk deform-mesh ' + fhippoamyg_lh + ' ' + fhippoamyg_lh_refined + ' -image ' + fmd_plusFA + ' -edge-distance 1.0 -edge-distance-averaging ' + averages + ' -edge-distance-smoothing 1 -edge-distance-median 1 -distance-image ' + fhipamyg_prob_force + ' -distance 0.25 -distance-smoothing 1 -distance-averaging ' + averages + ' -distance-measure normal -optimizer EulerMethod -step ' + str(step_size) + ' -steps ' + str(step_num) +
                  ' -epsilon 1e-6 -delta 0.001 -min-active 1% -reset-status -nointersection -fast-collision-test -min-width 0.01 -min-distance 0.01 -repulsion 4.0 -repulsion-distance 0.5 -repulsion-width 2.0 -curvature ' + str(curv_w) + ' -gauss-curvature ' + str(gcurv_w) + ' -edge-distance-type ClosestMaximum' + cpu_str + '-ascii -remesh 1 -min-edge-length ' + str(min_hippo_edgelength) + ' -max-edge-length ' + str(max_hippo_edgelength) + ' -edge-distance-min-intensity 1.0')
        surf_to_volume_mask(fdwi, fhippoamyg_lh_refined, 1,
                            atroposDir + subID + seg_prefix + '_LEFT_HIPPOAMYG' + fsl_ext())

    fhippoamyg_rh = initSegDir + subID + initial_seg_prefix + '_RIGHT_HIPPOAMYG.vtk'
    fhippoamyg_rh_refined = atroposDir + subID + seg_prefix + '_RIGHT_HIPPOAMYG.vtk'
    if not os.path.exists(fhippoamyg_rh_refined):
        os.system('mirtk deform-mesh ' + fhippoamyg_rh + ' ' + fhippoamyg_rh_refined + ' -image ' + fmd_plusFA + ' -edge-distance 1.0 -edge-distance-averaging ' + averages + ' -edge-distance-smoothing 1 -edge-distance-median 1 -distance-image ' + fhipamyg_prob_force + ' -distance 0.25 -distance-smoothing 1 -distance-averaging ' + averages + ' -distance-measure normal -optimizer EulerMethod -step ' + str(step_size) + ' -steps ' + str(step_num) +
                  ' -epsilon 1e-6 -delta 0.001 -min-active 1% -reset-status -nointersection -fast-collision-test -min-width 0.01 -min-distance 0.01 -repulsion 4.0 -repulsion-distance 0.5 -repulsion-width 2.0 -curvature ' + str(curv_w) + ' -gauss-curvature ' + str(gcurv_w) + ' -edge-distance-type ClosestMaximum' + cpu_str + '-ascii -remesh 1 -min-edge-length ' + str(min_hippo_edgelength) + ' -max-edge-length ' + str(max_hippo_edgelength) + ' -edge-distance-min-intensity 1.0')
        surf_to_volume_mask(fdwi, fhippoamyg_rh_refined, 1,
                            atroposDir + subID + seg_prefix + '_RIGHT_HIPPOAMYG' + fsl_ext())

    # Run 3 channel tissue segmentation to separate amygdala from hippocampus
    # prepare prob maps and separate primary eigenvector into separate nifti file
    atropos_prefix_lh = atroposDir + subID + \
        initial_seg_prefix + '_INITAMYGHIPPO_LH'
    atropos_prefix_lh_label = at, VP at Ubisoft’s Strategic Innovations Lab aoposDir + subID + \
        initial_seg_prefix + '_INITAMYGHIPPO_LH_LABELS' + fsl_ext()
    atropos_prefix_lh_probout = atroposDir + subID + \
        initial_seg_prefix + '_INITAMYGHIPPO_LH_PROBOUT_'

    famyg_lh_prob = atropos_prefix_lh + '_01' + fsl_ext()
    if not os.path.exists(famyg_lh_prob):
        nib.save(nib.Nifti1Image(
            harvard_data[:, :, :, LEFT_AMYG_IDX]/100, harvard_img.affine), famyg_lh_prob)

    fhippo_lh_prob = atropos_prefix_lh + '_02' + fsl_ext()
    if not os.path.exists(fhippo_lh_prob):
        nib.save(nib.Nifti1Image(
            harvard_data[:, :, :, LEFT_HIPPO_IDX]/100, harvard_img.affine), fhippo_lh_prob)

    fcortex_lh_prob = atropos_prefix_lh + '_03' + fsl_ext()
    if not os.path.exists(fcortex_lh_prob):
        nib.save(nib.Nifti1Image(
            harvard_data[:, :, :, LEFT_CORTEX_IDX]/100, harvard_img.affine), fcortex_lh_prob)

    atropos_prefix_rh = atroposDir + subID + \
        initial_seg_prefix + '_INITAMYGHIPPO_RH'
    atropos_prefix_rh_label = atroposDir + subID + \
        initial_seg_prefix + '_INITAMYGHIPPO_RH_LABELS' + fsl_ext()
    atropos_prefix_rh_probout = atroposDir + subID + \
        initial_seg_prefix + '_INITAMYGHIPPO_RH_PROBOUT_'

    famyg_rh_prob = atropos_prefix_rh + '_01' + fsl_ext()
    if not os.path.exists(famyg_rh_prob):
        nib.save(nib.Nifti1Image(
            harvard_data[:, :, :, RIGHT_AMYG_IDX]/100, harvard_img.affine), famyg_rh_prob)

    fhippo_rh_prob = atropos_prefix_rh + '_02' + fsl_ext()
    if not os.path.exists(fhippo_rh_prob):
        nib.save(nib.Nifti1Image(
            harvard_data[:, :, :, RIGHT_HIPPO_IDX]/100, harvard_img.affine), fhippo_rh_prob)

    fcortex_rh_prob = atropos_prefix_rh + '_03' + fsl_ext()
    if not os.path.exists(fcortex_rh_prob):
        nib.save(nib.Nifti1Image(
            harvard_data[:, :, :, RIGHT_CORTEX_IDX]/100, harvard_img.affine), fcortex_rh_prob)

    PriorWeight = 0.3
    if not os.path.exists(atropos_prefix_lh_label):
        system('Atropos' +
               ' -a [' + fmd_plusFA + ']' +
               ' -x ' + atr, VP at Ubisoft’s Strategic Innovations Lab aoposDir + subID + seg_prefix + '_LEFT_HIPPOAMYG' + fsl_ext() +
               ' -i PriorProbabilityImages[3, ' + atropos_prefix_lh + '_%02d' + fsl_ext() + ',' + str(PriorWeight) + ',0.0001]' +
               ' -m [0.3, 2x2x2] ' +
               ' --use-partial-volume-likelihoods false ' +
               ' -s 1x3 -s 1x2 ' +
               ' -o [' + atropos_prefix_lh_label + ',' + atropos_prefix_lh_probout + '%02d' + fsl_ext() + ']' +
               ' -k HistogramParzenWindows[1.0,32]' +
               ' -v 1')

    if not os.path.exists(atropos_prefix_rh_label):
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

    lh_labels = nib.load(atropos_prefix_lh_label).get_data()
    rh_labels = nib.load(atropos_prefix_rh_label).get_data()

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
        os.path.basename(fmd_plusFA.replace(
            fsl_ext(), '_hippocortex' + fsl_ext()))
    if not os.path.exists(fmd_plusFA_hippocortex):
        nib.save(nib.Nifti1Image(fa_md_hippocortex,
                 fa_img.affine), fmd_plusFA_hippocortex)

    famyg_lh_refined = meshDir + subID + seg_prefix + '_LEFT_AMYGDALA.vtk'
    if not os.path.exists(famyg_lh_refined):
        os.system('mirtk deform-mesh ' + famyg_lh_step1 + ' ' + famyg_lh_refined + ' -image ' + fmd_plusFA_hippocortex + ' -edge-distance 1.0 -edge-distance-averaging ' + averages + ' -edge-distance-smoothing 1 -edge-distance-median 1 -distance-image ' + famyg_prob_force + ' -distance 0.25 -distance-smoothing 1 -distance-averaging ' + averages + ' -distance-measure normal -optimizer EulerMethod -step ' + str(step_size) + ' -steps ' + str(step_num) +
                  ' -epsilon 1e-6 -delta 0.001 -min-active 1% -reset-status -nointersection -fast-collision-test -min-width 0.01 -min-distance 0.01 -repulsion 4.0 -repulsion-distance 0.5 -repulsion-width 2.0 -curvature ' + str(curv_w) + ' -gauss-curvature ' + str(gcurv_w) + ' -edge-distance-type ClosestMaximum' + cpu_str + '-ascii -remesh 1 -min-edge-length ' + str(min_hippo_edgelength) + ' -max-edge-length ' + str(max_hippo_edgelength) + ' -edge-distance-min-intensity 1.0')
        os.system('mirtk project-onto-surface ' + famyg_lh_refined + ' ' + famyg_lh_refined +
                  ' -constant ' + str(LEFT_AMYG_IDX) + ' -pointdata -name struct_label')
        surf_to_volume_mask(fdwi, famyg_lh_refined, 1,
                            voxelDir + subID + seg_prefix + '_LEFT_AMYGDALA' + fsl_ext())

    famyg_rh_refined = meshDir + subID + seg_prefix + '_RIGHT_AMYGDALA.vtk'
    if not os.path.exists(famyg_rh_refined):
        os.system('mirtk deform-mesh ' + famyg_rh_step1 + ' ' + famyg_rh_refined + ' -image ' + fmd_plusFA_hippocortex + ' -edge-distance 1.0 -edge-distance-averaging ' + averages + ' -edge-distance-smoothing 1 -edge-distance-median 1 -distance-image ' + famyg_prob_force + ' -distance 0.25 -distance-smoothing 1 -distance-averaging ' + averages + ' -distance-measure normal -optimizer EulerMethod -step ' + str(step_size) + ' -steps ' + str(step_num) +
                  ' -epsilon 1e-6 -delta 0.001 -min-active 1% -reset-status -nointersection -fast-collision-test -min-width 0.01 -min-distance 0.01 -repulsion 4.0 -repulsion-distance 0.5 -repulsion-width 2.0 -curvature ' + str(curv_w) + ' -gauss-curvature ' + str(gcurv_w) + ' -edge-distance-type ClosestMaximum' + cpu_str + '-ascii -remesh 1 -min-edge-length ' + str(min_hippo_edgelength) + ' -max-edge-length ' + str(max_hippo_edgelength) + ' -edge-distance-min-intensity 1.0')
        os.system('mirtk project-onto-surface ' + famyg_rh_refined + ' ' + famyg_rh_refined +
                  ' -constant ' + str(RIGHT_AMYG_IDX) + ' -pointdata -name struct_label')
        surf_to_volume_mask(fdwi, famyg_rh_refined, 1,
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
        os.path.basename(fmd_plusFA.replace(
            fsl_ext(), '_amyg' + fsl_ext()))
    if not os.path.exists(fmd_plusFA_amyg):
        nib.save(nib.Nifti1Image(fa_md_amyg,
                 fa_img.affine), fmd_plusFA_amyg)

    fhippo_lh_refined = meshDir + subID + seg_prefix + '_LEFT_HIPPO.vtk'
    if not os.path.exists(fhippo_lh_refined):
        os.system('mirtk deform-mesh ' + fhippo_lh_step1 + ' ' + fhippo_lh_refined + ' -image ' + fmd_plusFA_amyg + ' -edge-distance 1.0 -edge-distance-averaging ' + averages + ' -edge-distance-smoothing 1 -edge-distance-median 1 -distance-image ' + fhippo_prob_force + ' -distance 0.25 -distance-smoothing 1 -distance-averaging ' + averages + ' -distance-measure normal -optimizer EulerMethod -step ' + str(step_size) + ' -steps ' + str(step_num) +
                  ' -epsilon 1e-6 -delta 0.001 -min-active 1% -reset-status -nointersection -fast-collision-test -min-width 0.01 -min-distance 0.01 -repulsion 4.0 -repulsion-distance 0.5 -repulsion-width 2.0 -curvature ' + str(curv_w) + ' -gauss-curvature ' + str(gcurv_w) + ' -edge-distance-type ClosestMaximum' + cpu_str + '-ascii -remesh 1 -min-edge-length ' + str(min_hippo_edgelength) + ' -max-edge-length ' + str(max_hippo_edgelength) + ' -edge-distance-min-intensity 1.0')
        os.system('mirtk project-onto-surface ' + fhippo_lh_refined + ' ' + fhippo_lh_refined +
                  ' -constant ' + str(LEFT_HIPPO_IDX) + ' -pointdata -name struct_label')
        surf_to_volume_mask(fdwi, fhippo_lh_refined, 1,
                            voxelDir + subID + seg_prefix + '_LEFT_HIPPO' + fsl_ext())

    fhippo_rh_refined = meshDir + subID + seg_prefix + '_RIGHT_HIPPO.vtk'
    if not os.path.exists(fhippo_rh_refined):
        os.system('mirtk deform-mesh ' + fhippo_rh_step1 + ' ' + fhippo_rh_refined + ' -image ' + fmd_plusFA_amyg + ' -edge-distance 1.0 -edge-distance-averaging ' + averages + ' -edge-distance-smoothing 1 -edge-distance-median 1 -distance-image ' + fhippo_prob_force + ' -distance 0.25 -distance-smoothing 1 -distance-averaging ' + averages + ' -distance-measure normal -optimizer EulerMethod -step ' + str(step_size) + ' -steps ' + str(step_num) +
                  ' -epsilon 1e-6 -delta 0.001 -min-active 1% -reset-status -nointersection -fast-collision-test -min-width 0.01 -min-distance 0.01 -repulsion 4.0 -repulsion-distance 0.5 -repulsion-width 2.0 -curvature ' + str(curv_w) + ' -gauss-curvature ' + str(gcurv_w) + ' -edge-distance-type ClosestMaximum' + cpu_str + '-ascii -remesh 1 -min-edge-length ' + str(min_hippo_edgelength) + ' -max-edge-length ' + str(max_hippo_edgelength) + ' -edge-distance-min-intensity 1.0')
        os.system('mirtk project-onto-surface ' + fhippo_rh_refined + ' ' + fhippo_rh_refined +
                  ' -constant ' + str(RIGHT_HIPPO_IDX) + ' -pointdata -name struct_label')
        surf_to_volume_mask(fdwi, fhippo_rh_refined, 1,
                            voxelDir + subID + seg_prefix + '_RIGHT_HIPPO' + fsl_ext())

    # Append subcort surfs together into single vtk file
    fsubcortseg_vtk = meshDir + subID + seg_prefix + '_subcortGM.vtk'
    if not os.path.exists(fsubcortseg_vtk):
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
        appender.Update()

        deepGM_surf = appender.GetOutput()
        sutil.write_surf_vtk(deepGM_surf, fsubcortseg_vtk)

    return

# No longer needed move to it's own file
# def prepare_prob_maps_for_segmentation(fgm, fwm, fcsf, fmni, fharvard, regDir, tissue_base_prior, csf_vent_prior, cerebel_cereb_prior, cerebel_cereb_bs_prior):
#
#    first_tissue_prior_fname = regDir + tissue_base_prior + str(1).zfill(2) + fsl_ext()
#    if not os.path.exists(first_tissue_prior_fname):
#        gm_img = nib.load(fgm)
#        gm_data = gm_img.get_data()*100
#
#        wm_img = nib.load(fwm)
#        wm_data = wm_img.get_data()*100
#
#        csf_img = nib.load(fcsf)
#        csf_data = csf_img.get_data()*100
#
#        prob_comb = np.concatenate((gm_data[:,:,:,None], wm_data[:,:,:,None], csf_data[:,:,:,None]), axis=3)
#
#        # Model putamen and thalamus as independant gausians
#        harvard_img = nib.load(fharvard)
#        harvard_data = harvard_img.get_data()
#
#        shortT2_vent_prob = harvard_data[:,:,:,[3,4,5,6]] + harvard_data[:,:,:,[14,15,16,17]]
#        prob_comb = np.concatenate((prob_comb, shortT2_vent_prob), axis=3)
#
#        # Remove short T2 subcortical structures from GM probabilities
#        shortT2_mask = np.sum(prob_comb[:,:,:,[3,4,5,6]], axis=3) > 0
#        gm_data[shortT2_mask] = 0
#        prob_comb[:,:,:,0] = gm_data
#
#        # Remove Globus Pallidus and thalamus from WM probability
#        globus_thalmus_mask = np.sum(harvard_data[:,:,:,[6]] + harvard_data[:,:,:,[17]],axis=3)>10
#        wm_data[globus_thalmus_mask] = 0
#        prob_comb[:,:,:,1] = wm_data
#        for i in range(0,prob_comb.shape[3]):
#            nib.save(nib.Nifti1Image(prob_comb[:,:,:,i]/100, harvard_img.affine), regDir + tissue_base_prior + str(i+1).zfill(2) + fsl_ext())
#
#        # ventricle csf prior
#        subcort_mask = np.sum(harvard_data[:,:,:,[2,3,4,5,6]] + harvard_data[:,:,:,[13,14,15,16,17]], axis=3) > 0
#        csf_data[subcort_mask] = 0
#        vent_data = harvard_data[:,:,:,2] + harvard_data[:,:,:,13]
#        csf_prob_comb = np.concatenate((csf_data[:,:,:,None], vent_data[:,:,:,None]), axis=3)
#        for i in range(0,csf_prob_comb.shape[3]):
#            nib.save(nib.Nifti1Image(csf_prob_comb[:,:,:,i]/100, harvard_img.affine), regDir + csf_vent_prior + str(i+1).zfill(2) + fsl_ext())
#
#        # cerebellum / cerebrum / sub cortical structures for GM reclassification
#        mni_img = nib.load(fmni)
#        mni_data = mni_img.get_data()
#
#        cereb_data = np.sum(harvard_data[:,:,:,[0,1,2,3,5,6,8,9,10]] + harvard_data[:,:,:,[11,12,13,14,16,17,18,19,20]],axis=3)
#        #caud_mask = harvard_data[:,:,:,4] + harvard_data[:,:,:,15] > 0
#        cereb_data[subcort_mask>0] = 0
#        #caudate_data = harvard_data[:,:,:,4] + harvard_data[:,:,:,15]
#        cerebellum_data = mni_data[:,:,:,1]
#        cerebel_cereb_prob_comb = np.concatenate((cerebellum_data[:,:,:,None], cereb_data[:,:,:,None], shortT2_vent_prob), axis=3)
#        for i in range(0,cerebel_cereb_prob_comb.shape[3]):
#            nib.save(nib.Nifti1Image(cerebel_cereb_prob_comb[:,:,:,i]/100, harvard_img.affine), regDir + cerebel_cereb_prior + str(i+1).zfill(2) + fsl_ext())
#
#        # cerebellum / cerebrum / bs
#        cereb_data = np.sum(harvard_data[:,:,:,[0,1,2,3,4,5,6,8,9,10]] + harvard_data[:,:,:,[11,12,13,14,15,16,17,18,19,20]],axis=3)
#        bs_data = harvard_data[:,:,:,7]
#        cerebel_cereb_bs_prob_comb = np.concatenate((cerebellum_data[:,:,:,None], cereb_data[:,:,:,None], bs_data[:,:,:,None]), axis=3)
#
#        for i in range(0,cerebel_cereb_bs_prob_comb.shape[3]):
#            nib.save(nib.Nifti1Image(cerebel_cereb_bs_prob_comb[:,:,:,i]/100, harvard_img.affine), regDir + cerebel_cereb_bs_prior + str(i+1).zfill(2) + fsl_ext())
#    else:
#        print("Initial Tissue Prior Maps already generated")
#
#    return

# Multi channel tissue segmentation no longer used (move to cortex surf segmentation)
# def three_channel_tissue_classifier(ffa, fmd, fdwi, fmask, fgm_native, fwm_native, fcsf_native, segDir,tissue_prob_suffix, PriorWeight=0.1):
#    tissue_nclasses = 3
#
#    fmd_plusFA = fmd.replace(fsl_ext(),'_plusFA' + fsl_ext())
#    os.system('fslmaths ' + fmd + ' -mul 1000 -add ' + ffa + ' ' + fmd_plusFA)
#
#    fmd_mul = fmd.replace(fsl_ext(),'_scale1000' + fsl_ext())
#    os.system('fslmaths ' + fmd + ' -mul 1000 ' + fmd_mul)
#
#    print("Starting Tissue Segmentation")
#    if not os.path.exists(segDir):
#            os.system('mkdir ' + segDir)
#
#    tissue_base_name = segDir + 'tissue_prob_' + tissue_prob_suffix + '_'
#    ftissuelabels = tissue_base_name + 'out_seg' + fsl_ext()
#    ftissueprobs_base = tissue_base_name + 'out_prob_'
#    if not os.path.exists(ftissuelabels):
#        # Copy tissue probability maps to base tissue fname
#        fprob_list = [fgm_native, fwm_native, fcsf_native]
#        for i in range(0,tissue_nclasses):
#            os.system('cp ' + fprob_list[i] + ' ' + tissue_base_name + str(i+1).zfill(2) + fsl_ext())
#            process = subprocess.run(['fslcpgeom', fdwi, tissue_base_name + str(i+1).zfill(2) + fsl_ext()], stdout=subprocess.PIPE, universal_newlines=True)
#
#        process = subprocess.run(['fslcpgeom', fdwi, ffa], stdout=subprocess.PIPE, universal_newlines=True)
#        process = subprocess.run(['fslcpgeom', fdwi, fmask], stdout=subprocess.PIPE, universal_newlines=True)
#        system('Atropos' +
#            ' -a [' + ffa + ']' +
#            ' -a [' + fdwi + ']' +
#            ' -x ' + fmask +
#            ' -i PriorProbabilityImages[' + str(tissue_nclasses) + ', ' + tissue_base_name + '%02d' + fsl_ext() + ',' + str(PriorWeight) + ',0.0001]' +
#            ' -m [0.3, 2x2x2] ' +
#            ' -s 1x2 -s 1x3 -s 2x3 ' +
#            ' --use-partial-volume-likelihoods false ' +
#            ' -o [' + ftissuelabels + ',' + ftissueprobs_base + '%02d' + fsl_ext() + ']' +
#            ' -k HistogramParzenWindows[1.0,32]')
#
#        # remove files used for input to Atropos
#        for i in range(0,tissue_nclasses):
#            os.system('rm ' + tissue_base_name + str(i+1).zfill(2) + fsl_ext())
#    else:
#        print("Tissue segmentation already completed")
#
#    fseg_out = ftissuelabels
#    fgm_prob_out = ftissueprobs_base + '01' + fsl_ext()
#    fwm_prob_out = ftissueprobs_base + '02' + fsl_ext()
#    fcsf_prob_out = ftissueprobs_base + '03' + fsl_ext()
#
#    return fseg_out, fgm_prob_out, fwm_prob_out, fcsf_prob_out

# Originally used by multichannel tissue classification (no longer needed move to preprocessing)
# def N4_correction(finput, fmask, fwm_prob, fdiff, bvals, shells, itr_str, segDir, tolerance = 100):
#
#    print("N4 correction (ANTS)")
#    t = time()
#
#    if not os.path.exists(segDir):
#        os.system('mkdir ' + segDir)
#
#    finput_basename = os.path.basename(finput)
#    fn4correct = segDir + finput_basename.replace(fsl_ext(), '_n4_' + itr_str + fsl_ext())
#    fbias = segDir + finput_basename.replace(fsl_ext(), '_n4bias_' + itr_str + fsl_ext())
#
#    if not os.path.exists(fn4correct):
#        if is_tool('N4BiasFieldCorrection'):
#                process = subprocess.run(['fslcpgeom', finput, fmask], stdout=subprocess.PIPE, universal_newlines=True)
#                n4command = ['N4BiasFieldCorrection',
#                                '-i', finput,
#                                '-x', fmask,
#                                '-o', '[' + fn4correct + ',' + fbias + ']','-v']
#                if fwm_prob:
#                    process = subprocess.run(['fslcpgeom', finput, fwm_prob], stdout=subprocess.PIPE, universal_newlines=True)
#                    n4command.append('-w')
#                    n4command.append(fwm_prob)
#
#                process = subprocess.run(n4command,
#                            stdout=subprocess.PIPE,
#                            universal_newlines=True)
#                print("N4 finished, Processing Time: ", time() - t)
#                stdout = process.stdout
#                return_code = process.returncode
#        else:
#            print("DSurfer Preproc: Could not find N4, make sure it is installed to your path")
#            stdout = ''
#            return_code = 1
#    else:
#        print("N4 Correction previously done: Skipping")
#        stdout = ''
#        return_code = 0
#
#
#    fdiff_basename = os.path.basename(fdiff)
#
#    for shell in shells:
#        fmeanshell_n4 = segDir + fdiff_basename.replace(fsl_ext(),'_mean_b' + str(shell) + '_n4_' + itr_str + fsl_ext())
#        if shell == 0:
#            fb0_n4 = fmeanshell_n4
#
#        if shell == shells[1]:
#            fdwi_n4 = fmeanshell_n4
#
#        if not os.path.exists(fmeanshell_n4):
#            diff_img = nib.load(fdiff)
#            diff_data = diff_img.get_data()
#
#            biasField = nib.load(fbias).get_data()
#            mask = nib.load(fmask).get_data()
#
#            bshell_idx = np.squeeze(np.array(np.where(np.logical_and(bvals < shell + tolerance, bvals > shell - tolerance))))
#            bshell_data = diff_data[...,bshell_idx]
#
#            meanBSHELL_n4 = np.mean(bshell_data,axis=3) / biasField
#            meanBSHELL_n4[mask == 0] = 0
#            nib.save(nib.Nifti1Image(meanBSHELL_n4, diff_img.affine), fmeanshell_n4)
#
#
#    return fn4correct,fbias, fb0_n4, fdwi_n4


# Orignally needed this, now eliminated from algorithm (move to cortex segmentation)
# def multichannel_tissue_classifcation(ffirstb0, ffa, fmd, fdiff, fmask, fgm_native, fwm_native, fcsf_native, fharvard_native, bvals, shells, tissueDir, max_itr= 25, tol = 0.001):
#    fseg_final = tissueDir + 'tissue_prob_final_out_seg' + fsl_ext()
#    fgm_prob_final = tissueDir + 'tissue_prob_final_out_prob_GM' + fsl_ext()
#    fwm_prob_final = tissueDir + 'tissue_prob_final_out_prob_WM' + fsl_ext()
#    fcsf_prob_final = tissueDir + 'tissue_prob_final_out_prob_CSF' + fsl_ext()
#    fdwi_final = tissueDir + os.path.basename(fdiff).replace(fsl_ext(), '_mean_b1000_n4_final' + fsl_ext())
#    if not os.path.exists(fseg_final):
#        # initial n4 correction
#        fn4_out, fbias_out, fb0avg_out, fdwi_out = N4_correction(ffirstb0, fmask, fwm_native, fdiff, bvals, shells, 'init', tissueDir)
#
#        # Initial tissue segmentation, using FSL atlas probability maps
#        fseg_out, fgm_prob_out, fwm_prob_out, fcsf_prob_out = three_channel_tissue_classifier(ffa, fmd, fdwi_out, fmask, fgm_native, fwm_native, fcsf_native, tissueDir,'init', PriorWeight=0.1)
#
#        # Use new WM probability map for N4 correction
#        fn4_out, fbias_out, fb0avg_out, fdwi_out = N4_correction(ffirstb0, fmask, fwm_prob_out, fdiff, bvals, shells, 'refined', tissueDir)
#
#        # Use new N4 corrected DWI for refined tissue classification
#        fseg_out, fgm_prob_out, fwm_prob_out, fcsf_prob_out = three_channel_tissue_classifier(ffa, fmd, fdwi_out, fmask, fgm_prob_out, fwm_prob_out, fcsf_prob_out, tissueDir,'refined', PriorWeight=0.1)
#
#        # Final names for output
#        fn4_final = fn4_out.replace('refined', 'final')
#        fbias_final = fbias_out.replace('refined', 'final')
#        fb0avg_final = fb0avg_out.replace('refined', 'final')
#        fdwi_final = fdwi_out.replace('refined', 'final')
#        fseg_final = fseg_out.replace('refined', 'final')
#        fgm_prob_final = fgm_prob_out.replace('refined', 'final').replace('01' + fsl_ext(), 'GM' + fsl_ext())
#        fwm_prob_final = fwm_prob_out.replace('refined', 'final').replace('02' + fsl_ext(), 'WM' + fsl_ext())
#        fcsf_prob_final = fcsf_prob_out.replace('refined', 'final').replace('03' + fsl_ext(), 'CSF' + fsl_ext())
#
#        os.system('mv ' + fn4_out + ' ' + fn4_final)
#        os.system('mv ' + fbias_out + ' ' + fbias_final)
#        os.system('mv ' + fb0avg_out + ' ' + fb0avg_final)
#        os.system('mv ' + fdwi_out + ' ' + fdwi_final)
#        os.system('mv ' + fseg_out + ' ' + fseg_final)
#        os.system('mv ' + fgm_prob_out + ' ' + fgm_prob_final)
#        os.system('mv ' + fwm_prob_out + ' ' + fwm_prob_final)
#        os.system('mv ' + fcsf_prob_out + ' ' + fcsf_prob_final)
#    else:
#        print("Iterative Tissue Classification previously done: Skipping")
#
#    return fdwi_final, fseg_final, fgm_prob_final, fwm_prob_final, fcsf_prob_final

def segment(fmask, procDir, subID, preproc_suffix, shell_suffix, shells, cpu_num=0):
    subDir = procDir + '/' + subID
    segDir = subDir + '/subcortical_segmentation/'
    regDir = subDir + '/registration/'

    if preproc_suffix == '':
        suffix = '_' + shell_suffix
    else:
        suffix = '_' + preproc_suffix + '_' + shell_suffix

    fdwi = subDir + '/meanDWI/' + subID + '_' + \
        preproc_suffix + '_mean_b1000_n4' + fsl_ext()
    ffa = subDir + '/DTI_maps/' + subID + suffix + '_FA' + fsl_ext()
    fmd = subDir + '/DTI_maps/' + subID + suffix + '_MD' + fsl_ext()

    # Register probability maps
    ftemplate = os.environ['FSLDIR'] + \
        '/data/standard/FSL_HCP1065_FA_1mm.nii.gz'

    fgm = '../Data/tissuepriors/avg152T1_gm_resampled.nii'
    fwm = '../Data/tissuepriors/avg152T1_wm_resampled.nii'
    fcsf = '../Data/tissuepriors/avg152T1_csf_resampled.nii'

    fmni = os.environ['FSLDIR'] + '/data/atlases/MNI/MNI-prob-1mm.nii.gz'
    fharvard = os.environ['FSLDIR'] + \
        '/data/atlases/HarvardOxford/HarvardOxford-sub-prob-1mm.nii.gz'
    fgm_native, fwm_native, fcsf_native, fmni_native, fharvard_native = register_prob_maps_ants(
        ffa, ftemplate, fmask, fgm, fwm, fcsf, fmni, fharvard, regDir)

    # Perform iterative tissue classification
    #fdwi, fseg_out, fgm_prob_out, fwm_prob_out, fcsf_prob_out = multichannel_tissue_classifcation(ffirstb0, ffa, fmd, fdiff, fmask, fgm_native, fwm_native, fcsf_native, fharvard_native, bvals, shells, tissueDir)

    # Perform subcortical segmentation
    if not os.path.exists(segDir):
        os.system('mkdir ' + segDir)
    initSegDir = initial_voxel_labels(
        subID, segDir, fharvard_native, md_file=fmd, fa_file=ffa)

    deform_subcortical_surfaces(
        fdwi, ffa, fmd, fharvard_native, segDir, initSegDir, subID, cpu_num=cpu_num)

    return

#sublist_fa = np.zeros((len(subList),8))
#sublist_fa_std = np.zeros((len(subList),8))
#sublist_md = np.zeros((len(subList),8))
#sublist_md_std = np.zeros((len(subList),8))
#sub_ind = 0

#    FA, FAstd, MD, MDstd = extract_FA_MD_from_subcortical_segmentations(subID, segDir, ffa, fmd)
#    sublist_fa[sub_ind,:] = FA
#    sublist_fa_std[sub_ind,:] = FAstd
#    sublist_md[sub_ind,:] = MD
#    sublist_md[sub_ind,:] = MDstd
#    sub_ind = sub_ind + 1

# np.savetxt('seg_fa_mean.txt',sublist_fa)
# np.savetxt('seg_fa_std.txt',sublist_fa_std)
# np.savetxt('seg_md_mean.txt',sublist_md)
# np.savetxt('seg_md_std.txt',sublist_md_std)
