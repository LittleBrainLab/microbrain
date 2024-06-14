from asyncio import subprocess
import numpy as np
import nibabel as nib
import microbrain.utils.surf_util as sutil
from microbrain.subcort_segmentation.mbrain_segment import register_probatlas_to_native

from dipy.align.reslice import reslice
from scipy.ndimage.morphology import binary_fill_holes
from scipy.ndimage.morphology import binary_closing, binary_dilation
from scipy.ndimage.measurements import center_of_mass
from scipy import ndimage
from os import system, environ, path

import os
import inspect
import shutil
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

from os import system
import subprocess


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
    if os.environ['FSLOUTPUTTYPE'] == 'NIFTI':
        fsl_extension = '.nii'
    elif os.environ['FSLOUTPUTTYPE'] == 'NIFTI_GZ':
        fsl_extension = '.nii.gz'
    return fsl_extension


def get_tissue_dir():
    """
    Return tissue directory in microbrain repository

    Returns
    -------
    tissue_dir: string
        tissue path
    """

    import microbrain  # ToDo. Is this the only way?
    module_path = inspect.getfile(microbrain)

    tissue_dir = os.path.dirname(module_path) + "/data/tissuepriors/"

    return tissue_dir

def vtktogii(vtk_fname, giiT, giiT2, C):
    """
    Convert vtk file to gii file

    Parameters
    ----------
    vtk_fname: string
        filename of vtk file
    giiT: string
        surface type 1
    giiT2: string
        surface type 2
    C: string
        cortex type

    Returns
    -------
    gii_fname: string
        filename of output gii file
    """
    gii_fname = vtk_fname.replace('.vtk', '.surf.gii')
    tempgii_fname = gii_fname.replace('.surf.gii', '.temp.surf.gii')
    os.system('mirtk convert-pointset ' + vtk_fname +
              ' ' + tempgii_fname + ' -nopointdata')
    os.system('wb_command -set-structure ' + tempgii_fname + ' ' + C +
              ' -surface-type ' + ' ' + giiT + ' -surface-secondary-type ' + giiT2)
    os.system('mv ' + tempgii_fname + ' ' + gii_fname)

    return gii_fname


def giimap(vtk_fname, gii_fname, scalars, mapname, C, flip=True):
    """
    Map scalar data from vtk file to gii file

    Parameters
    ----------
    vtk_fname: string
        filename of vtk file
    gii_fname: string
        filename of gii file
    scalars: string
        name of scalar data in vtk
    mapname: string
        name of scalar data in output gii
    C: string
        cortex type
    flip: boolean
        flip the scalar data

    Returns
    -------
    none
    """

    tempvtk = vtk_fname.replace('.vtk', '_temp.vtk')
    filename, extension = os.path.splitext(gii_fname)
    giiType, filename = os.path.splitext(filename)
    tempgii = gii_fname.replace(
        giiType + '.' + extension, 'temp.' + giiType + '.' + extension)

    os.system('mirtk delete-pointset-attributes ' +
              vtk_fname + ' ' + tempvtk + ' -all')
    os.system('mirtk copy-pointset-attributes ' + vtk_fname + ' ' +
              tempvtk + ' ' + tempgii + ' -pointdata ' + scalars + ' curv')
    os.system('rm ' + tempvtk)
    os.system('wb_command -set-structure ' + tempgii + ' ' + C)
    if flip:
        os.system('wb_command -metric-math "var * -1" ' +
                  tempgii + ' -var var ' + tempgii)

    os.system('wb_command -set-map-name  ' +
              tempgii + ' 1 ' + C + '_' + mapname)
    os.system('wb_command -metric-palette ' + tempgii +
              ' MODE_AUTO_SCALE_PERCENTAGE -pos-percent 2 98 -palette-name Gray_Interp -disp-pos true -disp-neg true -disp-zero true')
    if mapname == "Thickness":
        os.system('wb_command -metric-math "abs(thickness)" ' +
                  tempgii + ' -var thickness ' + tempgii)
        os.system('wb_command -metric-palette ' + tempgii +
                  ' MODE_AUTO_SCALE_PERCENTAGE -pos-percent 4 96 -interpolate true -palette-name videen_style -disp-pos true -disp-neg false -disp-zero false')

    os.system('mv ' + tempgii + ' ' + gii_fname)

    return


def freesurf_fix_topology(wm_surf_fname, C, freesurf_subdir):
    """
    Use freesurfer tools to fix topology of white matter surface

    Parameters
    ----------
    wm_surf_fname: string
        filename of white matter surface
    C: string
        cortex hemisphere
    freesurf_subdir: string
        freesurfer subject directory

    Returns
    -------
    wm_orig_fix_fname: string
        filename of fixed white matter surface
    """

    if C == 'CORTEX_LEFT':
        hemi = 'lh'
    else:
        hemi = 'rh'

    wm_inflate_fname = wm_surf_fname.replace('.vtk', '_inflate.vtk')
    if not os.path.exists(wm_inflate_fname):
        os.system('mirtk deform-mesh ' + wm_surf_fname +
                  ' ' + wm_inflate_fname + ' -inflate-brain')

    wm_surf_gii = vtktogii(wm_surf_fname, 'ANATOMICAL', 'GRAY_WHITE', C)
    wm_inflate_gii = vtktogii(wm_inflate_fname, 'INFLATED', 'GRAY_WHITE', C)

    wm_sphere_gii = wm_inflate_gii.replace(
        '_inflate.surf.gii', '_sphere.surf.gii')
    if not os.path.exists(wm_sphere_gii):
        os.system('mris_sphere -q ' + wm_inflate_gii + ' ' + wm_sphere_gii)

    # Copy gii's to freesurfer directory so that I can run mris_fix_topology (Super Hacky way of doing this)
    os.system('mris_convert ' + wm_surf_gii + ' ' +
              freesurf_subdir + 'surf/' + hemi + '.smoothwm')
    os.system('mris_convert ' + wm_surf_gii + ' ' +
              freesurf_subdir + 'surf/' + hemi + '.orig')
    os.system('mris_convert ' + wm_inflate_gii + ' ' +
              freesurf_subdir + 'surf/' + hemi + '.inflated')
    os.system('mris_convert ' + wm_sphere_gii + ' ' +
              freesurf_subdir + 'surf/' + hemi + '.qsphere')

    # Run mris_fix_topology in the temp freesurf folder
    freesurf_sub = os.path.basename(os.path.normpath(freesurf_subdir))
    print("Freesurfer Sub:" + freesurf_sub)
    wm_orig_fix_fname = wm_surf_fname.replace('.vtk', '_fix.vtk')
    if not os.path.exists(wm_orig_fix_fname):
        os.system('mris_fix_topology ' + freesurf_sub + ' ' + hemi)
        os.system('mris_convert ' + freesurf_subdir + 'surf/' +
                  hemi + '.orig ' + hemi + '.orig.vtk')
        os.system('mv ' + freesurf_subdir + 'surf/' +
                  hemi + '.orig.vtk ' + wm_orig_fix_fname)
    return wm_orig_fix_fname


def freesurfer_mris_sphere(wm_gii, wm_inflate2_gii, wm_sphere_gii, hemi, C, freesurf_subdir):
    """
    Use freesurfer tools to map white matter surface to sphere

    Parameters
    ----------
    wm_gii: string
        filename of white matter surface
    wm_inflate2_gii: string
        filename of inflated white matter surface
    wm_sphere_gii: string
        filename of sphere white matter surface
    hemi: string
        hemisphere
    C: string
        cortex hemisphere
    freesurf_subdir: string
        freesurfer subject directory

    Returns
    -------
    none
    """

    fs_wm = freesurf_subdir + 'surf/' + hemi + '.white'
    fs_smoothwm = freesurf_subdir + 'surf/' + hemi + '.smoothwm'
    fs_inflate = freesurf_subdir + 'surf/' + hemi + '.inflated'

    os.system('mris_convert ' + wm_gii + ' ' + fs_wm)
    os.system('mris_convert ' + wm_gii + ' ' + fs_smoothwm)
    os.system('mris_convert ' + wm_inflate2_gii + ' ' + fs_inflate)

    # os.system('mris_inflate ' + fs_wm + ' ' + fs_inflate)

    fs_sphere = freesurf_subdir + 'surf/' + hemi + '.sphere'
    os.system('mris_sphere ' + fs_inflate + ' ' + fs_sphere)

    # fs_sphere_reg = freesurf_subdir + 'surf/' + hemi + '.sphere.reg'
    # os.system('mris_register -nocurv -inflated ' + fs_sphere + ' /usr/local/freesurfer/average/' + hemi + '.average.curvature.filled.buckner40.tif ' + fs_sphere_reg)

    os.system('mris_convert ' + fs_sphere + ' ' + hemi + '.sphere.gii')
    os.system('mv ' + freesurf_subdir + 'surf/' +
              hemi + '.sphere.gii' + ' ' + wm_sphere_gii)
    os.system('wb_command -set-structure ' + wm_sphere_gii + ' ' + C +
              ' -surface-type SPHERICAL -surface-secondary-type GRAY_WHITE')

    return


def freesurf_mris_inflate(wm_gii, wm_inflate_gii, wm_sulc_gii, C):
    """
    Use freesurfer tools to inflate white matter surface

    Parameters
    ----------
    wm_gii: string
        filename of white matter surface
    wm_inflate_gii: string
        filename of inflated white matter surface
    wm_sulc_gii: string
        filename of sulc white matter surface
    C: string
        cortex hemisphere

    Returns
    -------
    none
    """

    os.system('mris_inflate -n 20 -sulc tmpsulc ' +
              wm_gii + ' ' + wm_inflate_gii)
    os.system('wb_command -set-structure ' + wm_inflate_gii + ' ' +
              C + ' -surface-type INFLATED -surface-secondary-type GRAY_WHITE')
    os.system('mris_convert -c rh.tmpsulc ' +
              wm_inflate_gii + ' ' + wm_sulc_gii)
    os.system('wb_command -set-structure ' + wm_sulc_gii + ' ' + C)

    return


def msm_reg(wm_sphere_gii, ref_sphere_gii, wm_sulc_gii, ref_sulc_gii, hemi, conf_file, trans_reg=None):
    """
    Register white matter surface to reference sphere

    Parameters
    ----------
    wm_sphere_gii: string
        filename of white matter sphere surface
    ref_sphere_gii: string
        filename of reference sphere surface
    wm_sulc_gii: string
        filename of white matter sulc surface
    ref_sulc_gii: string
        filename of reference sulc surface
    hemi: string
        hemisphere
    conf_file: string
        MSM configuration file
    trans_reg: string
        filename of initial linear transformation to be applied prior to nonlinear registration

    Returns
    -------
    none
    """
    print("Registering Sphere")
    # HCP configuration files
    msmConfDir = '/home/graham/Software/NeuroSoftware/HCPpipelines-master/MSMConfig/'

    if not trans_reg:
        os.system('msm --conf=' + msmConfDir + conf_file + ' --inmesh=' + wm_sphere_gii + ' --refmesh=' +
                  ref_sphere_gii + ' --indata=' + wm_sulc_gii + ' --refdata=' + ref_sulc_gii + ' --out=' + hemi + '. --verbose')
    else:
        print("Using trans reg")
        os.system('msm --conf=' + msmConfDir + conf_file + ' --inmesh=' + wm_sphere_gii + ' --refmesh=' + ref_sphere_gii +
                  ' --indata=' + wm_sulc_gii + ' --refdata=' + ref_sulc_gii + ' --trans=' + trans_reg + ' --out=' + hemi + '. --verbose')

    return


def register_probmap_to_native(fsource, ftemplate, fatlas, regDir, cpu_num=0):
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
        4D nifti file in MNI space to be registered
    regDir: string
        directory to save registration files

    Optional Parameters
    -------------------
    cpu_num: int
        integer defining the number of cpu threads to use for registration

    Returns
    -------
    fatlas_out: string
        filename for atlas registered to native space
    """
    os.makedirs(regDir, exist_ok=True)

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
               ' -i ' + fatlas +
               ' -o ' + fatlas_out)
    else:
        print("ANTs nonlinear registeration of atlases already performed")

    return fatlas_out


def multi_channel_tissue_classifier(ffa, fdwi, fmask, fgm_native, fwm_native, fcsf_native, segDir, tissue_prob_suffix, PriorWeight=0.1):
    """
    Run Atropos for multi-channel tissue classification

    Parameters
    ----------
    ffa: string
        filename of FA map
    fdwi: string
        filename of mean DWI
    fmask: string
        filename of brain mask
    fgm_native: string
        filename of GM probability map in native space
    fwm_native: string
        filename of WM probability map in native space
    fcsf_native: string
        filename of CSF probability map in native space
    segDir: string
        directory to save segmentation files
    tissue_prob_suffix: string
        suffix for tissue probability maps
    PriorWeight: float
        weight for prior probability

    Returns
    -------
    fseg_out: string
        filename of tissue segmentation
    fgm_prob_out: string
        filename of GM probability map
    fwm_prob_out: string
        filename of WM probability map
    fcsf_prob_out: string
        filename of CSF probability map
    """

    tissue_nclasses = 3

    print("Starting Tissue Segmentation")
    os.makedirs(segDir, exist_ok=True)

    tissue_base_name = segDir + 'tissue_prob_' + tissue_prob_suffix + '_'
    ftissuelabels = tissue_base_name + 'out_seg' + fsl_ext()
    ftissueprobs_base = tissue_base_name + 'out_prob_'
    if not path.exists(ftissuelabels):
        # Copy tissue probability maps to base tissue fname
        fprob_list = [fgm_native, fwm_native, fcsf_native]
        for i in range(0, tissue_nclasses):
            shutil.copyfile(fprob_list[i], tissue_base_name + str(i+1).zfill(2) + fsl_ext())
            process = subprocess.run(['fslcpgeom', fdwi, tissue_base_name + str(i+1).zfill(
                2) + fsl_ext()], stdout=subprocess.PIPE, universal_newlines=True)

        process = subprocess.run(
            ['fslcpgeom', fdwi, ffa], stdout=subprocess.PIPE, universal_newlines=True)
        process = subprocess.run(
            ['fslcpgeom', fdwi, fmask], stdout=subprocess.PIPE, universal_newlines=True)
        cmd = 'Atropos' + \
              ' -a [' + ffa + ']' + \
               ' -a [' + fdwi + ']' + \
               ' -x ' + fmask + \
               ' -i PriorProbabilityImages[' + str(tissue_nclasses) + ', ' + tissue_base_name + '%02d' + fsl_ext() + ',' + str(PriorWeight) + ',0.0001]' + \
               ' -m [0.3, 2x2x2] ' + \
               ' -s 1x2 -s 1x3 -s 2x3 ' + \
               ' --use-partial-volume-likelihoods false ' + \
               ' -o [' + ftissuelabels + ',' + ftissueprobs_base + '%02d' + fsl_ext() + ']' + \
               ' -k HistogramParzenWindows[1.0,32]'
        system(cmd)

        # Remove files used for input to Atropos
        for i in range(0, tissue_nclasses):
            system('rm ' + tissue_base_name + str(i+1).zfill(2) + fsl_ext())
    else:
        print("Tissue segmentation already completed")

    fseg_out = ftissuelabels
    fgm_prob_out = ftissueprobs_base + '01' + fsl_ext()
    fwm_prob_out = ftissueprobs_base + '02' + fsl_ext()
    fcsf_prob_out = ftissueprobs_base + '03' + fsl_ext()

    return fseg_out, fgm_prob_out, fwm_prob_out, fcsf_prob_out


def generate_initial_lr_wm(fwm_lh, fwm_rh, finter, finter_hippo, fwm_dist, fcortex_dist, fdwi_resamp, fdwi_neg, ftb_force, fmask, voxelDir, tissueDir, subDir, thisSub, suffix, preproc_suffix, surfDir, cpu_num=0):
    """
    Generate initial white matter surface

    Parameters
    ----------
    fwm_lh: string
        filename of left hemisphere white matter
    fwm_rh: string
        filename of right hemisphere white matter
    finter: string
        filename of interface mask
    finter_hippo: string
        filename of interface mask with hippocampus
    fwm_dist: string
        filename of white matter distance map
    fcortex_dist: string
        filename of cortex distance map
    fdwi_resamp: string
        filename of resampled mean DWI
    fdwi_neg: string
        filename of mean DWI with negative values
    ftb_force: string
        filename of tensor-based force map
    fmask: string
        filename of brain mask
    voxelDir: string
        directory to save voxel files
    tissueDir: string
        directory to save tissue files
    subDir: string
        directory to save subject files
    thisSub: string
        microbrain subject ID
    suffix: string
        suffix for bvalue shells used
    preproc_suffix: string
        suffix for preprocessing performed
    surfDir: string
        directory to save surface files

    Optional Parameters
    -------------------
    cpu_num: int
        integer defining the number of cpu threads to use for steps where applicable

    Returns
    -------
    none
    """

    # Register GM, WM, CSF probability map to native space
    regDir = subDir + 'registration/'

    # Get root project directory (FIX THIS LATER)
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    ftemplate = os.environ['FSLDIR'] + \
        '/data/standard/FSL_HCP1065_FA_1mm.nii.gz'
    fgm = get_tissue_dir() + 'avg152T1_gm_resampled.nii.gz'
    fwm = get_tissue_dir() + 'avg152T1_wm_resampled.nii.gz'
    fcsf = get_tissue_dir() + 'avg152T1_csf_resampled.nii.gz'

    ffa = subDir + 'DTI_maps/' + thisSub + suffix + '_FA' + fsl_ext()

    if preproc_suffix != '':
        fdwi = subDir + 'meanDWI/' + thisSub + '_' + \
            preproc_suffix + '_mean_b1000_n4' + fsl_ext()
    else:
        fdwi = subDir + 'meanDWI/' + thisSub + '_mean_b1000_n4' + fsl_ext()

    fgm_native = register_probmap_to_native(
        ffa, ftemplate, fgm, regDir, cpu_num=cpu_num)
    fwm_native = register_probmap_to_native(
        ffa, ftemplate, fwm, regDir, cpu_num=cpu_num)
    fcsf_native = register_probmap_to_native(
        ffa, ftemplate, fcsf, regDir, cpu_num=cpu_num)

    # Run multichannel tissue classificaiton using FA map and mean DWI
    fseg, fgm_prob, fwm_prob, fcsf_prob = multi_channel_tissue_classifier(
        ffa, fdwi, fmask, fgm_native, fwm_native, fcsf_native, tissueDir, thisSub)
    seg_img = nib.load(fseg)
    seg_data = seg_img.get_fdata()

    internal_struct_label = ['LEFT_THALAMUS',
                             'LEFT_GLOBUS',
                             'LEFT_STRIATUM',
                             'RIGHT_THALAMUS',
                             'RIGHT_GLOBUS',
                             'RIGHT_STRIATUM']

    external_struct_label = ['LEFT_HIPPO',
                             'LEFT_AMYGDALA',
                             'RIGHT_HIPPO',
                             'RIGHT_AMYGDALA']

    thalamus_struct_label = ['LEFT_THALAMUS',
                             'RIGHT_THALAMUS']

    # Make mask of internal structures
    pseudo_white = np.zeros(nib.load(
        voxelDir + thisSub + '_refined_LEFT_THALAMUS' + fsl_ext()).get_fdata().shape)
    for thisLabel in internal_struct_label:
        thisStructFile = voxelDir + thisSub + '_refined_' + thisLabel + fsl_ext()
        pseudo_white[nib.load(thisStructFile).get_fdata() > 0] = 1

    internal_mask = np.copy(pseudo_white)

    external_mask = np.zeros(nib.load(
        voxelDir + thisSub + '_refined_LEFT_THALAMUS' + fsl_ext()).get_fdata().shape)
    for thisLabel in external_struct_label:
        thisStructFile = voxelDir + thisSub + '_refined_' + thisLabel + fsl_ext()
        external_mask[nib.load(thisStructFile).get_fdata() > 0] = 1

    thalamus_label = np.zeros(nib.load(
        voxelDir + thisSub + '_refined_LEFT_THALAMUS' + fsl_ext()).get_fdata().shape)
    for thisLabel in thalamus_struct_label:
        thisStructFile = voxelDir + thisSub + '_refined_' + thisLabel + fsl_ext()
        thalamus_label[nib.load(thisStructFile).get_fdata() > 0] = 1

    # Read in harvard atlas to separate sub structures
    fharvard = environ['FSLDIR'] + \
        '/data/atlases/HarvardOxford/HarvardOxford-sub-prob-1mm.nii.gz'
    fharvard_native = register_probatlas_to_native(
        ffa, ftemplate, fharvard, regDir, cpu_num=cpu_num)
    harvard_img = nib.load(fharvard_native)
    harvard_data = harvard_img.get_fdata()

    fmni = environ['FSLDIR'] + \
        '/data/atlases/MNI/MNI-prob-1mm.nii.gz'
    fmni_native = register_probatlas_to_native(
        ffa, ftemplate, fmni, regDir, cpu_num=cpu_num)
    mni_img = nib.load(fmni_native)
    mni_data = mni_img.get_fdata()

    # Generate Cerrebellum, brainstem and ventricle segmentations
    # Output tissue masks
    dwi_img = nib.load(fdwi)

    fgm_mask = surfDir + thisSub + suffix + '_gm_mask' + fsl_ext()
    gm_mask = np.zeros(seg_data.shape)
    gm_mask[seg_data == 1] = 1
    nib.save(nib.Nifti1Image(gm_mask, dwi_img.affine), fgm_mask)

    fwm_mask = surfDir + thisSub + suffix + '_wm_mask' + fsl_ext()
    wm_mask = np.zeros(seg_data.shape)
    wm_mask[seg_data == 2] = 1
    nib.save(nib.Nifti1Image(wm_mask, dwi_img.affine), fwm_mask)

    fcsf_mask = surfDir + thisSub + suffix + '_csf_mask' + fsl_ext()
    csf_mask = np.zeros(seg_data.shape)
    csf_mask[seg_data == 3] = 1
    nib.save(nib.Nifti1Image(csf_mask, dwi_img.affine), fcsf_mask)

    # Output cerebral GM and cerebellar probs
    gm_prob = nib.load(fgm_prob).get_fdata()
    wm_prob = nib.load(fwm_prob).get_fdata()
    csf_prob = nib.load(fcsf_prob).get_fdata()
    vent_prob = harvard_data[:, :, :, 2] + harvard_data[:, :, :, 13]
    brain_stem_prob = harvard_data[:, :, :, 7]
    cerebellum_prob = mni_data[:, :, :, 1]/100

    gm_prob[cerebellum_prob > 0] = 0
    wm_prob[brain_stem_prob > 0] = 0
    wm_prob[cerebellum_prob > 0] = 0
    csf_prob[vent_prob > 0] = 0

    # Segment GM into Cerebellum GM / cerebrum GM
    atropos_gm_base = surfDir + thisSub + suffix + 'GM_'
    atropos_gm_label = atropos_gm_base + 'label' + fsl_ext()

    atropos_gm_prob = atropos_gm_base + '01' + fsl_ext()
    nib.save(nib.Nifti1Image(gm_prob, dwi_img.affine), atropos_gm_prob)

    atropos_cerebellum_prob = atropos_gm_base + '02' + fsl_ext()
    nib.save(nib.Nifti1Image(cerebellum_prob, dwi_img.affine),
             atropos_cerebellum_prob)

    if not os.path.exists(atropos_gm_label):
        system('Atropos -d 3' +
               ' -a [' + fdwi + ',0.1]' +
               ' -a [' + ffa + ',0.1]' +
               ' -x ' + fgm_mask +
               ' -i PriorProbabilityImages[2, ' + atropos_gm_base + '%02d' + fsl_ext() + ',0.1,0.0001]' +
               ' -m [0.3, 2x2x2] ' +
               ' -o ' + atropos_gm_label +
               ' -k HistogramParzenWindows[1.0,32]' +
               ' -v')
    else:
        print("GM segmentation already completed")

    gm_label = nib.load(atropos_gm_label).get_fdata()
    cerebellar_gm = gm_label == 2

    # Segment CSF into vewm_lh_fname, wm_rh_fname,ntricles / exterior csf
    atropos_csf_base = surfDir + thisSub + suffix + 'CSF_'
    atropos_csf_label = atropos_csf_base + 'label' + fsl_ext()

    atropos_csf_prob = atropos_csf_base + '01' + fsl_ext()
    nib.save(nib.Nifti1Image(csf_prob, dwi_img.affine), atropos_csf_prob)

    atropos_vent_prob = atropos_csf_base + '02' + fsl_ext()
    nib.save(nib.Nifti1Image(vent_prob, dwi_img.affine), atropos_vent_prob)

    if not os.path.exists(atropos_csf_label):
        system('Atropos -d 3' +
               ' -a [' + fdwi + ',0.1]' +
               ' -a [' + ffa + ',0.1]' +
               ' -x ' + fcsf_mask +
               ' -i PriorProbabilityImages[2, ' + atropos_csf_base + '%02d' + fsl_ext() + ',0.1,0.0001]' +
               ' -m [0.3, 2x2x2] ' +
               ' -o ' + atropos_csf_label +
               ' -k HistogramParzenWindows[1.0,32]' +
               ' -v ')
    else:
        print("CSF segmentation already completed")
    csf_label = nib.load(atropos_csf_label).get_fdata()
    ventricles = csf_label == 2

    # Segment WM into Cerebellum WM / Brain Stem WM / Cerebrum WM
    atropos_wm_base = surfDir + thisSub + suffix + 'WM_'
    atropos_wm_label = atropos_wm_base + 'label' + fsl_ext()

    atropos_wm_prob = atropos_wm_base + '01' + fsl_ext()
    nib.save(nib.Nifti1Image(wm_prob, dwi_img.affine), atropos_wm_prob)

    atropos_brain_stem_prob = atropos_wm_base + '02' + fsl_ext()
    nib.save(nib.Nifti1Image(brain_stem_prob, dwi_img.affine),
             atropos_brain_stem_prob)

    atropos_cerebellum_prob = atropos_wm_base + '03' + fsl_ext()
    nib.save(nib.Nifti1Image(cerebellum_prob, dwi_img.affine),
             atropos_cerebellum_prob)

    if not os.path.exists(atropos_wm_label):
        system('Atropos -d 3' +
               ' -a [' + fdwi + ',0.1]' +
               ' -a [' + ffa + ',0.1]' +
               ' -x ' + fwm_mask +
               ' -i PriorProbabilityImages[3, ' + atropos_wm_base + '%02d' + fsl_ext() + ',0.1,0.0001]' +
               ' -m [0.3, 2x2x2] ' +
               ' -o ' + atropos_wm_label +
               ' -k HistogramParzenWindows[1.0,32]' +
               ' -v')
    else:
        print("WM segmentation already completed")

    wm_label = nib.load(atropos_wm_label).get_fdata()
    brain_stem = wm_label == 2
    cerebellar_wm = wm_label == 3

    # Initial WM segmentation in native image voxel space
    pseudo_white[seg_data == 2] = 1
    pseudo_white[external_mask == 1] = 0
    pseudo_white[cerebellar_gm] = 0
    pseudo_white[cerebellar_wm] = 0
    pseudo_white[ventricles] = 1
    pseudo_white[brain_stem] = 0

    # Resample segmentations to 0.75 mm isotropic
    old_vsize = seg_img.header.get_zooms()[:3]
    new_vsize = (0.75, 0.75, 0.75)
    seg_data, new_affine = reslice(
        seg_data, seg_img.affine, old_vsize, new_vsize, order=0)
    pseudo_white, new_affine = reslice(
        pseudo_white, seg_img.affine, old_vsize, new_vsize, order=0)
    internal_mask, new_affine = reslice(
        internal_mask, seg_img.affine, old_vsize, new_vsize, order=0)
    external_mask, new_affine = reslice(
        external_mask, seg_img.affine, old_vsize, new_vsize, order=0)
    thalamus_label, new_affine = reslice(
        thalamus_label, seg_img.affine, old_vsize, new_vsize, order=0)
    brain_stem, new_affine = reslice(
        brain_stem, seg_img.affine, old_vsize, new_vsize, order=0)
    ventricles, new_affine = reslice(
        ventricles, seg_img.affine, old_vsize, new_vsize, order=0)
    cerebellar_gm, new_affine = reslice(
        cerebellar_gm, seg_img.affine, old_vsize, new_vsize, order=0)
    cerebellar_wm, new_affine = reslice(
        cerebellar_wm, seg_img.affine, old_vsize, new_vsize, order=0)
    dwi_data_reslice, new_affine = reslice(
        dwi_img.get_fdata(), dwi_img.affine, old_vsize, new_vsize, order=3)

    nib.save(nib.Nifti1Image(dwi_data_reslice, new_affine), fdwi_resamp)
    nib.save(nib.Nifti1Image(-1 * dwi_data_reslice, new_affine), fdwi_neg)

    # Segmentation is not perfect at the moment, remove any pieces of the pseudo wm mask on islands
    label_im, nb_labels = ndimage.label(pseudo_white == 1)
    sizes = ndimage.sum(pseudo_white == 1, label_im, range(nb_labels + 1))
    size_img = sizes[label_im]
    pseudo_white[size_img == max(sizes)] = 1
    pseudo_white[size_img != max(sizes)] = 0

    # Fill in holes from crossing fibre regions
    pseudo_white[binary_fill_holes(pseudo_white)] = 1
    pseudo_white[binary_closing(pseudo_white, iterations=3)] = 1
    pseudo_white[binary_fill_holes(pseudo_white)] = 1

    fpseudo_white = surfDir + thisSub + suffix + '_initwm' + fsl_ext()
    nib.save(nib.Nifti1Image(np.short(pseudo_white), new_affine), fpseudo_white)

    # Separate into left and right hemispheres based on the center of mass of the thalamus
    thalamus_center = np.uint8(np.round(center_of_mass(thalamus_label)))

    lh_pseudo_white = np.zeros(pseudo_white.shape)
    lh_pseudo_white[thalamus_center[0]+1:, :,
                    :] = pseudo_white[thalamus_center[0]+1:, :, :]

    rh_pseudo_white = np.zeros(pseudo_white.shape)
    rh_pseudo_white[:thalamus_center[0]+1, :,
                    :] = pseudo_white[:thalamus_center[0]+1, :, :]

    # Make interhemispheric mask
    inter_mask = np.ones(seg_data.shape)

    # Shift left hemisphere right/ right hemisphere left and fill in intersection
    lh_shifted = np.roll(lh_pseudo_white, -1, axis=0)
    rh_shifted = np.roll(rh_pseudo_white, 1, axis=0)
    wm_intersect = np.zeros(lh_shifted.shape)
    wm_intersect[np.logical_and(lh_shifted, rh_shifted)] = 1
    inter_mask[wm_intersect == 1] = 0
    lh_pseudo_white[wm_intersect == 1] = 0
    rh_pseudo_white[wm_intersect == 1] = 0

    # fill 2 voxels left/right of intersect for inter hemispheric mask
    inter_mask[np.logical_or(np.roll(
        wm_intersect, -1, axis=0) == 1, np.roll(wm_intersect, 1, axis=0) == 1)] = 0
    inter_mask[np.logical_or(np.roll(
        wm_intersect, -2, axis=0) == 1, np.roll(wm_intersect, 2, axis=0) == 1)] = 0
    inter_mask[brain_stem] = 0
    inter_mask[thalamus_label == 1] = 0
    inter_mask[ventricles] = 0

    # Segmentation is not perfect at the moment, remove any pieces of the interhempispheric mask on islands
    label_im, nb_labels = ndimage.label(inter_mask == 0)
    sizes = ndimage.sum(inter_mask == 0, label_im, range(nb_labels + 1))
    size_img = sizes[label_im]
    inter_mask[size_img == max(sizes)] = 0
    inter_mask[size_img != max(sizes)] = 1

    nib.save(nib.Nifti1Image(np.byte(lh_pseudo_white), new_affine), fwm_lh)

    nib.save(nib.Nifti1Image(np.byte(rh_pseudo_white), new_affine), fwm_rh)

    nib.save(nib.Nifti1Image(np.byte(inter_mask), new_affine), finter)

    inter_mask[binary_dilation(binary_dilation(external_mask == 1))] = 0
    nib.save(nib.Nifti1Image(np.byte(inter_mask), new_affine), finter_hippo)

    # Output distance maps for surface based cortical segmentation
    wm_prob_img = nib.load(fwm_prob)
    wm_prob = wm_prob_img.get_fdata()
    gm_prob = nib.load(fgm_prob).get_fdata()
    csf_prob = nib.load(fcsf_prob).get_fdata()

    # Resample tissue probability maps
    wm_prob, new_affine = reslice(
        wm_prob, seg_img.affine, old_vsize, new_vsize, order=3)
    gm_prob, new_affine = reslice(
        gm_prob, seg_img.affine, old_vsize, new_vsize, order=3)
    csf_prob, new_affine = reslice(
        csf_prob, seg_img.affine, old_vsize, new_vsize, order=3)

    # WM/cortex distance map (from tissue probs)
    wm_dist = -1 * wm_prob
    wm_dist = wm_dist + gm_prob
    wm_dist = wm_dist + csf_prob
    wm_dist[internal_mask == 1] = -1
    wm_dist[ventricles] = -1
    wm_dist[binary_dilation(binary_dilation(brain_stem))] = 1
    wm_dist[external_mask == 1] = 1

    nib.save(nib.Nifti1Image(wm_dist, new_affine), fwm_dist)

    # cortex/CSF distance map (from tissue probs)
    cortex_dist = -1 * wm_prob
    cortex_dist = cortex_dist - gm_prob
    cortex_dist = cortex_dist + csf_prob
    cortex_dist[internal_mask == 1] = -1
    cortex_dist[ventricles] = -1
    cortex_dist[binary_dilation(binary_dilation(brain_stem))] = 2
    cortex_dist[cerebellar_gm] = 2
    cortex_dist[cerebellar_wm] = 2
    cortex_dist[external_mask == 1] = 2

    nib.save(nib.Nifti1Image(cortex_dist, new_affine), fcortex_dist)

    # Estimate boundary FA
    fwm_edge = fpseudo_white.replace(fsl_ext(), '_edge' + fsl_ext())
    os.system('fslmaths ' + fpseudo_white + ' -edge -bin -mas ' +
              fpseudo_white + ' ' + fwm_edge)

    edge_img = nib.load(fwm_edge)
    edge_data = edge_img.get_fdata()

    ffa = subDir + 'DTI_maps/' + thisSub + suffix + '_FA' + fsl_ext()
    fa_img = nib.load(ffa)
    fa_data = fa_img.get_fdata()
    new_vsize = (0.75, 0.75, 0.75)
    fa_data, new_affine = reslice(
        fa_data, fa_img.affine, fa_img.header.get_zooms()[:3], new_vsize, order=1)
    fa_target = np.percentile(fa_data[edge_data == 1], 20)

    # Output tensor based force map
    force_data = -fa_data + fa_target
    force_data[csf_prob > 0.5] = csf_prob[csf_prob > 0.5]
    force_data[wm_dist == -1] = -1.0
    force_data[external_mask == 1] == 1.0
    force_data[inter_mask == 0] = -1.0
    nib.save(nib.Nifti1Image(force_data, new_affine), ftb_force)

    return finter, finter_hippo, fwm_dist, fcortex_dist, fdwi_resamp, fdwi_neg, ftb_force


def generate_initial_wm_surface(surfDir, freesurf_subdir, thisSub, suffix, wm_lh_fname, wm_rh_fname, wm_surf_fname, finter):
    """
    Generate initial white matter surface

    Parameters
    ----------
    surfDir: string
        directory to save surface files
    freesurf_subdir: string
        directory to save freesurfer files
    thisSub: string
        microbrain subject ID
    suffix: string
        suffix for bvalue shells used
    wm_lh_fname: string
        filename of left hemisphere white matter mask
    wm_rh_fname: string
        filename of right hemisphere white matter mask
    wm_surf_fname: string
        filename of initial white matter surface
    finter: string
        filename of output interface mask

    Returns
    -------
    none
    """

    wm_lh_surf_fname = surfDir + thisSub + suffix + '_wm_lh.vtk'
    os.system('mirtk extract-surface ' + wm_lh_fname +
              ' ' + wm_lh_surf_fname + ' -isovalue 0.5')
    os.system('mirtk extract-connected-points ' +
              wm_lh_surf_fname + ' ' + wm_lh_surf_fname)
    os.system('mirtk remesh-surface ' + wm_lh_surf_fname + ' ' +
              wm_lh_surf_fname + ' -min-edgelength 0.5 -max-edgelength 1.0')

    wm_lh_surf = sutil.read_surf_vtk(wm_lh_surf_fname)

    holeFiller = vtk.vtkFillHolesFilter()
    holeFiller.SetInputData(wm_lh_surf)
    holeFiller.SetHoleSize(0)
    holeFiller.Update()
    wm_lh_surf = holeFiller.GetOutput()

    connFilter = vtk.vtkConnectivityFilter()
    connFilter.SetInputData(wm_lh_surf)
    connFilter.SetExtractionModeToLargestRegion()
    connFilter.Update()
    wm_lh_surf = connFilter.GetOutput()

    smoother = vtk.vtkSmoothPolyDataFilter()
    smoother.SetInputData(wm_lh_surf)
    smoother.SetNumberOfIterations(300)
    smoother.SetRelaxationFactor(0.1)
    smoother.Update()
    wm_lh_surf = smoother.GetOutput()

    sutil.write_surf_vtk(wm_lh_surf, wm_lh_surf_fname)

    wm_lh_fix_fname = freesurf_fix_topology(
        wm_lh_surf_fname, 'CORTEX_LEFT', freesurf_subdir)  # Outputs wm_surf_fname_fix
    wm_lh_surf = sutil.read_surf_vtk(wm_lh_fix_fname)

    leftLabels = np.ones((wm_lh_surf.GetNumberOfPoints(),))
    vtk_pntList = numpy_to_vtk(leftLabels)
    vtk_pntList.SetName("HemiLabels")
    wm_lh_surf.GetPointData().AddArray(vtk_pntList)

    wm_rh_surf_fname = surfDir + thisSub + suffix + '_wm_rh.vtk'
    os.system('mirtk extract-surface ' + wm_rh_fname +
              ' ' + wm_rh_surf_fname + ' -isovalue 0.5')
    os.system('mirtk extract-connected-points ' +
              wm_rh_surf_fname + ' ' + wm_rh_surf_fname)
    os.system('mirtk remesh-surface ' + wm_rh_surf_fname + ' ' +
              wm_rh_surf_fname + ' -min-edgelength 0.5 -max-edgelength 1.0')

    wm_rh_surf = sutil.read_surf_vtk(wm_rh_surf_fname)

    holeFiller = vtk.vtkFillHolesFilter()
    holeFiller.SetInputData(wm_rh_surf)
    holeFiller.SetHoleSize(0)
    holeFiller.Update()
    wm_rh_surf = holeFiller.GetOutput()

    connFilter = vtk.vtkConnectivityFilter()
    connFilter.SetInputData(wm_rh_surf)
    connFilter.SetExtractionModeToLargestRegion()
    connFilter.Update()
    wm_rh_surf = connFilter.GetOutput()

    smoother = vtk.vtkSmoothPolyDataFilter()
    smoother.SetInputData(wm_rh_surf)
    smoother.SetNumberOfIterations(300)
    smoother.SetRelaxationFactor(0.1)
    smoother.Update()
    wm_rh_surf = smoother.GetOutput()

    sutil.write_surf_vtk(wm_rh_surf, wm_rh_surf_fname)

    wm_rh_fix_fname = freesurf_fix_topology(
        wm_rh_surf_fname, 'CORTEX_RIGHT', freesurf_subdir)
    wm_rh_surf = sutil.read_surf_vtk(wm_rh_fix_fname)

    rightLabels = np.ones((wm_rh_surf.GetNumberOfPoints(),))*2
    vtk_pntList = numpy_to_vtk(rightLabels)
    vtk_pntList.SetName("HemiLabels")
    wm_rh_surf.GetPointData().AddArray(vtk_pntList)

    # combine left and right surfaces
    appender = vtk.vtkAppendPolyData()
    appender.AddInputData(wm_lh_surf)
    appender.AddInputData(wm_rh_surf)
    appender.Update()
    wm_surf = appender.GetOutput()
    sutil.write_surf_vtk(wm_surf, wm_surf_fname)

    # Initalize white matter surface with mask
    wm_surf = sutil.read_surf_vtk(wm_surf_fname)
    imaskDil_ImgVTK, imaskDil_HdrVTK, sformMat = sutil.read_image(finter)
    wm_surf = sutil.interpolate_voldata_to_surface(
        wm_surf, imaskDil_ImgVTK, sformMat, pntDataName='Mask', categorical=True)
    wm_surf = sutil.interpolate_voldata_to_surface(
        wm_surf, imaskDil_ImgVTK, sformMat, pntDataName='InitialStatus', categorical=True)
    sutil.write_surf_vtk(wm_surf, wm_surf_fname)

    # Note that because the way 1's and 0's are used here, erode will make the mask bigger
    os.system('mirtk open-scalars ' + wm_surf_fname +
              ' ' + wm_surf_fname + ' -a Mask -n 5 ')
    os.system('mirtk dilate-scalars ' + wm_surf_fname +
              ' ' + wm_surf_fname + ' -a Mask -n 5 ')
    os.system('mirtk erode-scalars ' + wm_surf_fname +
              ' ' + wm_surf_fname + ' -a Mask -n 6 ')

    os.system('mirtk open-scalars ' + wm_surf_fname + ' ' +
              wm_surf_fname + ' -a InitialStatus -n 5 ')
    os.system('mirtk dilate-scalars ' + wm_surf_fname +
              ' ' + wm_surf_fname + ' -a InitialStatus -n 5 ')
    os.system('mirtk erode-scalars ' + wm_surf_fname + ' ' +
              wm_surf_fname + ' -a InitialStatus -n 6 ')

    return


def deform_initial_wm_surface_with_tissue_probabilities(wm_surf_fname, wm_surf_dist_fname, fwm_dist, finter_hippo, cpu_str):
    """
    Deform initial white matter surface with tissue probabilities

    Parameters
    ----------
    wm_surf_fname: string
        filename of initial white matter surface
    wm_surf_dist_fname: string
        filename of output white matter surface
    fwm_dist: string
        filename of white matter distance map
    finter_hippo: string
        filename of output hippocampus/amygdala mask
    cpu_str: string
        string defining the number of cpu threads to use for steps where applicable

    Returns
    -------
    none
    """

    os.system('mirtk deform-mesh ' + wm_surf_fname + ' ' + wm_surf_dist_fname + ' -distance-image ' + fwm_dist + ' -distance 1.0 -distance-smoothing 1 -distance-averaging 4 2 1 -distance-measure normal -optimizer EulerMethod -step 0.2 -steps 100 200 -epsilon 1e-6 -delta 0.001 -min-active 1% -reset-status -nointersection -fast-collision-test -min-width 0.1 -min-distance 0.1 -repulsion 4 -repulsion-distance 0.5 -repulsion-width 1.0 -curvature 4.0 -gauss-curvature 1.0 -gauss-curvature-minimum .1 -gauss-curvature-maximum .2 -gauss-curvature-outside 0.5 -edge-distance-type ClosestMaximum -remesh 1 -min-edge-length 0.5 -max-edge-length 1.0' + cpu_str)

    # Add hippocampus/amygdala region to mask (no deformation of hippo/amygdala region from here on)
    wm_surf = sutil.read_surf_vtk(wm_surf_dist_fname)
    imaskDil_ImgVTK, _, sformMat = sutil.read_image(finter_hippo)
    wm_surf = sutil.interpolate_voldata_to_surface(
        wm_surf, imaskDil_ImgVTK, sformMat, pntDataName='Mask', categorical=True)
    wm_surf = sutil.interpolate_voldata_to_surface(
        wm_surf, imaskDil_ImgVTK, sformMat, pntDataName='InitialStatus', categorical=True)
    sutil.write_surf_vtk(wm_surf, wm_surf_dist_fname)

    # Note that because the way 1's and 0's are used here, erode will make the mask bigger
    os.system('mirtk open-scalars ' + wm_surf_dist_fname +
              ' ' + wm_surf_dist_fname + ' -a Mask -n 5 ')
    os.system('mirtk dilate-scalars ' + wm_surf_dist_fname +
              ' ' + wm_surf_dist_fname + ' -a Mask -n 5 ')
    os.system('mirtk erode-scalars ' + wm_surf_dist_fname +
              ' ' + wm_surf_dist_fname + ' -a Mask -n 6 ')

    os.system('mirtk open-scalars ' + wm_surf_dist_fname + ' ' +
              wm_surf_dist_fname + ' -a InitialStatus -n 5 ')
    os.system('mirtk dilate-scalars ' + wm_surf_dist_fname +
              ' ' + wm_surf_dist_fname + ' -a InitialStatus -n 5 ')
    os.system('mirtk erode-scalars ' + wm_surf_dist_fname + ' ' +
              wm_surf_dist_fname + ' -a InitialStatus -n 6 ')

    return


def deform_wm_surface_with_tbforce(wm_surf_dist_fname, wm_tensor_fname, fdwi_resamp, ftb_force, cpu_str):
    """
    Deform white matter surface with tensor based force

    Parameters
    ----------
    wm_surf_dist_fname: string
        filename of input white matter surface
    wm_tensor_fname: string
        filename of output white matter surface
    fdwi_resamp: string
        filename of resampled mean DWI image
    ftb_force: string
        filename of tensor based force map
    cpu_str: string
        string defining the number of cpu threads to use for deformation

    Returns
    -------
    none
    """

    os.system('mirtk deform-mesh ' + wm_surf_dist_fname + ' ' + wm_tensor_fname + ' -image ' + fdwi_resamp + ' -edge-distance 1.0 -edge-distance-smoothing 1 -edge-distance-median 1 -edge-distance-averaging 4 2 1 -distance-image ' + ftb_force +
              ' -distance 1.0 -distance-smoothing 1 -distance-averaging 4 2 1 -distance-measure normal -optimizer EulerMethod -step 0.2 -steps 100 200 -epsilon 1e-6 -delta 0.001 -min-active 1% -reset-status -nointersection -fast-collision-test -min-width 0.1 -min-distance 0.1 -repulsion 4 -repulsion-distance 0.5 -repulsion-width 1.0 -curvature 4.0 -gauss-curvature 1.0 -gauss-curvature-minimum .1 -gauss-curvature-maximum .2 -gauss-curvature-outside 0.5 -edge-distance-type ClosestMaximum -remesh 1 -min-edge-length 0.5 -max-edge-length 1.0' + cpu_str)

    return


def deform_wm_surface_with_meanDWI(wm_tensor_fname, wm_final_fname, fdwi_resamp, cpu_str):
    """
    Deform white matter surface with mean DWI image

    Parameters
    ----------
    wm_tensor_fname: string
        filename of input white matter surface
    wm_final_fname: string
        filename of output white matter surface
    fdwi_resamp: string
        filename of resampled mean DWI image
    cpu_str: string
        string defining the number of cpu threads to use for deformation

    Returns
    -------
    none
    """

    os.system('mirtk deform-mesh ' + wm_tensor_fname + ' ' + wm_final_fname + ' -image ' + fdwi_resamp + ' -edge-distance 1.0 -edge-distance-smoothing 1 -edge-distance-median 1 -edge-distance-averaging 1 -optimizer EulerMethod -step 0.2 -steps 300 -epsilon 1e-6 -delta 0.001 -min-active 1% -reset-status -nointersection -fast-collision-test -min-width 0.1 -min-distance 0.1 -repulsion 4 -repulsion-distance 0.5 -repulsion-width 1.0 -curvature 4.0 -gauss-curvature 1.0 -gauss-curvature-minimum .1 -gauss-curvature-maximum .2 -gauss-curvature-outside 0.5 -edge-distance-type ClosestMaximum -remesh 1 -min-edge-length 0.5 -max-edge-length 1.0' + cpu_str)

    return


def deform_cortex_surface_with_tissue_probabilities(wm_final_fname, wm_expand_fname, fcortex_dist, cpu_str):
    """
    Deform cortex surface with tissue probabilities

    Parameters
    ----------
    wm_final_fname: string
        filename of input white matter surface
    wm_expand_fname: string
        filename of output cortex surface
    fcortex_dist: string
        filename of cortex distance map
    cpu_str: string
        string defining the number of cpu threads to use for deformation

    Returns
    -------
    none
    """

    os.system('mirtk deform-mesh ' + wm_final_fname + ' ' + wm_expand_fname + ' -distance-image ' + fcortex_dist + ' -distance 0.5 -distance-smoothing 1 -distance-averaging 4 2 1 -distance-measure normal -optimizer EulerMethod -step 0.2 -steps 100 200 -epsilon 1e-6 -delta 0.001 -min-active 5% -reset-status -nointersection -fast-collision-test -min-width 0.1 -min-distance 0.1 -repulsion 2.0 -repulsion-distance 0.5 -repulsion-width 1.0 -curvature 4.0 -edge-distance-type ClosestMaximum -edge-distance-max-intensity -1 -gauss-curvature 1.6 -gauss-curvature-minimum .1 -gauss-curvature-maximum .4 -gauss-curvature-inside 2 -negative-gauss-curvature-action inflate' + cpu_str)

    return


def deform_cortex_surface_with_meanDWI(wm_expand_fname, pial_fname, fcortex_dist, fdwi_neg, cpu_str):
    """
    Deform cortex surface with mean DWI image

    Parameters
    ----------
    wm_expand_fname: string
        filename of input cortex surface
    pial_fname: string
        filename of output cortex surface
    fcortex_dist: string
        filename of cortex distance map
    fdwi_neg: string
        filename of negative mean DWI image
    cpu_str: string
        string defining the number of cpu threads to use for deformation

    Returns
    -------
    none
    """

    os.system('mirtk deform-mesh ' + wm_expand_fname + ' ' + pial_fname + ' -image ' + fdwi_neg + ' -edge-distance 1.0 -edge-distance-smoothing 1 -edge-distance-median 1 -edge-distance-averaging 1 -distance-image ' + fcortex_dist +
              ' -distance 0.35 -distance-smoothing 1 -distance-averaging 1 -distance-measure normal -optimizer EulerMethod -step 0.2 -steps 300 -epsilon 1e-6 -delta 0.001 -min-active 5% -reset-status -nointersection -fast-collision-test -min-width 0.1 -min-distance 0.1 -repulsion 2.0 -repulsion-distance 0.5 -repulsion-width 1.0 -curvature 4.0 -edge-distance-type ClosestMaximum -edge-distance-max-intensity -1 -gauss-curvature 1.6 -gauss-curvature-minimum .1 -gauss-curvature-maximum .4 -gauss-curvature-inside 2 -negative-gauss-curvature-action inflate' + cpu_str)

    return


def split_surface(surf_fname, lh_surf_fname, rh_surf_fname):
    """
    Split surface into left and right hemispheres

    Parameters
    ----------
    surf_fname: string
        filename of input surface
    lh_surf_fname: string
        filename of output left hemisphere surface
    rh_surf_fname: string
        filename of output right hemisphere surface

    Returns
    -------
    none
    """
    surf = sutil.read_surf_vtk(surf_fname)
    print(surf_fname)
    [lh_surf, rh_surf] = sutil.split_surface_by_label(surf)
    sutil.write_surf_vtk(lh_surf, lh_surf_fname)
    sutil.write_surf_vtk(rh_surf, rh_surf_fname)

    return


def generate_midthickness(finner_surf, fouter_surf, fmedial):
    """
    Generate mid-surface

    Parameters
    ----------
    finner_surf: string
        filename of inner surface
    fouter_surf: string
        filename of outer surface
    fmedial: string
        filename of output mid-surface

    Returns
    -------
    none
    """

    inner_surf = sutil.read_surf_vtk(finner_surf)
    outer_surf = sutil.read_surf_vtk(fouter_surf)

    # compute mid-surface
    medial_surf = sutil.compute_mid_surface(inner_surf, outer_surf)
    sutil.write_surf_vtk(medial_surf, fmedial)

    return


def generate_surfaces_from_dwi(fmask, voxelDir, outDir, thisSub, preproc_suffix, shell_suffix, freesurf_subdir, cpu_num=0, use_tensor_wm=False):
    """
    Generate cortical surfaces from DWI data

    Parameters
    ----------
    fmask: string
        filename of brain mask
    voxelDir: string
        directory containing voxel-wise data
    outDir: string
        output directory
    thisSub: string
        microbrain subject ID
    preproc_suffix: string
        suffix for preprocessed data
    shell_suffix: string
        suffix for bvalue shells used
    freesurf_subdir: string
        directory to save freesurfer files
    cpu_num: int
        number of cpu threads to use for deformation
    use_tensor_wm: bool
        flag to use tensor based force for white matter surface deformation

    Returns
    -------
    none
    """

    print("Surfing: " + thisSub)
    subDir = outDir + '/' + thisSub + '/'
    surfDir = subDir + 'surf/'
    tissueDir = surfDir + 'tissue_classification/'
    initialMaskDir = surfDir + 'initial_masks/'
    initialSurfDir = surfDir + 'initial_surfaces/'
    surfSegDir = surfDir + 'mesh_segmentation/'

    for curr_dir in [surfDir, tissueDir, initialMaskDir, initialSurfDir, surfSegDir]:
        os.makedirs(curr_dir, exist_ok=True)

    if preproc_suffix != '':
        suffix = '_' + preproc_suffix + '_' + shell_suffix
    else:
        suffix = '_' + shell_suffix

    if cpu_num > 0:
        cpu_str = ' -threads ' + str(cpu_num) + ' '
    else:
        cpu_str = ' '

    wm_lh_fname = initialMaskDir + thisSub + suffix + '_wm_lh' + fsl_ext()
    wm_rh_fname = initialMaskDir + thisSub + suffix + '_wm_rh' + fsl_ext()
    finter = initialMaskDir + thisSub + suffix + '_inter_mask' + fsl_ext()
    finter_hippo = initialMaskDir + thisSub + \
        suffix + '_inter_mask_hippo' + fsl_ext()
    fwm_dist = initialMaskDir + thisSub + suffix + '_wm_cortex_dist' + fsl_ext()
    fcortex_dist = initialMaskDir + thisSub + \
        suffix + '_cortex_csf_dist' + fsl_ext()
    ftb_force = initialMaskDir + thisSub + \
        suffix + '_tensor-based-force' + fsl_ext()
    fdwi_resamp = initialMaskDir + thisSub + suffix + 'DWI_resamp' + fsl_ext()
    fdwi_neg = initialMaskDir + thisSub + suffix + 'DWI_neg' + fsl_ext()

    if not os.path.exists(wm_lh_fname) or not os.path.exists(wm_rh_fname):
        generate_initial_lr_wm(wm_lh_fname, wm_rh_fname, finter, finter_hippo, fwm_dist, fcortex_dist, fdwi_resamp, fdwi_neg,
                               ftb_force, fmask, voxelDir, tissueDir, subDir, thisSub, suffix, preproc_suffix, initialMaskDir, cpu_num=cpu_num)

    wm_surf_fname = surfSegDir + thisSub + suffix + '_wm.vtk'
    if not os.path.exists(wm_surf_fname):
        generate_initial_wm_surface(initialSurfDir, freesurf_subdir, thisSub,
                                    suffix, wm_lh_fname, wm_rh_fname, wm_surf_fname, finter)

    # Refine WM Surface
    print("Initial WM Surface Deformation with Tissue Probabilities")
    wm_surf_dist_fname = wm_surf_fname.replace('.vtk', '_TissueProbForce.vtk')
    if not os.path.exists(wm_surf_dist_fname):
        deform_initial_wm_surface_with_tissue_probabilities(
            wm_surf_fname, wm_surf_dist_fname, fwm_dist, finter_hippo, cpu_str)
    else:
        print("white surface already refined on tissue probability map")

    # Refine WM Surface with Tensor-Based Force
    print("WM Surface with mean DWI with tensor-based force")
    wm_tensor_fname = wm_surf_fname.replace('.vtk', '_tensorBasedForce.vtk')
    if not os.path.exists(wm_tensor_fname):
        deform_wm_surface_with_tbforce(
            wm_surf_dist_fname, wm_tensor_fname, fdwi_resamp, ftb_force, cpu_str)
    else:
        print("white surface already generated with Tensor-Based Force")

    # Refine WM Surface on mean DWI
    wm_final_fname = wm_surf_fname.replace('.vtk', '_final.vtk')
    if use_tensor_wm:
        shutil.copyfile(wm_tensor_fname, wm_final_fname)
    else:
        print("Refining WM Surface with mean DWI only")
        if not os.path.exists(wm_final_fname):
            deform_wm_surface_with_meanDWI(
                wm_tensor_fname, wm_final_fname, fdwi_resamp, cpu_str)
        else:
            print("white surface already refined on DWI")

    wm_final_mask_fname = wm_final_fname.replace('.vtk', '_mask' + fsl_ext())
    if not os.path.exists(wm_final_mask_fname):
        sutil.surf_to_volume_mask(
            finter, wm_final_fname, 1, wm_final_mask_fname)

    # read in mask
    mask = nib.load(wm_final_mask_fname).get_fdata()
    cortex_dist_img = nib.load(fcortex_dist)
    cortex_dist = cortex_dist_img.get_fdata()
    cortex_dist[mask == 1] = -1
    cortex_dist_img = nib.Nifti1Image(
        cortex_dist, cortex_dist_img.affine, cortex_dist_img.header)
    nib.save(cortex_dist_img, fcortex_dist)

    # Move surface outward edge based on DWI and Tissue Classification Probability maps
    print("Expanding white surface Using GM/CSF Boundary Distance Map")
    wm_expand_fname = surfSegDir + thisSub + suffix + '_pial_init.vtk'
    if not os.path.exists(wm_expand_fname):
        deform_cortex_surface_with_tissue_probabilities(
            wm_final_fname, wm_expand_fname, fcortex_dist, cpu_str)
    else:
        print("Expanded surface already generated")

    # Generate pial surface on mean DWI edges are more heavily weighted.
    print("Refining CSF/GREY border on mean DWI")
    pial_fname = surfSegDir + thisSub + suffix + '_pial.vtk'
    if not os.path.exists(pial_fname):
        deform_cortex_surface_with_meanDWI(
            wm_expand_fname, pial_fname, fcortex_dist, fdwi_neg, cpu_str)
    else:
        print("pial remesh already made")

    # Split WM surface
    lh_wm_fname = wm_final_fname.replace('.vtk', '_lh.vtk')
    rh_wm_fname = wm_final_fname.replace('.vtk', '_rh.vtk')
    if not os.path.exists(lh_wm_fname) or not os.path.exists(rh_wm_fname):
        split_surface(wm_final_fname, lh_wm_fname, rh_wm_fname)
    lh_wm_gii = vtktogii(lh_wm_fname, 'ANATOMICAL',
                         'GRAY_WHITE', 'CORTEX_LEFT')
    rh_wm_gii = vtktogii(rh_wm_fname, 'ANATOMICAL',
                         'GRAY_WHITE', 'CORTEX_RIGHT')

    # Split Pial surface
    lh_pial_fname = pial_fname.replace('.vtk', '_lh.vtk')
    rh_pial_fname = pial_fname.replace('.vtk', '_rh.vtk')
    if not os.path.exists(lh_pial_fname) or not os.path.exists(rh_pial_fname):
        split_surface(pial_fname, lh_pial_fname, rh_pial_fname)
    lh_pial_gii = vtktogii(lh_pial_fname, 'ANATOMICAL', 'PIAL', 'CORTEX_LEFT')
    rh_pial_gii = vtktogii(rh_pial_fname, 'ANATOMICAL', 'PIAL', 'CORTEX_RIGHT')

    # Generate mid-thickness surface
    print("Getting mid thickness surface")
    lh_mid_fname = lh_pial_fname.replace('pial', 'midthick')
    rh_mid_fname = rh_pial_fname.replace('pial', 'midthick')
    if not os.path.exists(lh_mid_fname) or not os.path.exists(rh_mid_fname):
        generate_midthickness(lh_wm_fname, lh_pial_fname, lh_mid_fname)
        generate_midthickness(rh_wm_fname, rh_pial_fname, rh_mid_fname)

    lh_mid_gii = vtktogii(lh_mid_fname, 'ANATOMICAL',
                          'MIDTHICKNESS', 'CORTEX_LEFT')
    rh_mid_gii = vtktogii(rh_mid_fname, 'ANATOMICAL',
                          'MIDTHICKNESS', 'CORTEX_RIGHT')

    return