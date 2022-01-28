import numpy as np
import nibabel as nib
import utils.surf_util as sutil
from time import time
from nibabel.processing import smooth_image
from dipy.align.reslice import reslice
from scipy.ndimage.morphology import binary_fill_holes
from scipy.ndimage.morphology import binary_closing, binary_dilation
from scipy.ndimage.measurements import center_of_mass
from scipy import ndimage

import os
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

import matplotlib.pyplot as plt

from os import system

def fsl_ext():
    fsl_extension = ''
    if os.environ['FSLOUTPUTTYPE'] == 'NIFTI':
        fsl_extension = '.nii'
    elif os.environ['FSLOUTPUTTYPE'] == 'NIFTI_GZ':
        fsl_extension = '.nii.gz'
    return fsl_extension

def vtktogii(vtk_fname, giiT, giiT2, C):
    gii_fname = vtk_fname.replace('.vtk', '.surf.gii')
    tempgii_fname = gii_fname.replace('.surf.gii','.temp.surf.gii')
    os.system('mirtk convert-pointset ' + vtk_fname + ' ' + tempgii_fname + ' -nopointdata')
    os.system('wb_command -set-structure ' + tempgii_fname + ' ' + C + ' -surface-type ' + ' ' + giiT + ' -surface-secondary-type ' + giiT2)
    os.system('mv ' + tempgii_fname + ' ' + gii_fname)
    
    return gii_fname

def giimap(vtk_fname, gii_fname, scalars, mapname, C, flip=True):
    tempvtk = vtk_fname.replace('.vtk','_temp.vtk')
    filename, extension = os.path.splitext(gii_fname)
    giiType, filename = os.path.splitext(filename)
    tempgii = gii_fname.replace(giiType +'.' + extension, 'temp.' + giiType +'.' + extension)
    
    os.system('mirtk delete-pointset-attributes ' + vtk_fname + ' ' + tempvtk + ' -all')
    os.system('mirtk copy-pointset-attributes ' + vtk_fname + ' ' + tempvtk + ' ' + tempgii + ' -pointdata ' + scalars + ' curv')
    os.system('rm ' + tempvtk)
    os.system('wb_command -set-structure ' + tempgii + ' ' + C)
    if flip:
        os.system('wb_command -metric-math "var * -1" ' + tempgii + ' -var var ' + tempgii)
    
    os.system('wb_command -set-map-name  ' + tempgii + ' 1 ' + C + '_' + mapname)
    os.system('wb_command -metric-palette ' + tempgii + ' MODE_AUTO_SCALE_PERCENTAGE -pos-percent 2 98 -palette-name Gray_Interp -disp-pos true -disp-neg true -disp-zero true')
    if mapname == "Thickness":
        os.system('wb_command -metric-math "abs(thickness)" ' + tempgii + ' -var thickness ' + tempgii)
        os.system('wb_command -metric-palette ' + tempgii + ' MODE_AUTO_SCALE_PERCENTAGE -pos-percent 4 96 -interpolate true -palette-name videen_style -disp-pos true -disp-neg false -disp-zero false')
    
    os.system('mv ' + tempgii + ' ' + gii_fname)
    
    return

def freesurf_fix_topology(wm_surf_fname, C, freesurf_subdir):
    if C == 'CORTEX_LEFT':
        hemi = 'lh'
    else:
        hemi = 'rh'
    
    wm_inflate_fname = wm_surf_fname.replace('.vtk','_inflate.vtk')
    if not os.path.exists(wm_inflate_fname):
        os.system('mirtk deform-mesh ' + wm_surf_fname + ' ' + wm_inflate_fname + ' -inflate-brain')

    wm_surf_gii = vtktogii(wm_surf_fname, 'ANATOMICAL', 'GRAY_WHITE' , C)
    wm_inflate_gii = vtktogii(wm_inflate_fname, 'INFLATED', 'GRAY_WHITE' , C)
    
    wm_sphere_gii = wm_inflate_gii.replace('_inflate.surf.gii', '_sphere.surf.gii')
    if not os.path.exists(wm_sphere_gii):
        os.system('mris_sphere -q ' + wm_inflate_gii + ' ' + wm_sphere_gii)

    # Copy gii's to freesurfer directory so that I can run mris_fix_topology (Super Hacky way of doing this)
    os.system('mris_convert ' + wm_surf_gii + ' ' + freesurf_subdir + 'surf/' + hemi + '.smoothwm')
    os.system('mris_convert ' + wm_surf_gii + ' ' + freesurf_subdir + 'surf/' + hemi + '.orig')
    os.system('mris_convert ' + wm_inflate_gii + ' ' + freesurf_subdir + 'surf/' + hemi + '.inflated')
    os.system('mris_convert ' + wm_sphere_gii + ' ' + freesurf_subdir + 'surf/' + hemi + '.qsphere')
    
    # Run mris_fix_topology in the temp freesurf folder
    freesurf_sub = os.path.basename(os.path.normpath(freesurf_subdir))
    print("Freesurf Sub:" + freesurf_sub)
    wm_orig_fix_fname = wm_surf_fname.replace('.vtk','_fix.vtk')
    if not os.path.exists(wm_orig_fix_fname):
        os.system('mris_fix_topology ' + freesurf_sub + ' ' + hemi )
        os.system('mris_convert ' + freesurf_subdir + 'surf/' + hemi + '.orig ' + hemi + '.orig.vtk')
        os.system('mv ' + freesurf_subdir + 'surf/' + hemi + '.orig.vtk ' + wm_orig_fix_fname) 

    return wm_orig_fix_fname

def freesurfer_mris_sphere(wm_gii, wm_inflate2_gii, wm_sphere_gii, hemi, C, freesurf_subdir):
    fs_wm = freesurf_subdir + 'surf/' + hemi + '.white'
    fs_smoothwm = freesurf_subdir + 'surf/' + hemi + '.smoothwm'
    fs_inflate = freesurf_subdir + 'surf/' + hemi + '.inflated'
    
    os.system('mris_convert ' + wm_gii + ' ' + fs_wm)
    os.system('mris_convert ' + wm_gii + ' ' + fs_smoothwm)
    os.system('mris_convert ' + wm_inflate2_gii + ' ' + fs_inflate)
 
    #os.system('mris_inflate ' + fs_wm + ' ' + fs_inflate)

    fs_sphere = freesurf_subdir + 'surf/' + hemi + '.sphere'
    os.system('mris_sphere ' + fs_inflate + ' ' + fs_sphere)

    #fs_sphere_reg = freesurf_subdir + 'surf/' + hemi + '.sphere.reg'
    #os.system('mris_register -nocurv -inflated ' + fs_sphere + ' /usr/local/freesurfer/average/' + hemi + '.average.curvature.filled.buckner40.tif ' + fs_sphere_reg)
    
    os.system('mris_convert ' + fs_sphere + ' ' + hemi + '.sphere.gii')
    os.system('mv ' + freesurf_subdir + 'surf/' + hemi + '.sphere.gii' + ' ' + wm_sphere_gii)
    os.system('wb_command -set-structure ' + wm_sphere_gii + ' ' + C + ' -surface-type SPHERICAL -surface-secondary-type GRAY_WHITE')
    
    return

def freesurf_mris_inflate(wm_gii, wm_inflate_gii, wm_sulc_gii, C):
    os.system('mris_inflate -n 20 -sulc tmpsulc ' + wm_gii + ' ' + wm_inflate_gii)
    os.system('wb_command -set-structure ' + wm_inflate_gii + ' ' + C + ' -surface-type INFLATED -surface-secondary-type GRAY_WHITE')
    os.system('mris_convert -c rh.tmpsulc ' + wm_inflate_gii + ' ' + wm_sulc_gii)
    os.system('wb_command -set-structure ' + wm_sulc_gii + ' ' + C )
    
    return

def msm_reg(wm_sphere_gii, ref_sphere_gii, wm_sulc_gii, ref_sulc_gii, hemi, conf_file, trans_reg=None):
    # Do I need this???
    #print("Rescaling Sphere for Registration")
    #wm_sphere_rescaled_gii = wm_sphere_gii.replace('.surf.gii','-rescaled.surf.gii')
    #os.system('wb_command -surface-modify-sphere ' + wm_sphere_gii + ' 100 ' + wm_sphere_rescaled_gii + ' -recenter')
    
    print("Registering Sphere")
    # HCP configuration files
    msmConfDir = '/home/graham/Software/NeuroSoftware/HCPpipelines-master/MSMConfig/'
    
    if not trans_reg:
        os.system('msm --conf=' + msmConfDir + conf_file + ' --inmesh=' + wm_sphere_gii + ' --refmesh=' + ref_sphere_gii + ' --indata=' + wm_sulc_gii + ' --refdata=' + ref_sulc_gii + ' --out=' + hemi + '. --verbose')
    else:
        print("Using trans reg")
        os.system('msm --conf=' + msmConfDir + conf_file + ' --inmesh=' + wm_sphere_gii + ' --refmesh=' + ref_sphere_gii + ' --indata=' + wm_sulc_gii + ' --refdata=' + ref_sulc_gii + ' --trans=' + trans_reg + ' --out=' + hemi + '. --verbose')
    
    return

def generate_initial_lr_wm(subDir, thisSub, suffix, surfDir):

    tissueDir = subDir + 'tissue_classification/'
    segDir = subDir + 'subcortical_segmentation/'
    regDir = subDir + 'registration/'

    internal_struct_label = ['LEFT_THALAMUS',
                    'LEFT_CAUDATE',
                    'LEFT_PUTAMEN',
                    'LEFT_GLOBUS',
                    'LEFT_ACCUMBENS',
                    'RIGHT_THALAMUS',
                    'RIGHT_CAUDATE',
                    'RIGHT_PUTAMEN',
                    'RIGHT_GLOBUS',
                    'RIGHT_ACCUMBENS']

    external_struct_label = ['LEFT_HIPPO',
                             'LEFT_AMYGDALA',
                             'RIGHT_HIPPO',
                             'RIGHT_AMYGDALA']

    thalamus_struct_label = ['LEFT_THALAMUS',
                             'RIGHT_THALAMUS']

    # Make mask of internal structures
    pseudo_white = np.zeros(nib.load(segDir + thisSub + '_refined_LEFT_THALAMUS' + fsl_ext()).get_data().shape)
    for thisLabel in internal_struct_label:
        thisStructFile = segDir + thisSub + '_refined_' + thisLabel + fsl_ext()
        pseudo_white[nib.load(thisStructFile).get_data() > 0] = 1

    internal_mask = np.copy(pseudo_white)

    external_mask = np.zeros(nib.load(segDir + thisSub + '_refined_LEFT_THALAMUS' + fsl_ext()).get_data().shape)
    for thisLabel in external_struct_label:
        thisStructFile = segDir + thisSub + '_refined_' + thisLabel + fsl_ext()
        external_mask[nib.load(thisStructFile).get_data() > 0] = 1

    thalamus_label = np.zeros(nib.load(segDir + thisSub + '_refined_LEFT_THALAMUS' + fsl_ext()).get_data().shape)
    for thisLabel in thalamus_struct_label:
        thisStructFile = segDir + thisSub + '_refined_' + thisLabel + fsl_ext()
        thalamus_label[nib.load(thisStructFile).get_data() > 0] = 1

    # Get WM segmentation
    fseg = tissueDir + 'tissue_prob_final_out_seg'  + fsl_ext()
    seg_img = nib.load(fseg)
    seg_data = seg_img.get_data()
    pseudo_white[seg_data == 2] = 1
    pseudo_white[external_mask == 1] = 0

    # Read in harvard atlas to separate sub structures
    fharvard = regDir + 'HarvardOxford-sub-prob-1mm_flirt_native'  + fsl_ext()
    harvard_img = nib.load(fharvard)
    harvard_data = harvard_img.get_data()

    fmni =  regDir + 'MNI-prob-1mm_flirt_native'  + fsl_ext()
    mni_img = nib.load(fmni)
    mni_data = mni_img.get_data()

    # Generate Cerrebellum, brainstem and ventricle segmentations
    # Output tissue masks
    ffa = subDir + 'DTI_maps/' + thisSub + suffix + '_FA' + fsl_ext()
    fdwi = subDir + 'meanDWI/' + thisSub + '_DNLSAM_GR_EDDY_mean_b1000_n4' + fsl_ext() ## Fix this hard coded suffix
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
    gm_prob = nib.load(tissueDir + 'tissue_prob_final_out_prob_GM'  + fsl_ext()).get_data()
    wm_prob = nib.load(tissueDir + 'tissue_prob_final_out_prob_WM'  + fsl_ext()).get_data()
    csf_prob = nib.load(tissueDir + 'tissue_prob_final_out_prob_CSF'  + fsl_ext()).get_data()
    vent_prob = harvard_data[:,:,:,2] + harvard_data[:,:,:,13]
    brain_stem_prob = harvard_data[:,:,:,7]
    cerebellum_prob = mni_data[:,:,:,1]/100

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
    nib.save(nib.Nifti1Image(cerebellum_prob, dwi_img.affine), atropos_cerebellum_prob)
    
    if not os.path.exists(atropos_gm_label):
        system('Atropos -d 3' +
                ' -a [' + fdwi + ',0.1]' +
                ' -a [' + ffa  + ',0.1]' +
                ' -x ' + fgm_mask +
                ' -i PriorProbabilityImages[2, ' + atropos_gm_base + '%02d' + fsl_ext() + ',0.1,0.0001]' +
                ' -m [0.3, 2x2x2] ' +
                ' -o ' + atropos_gm_label +
                ' -k HistogramParzenWindows[1.0,32]' +
                ' -v')
    else:
        print("GM segmentation already completed")
    gm_label = nib.load(atropos_gm_label).get_data()
    cerebellar_gm = gm_label ==2
    
    # Segment CSF into ventricles / exterior csf 
    atropos_csf_base = surfDir + thisSub + suffix + 'CSF_'
    atropos_csf_label = atropos_csf_base + 'label' + fsl_ext()

    atropos_csf_prob = atropos_csf_base + '01' + fsl_ext()
    nib.save(nib.Nifti1Image(csf_prob, dwi_img.affine), atropos_csf_prob)
    
    atropos_vent_prob = atropos_csf_base + '02' + fsl_ext()
    nib.save(nib.Nifti1Image(vent_prob, dwi_img.affine), atropos_vent_prob)
    
    if not os.path.exists(atropos_csf_label):
        system('Atropos -d 3' +
                ' -a [' + fdwi + ',0.1]'+
                ' -a [' + ffa  + ',0.1]'+
                ' -x ' + fcsf_mask +
                ' -i PriorProbabilityImages[2, ' + atropos_csf_base + '%02d' + fsl_ext() + ',0.1,0.0001]' +
                ' -m [0.3, 2x2x2] ' +
                ' -o ' + atropos_csf_label +
                ' -k HistogramParzenWindows[1.0,32]' +
                ' -v ')
    else:
        print("CSF segmentation already completed")
    csf_label = nib.load(atropos_csf_label).get_data()
    ventricles = csf_label == 2


    # Segment WM into Cerebellum WM / Brain Stem WM / Cerebrum WM
    atropos_wm_base = surfDir + thisSub + suffix + 'WM_'
    atropos_wm_label = atropos_wm_base + 'label' + fsl_ext()

    atropos_wm_prob = atropos_wm_base + '01' + fsl_ext()
    nib.save(nib.Nifti1Image(wm_prob, dwi_img.affine), atropos_wm_prob)

    atropos_brain_stem_prob = atropos_wm_base + '02' + fsl_ext()
    nib.save(nib.Nifti1Image(brain_stem_prob, dwi_img.affine), atropos_brain_stem_prob)

    atropos_cerebellum_prob = atropos_wm_base + '03' + fsl_ext()
    nib.save(nib.Nifti1Image(cerebellum_prob, dwi_img.affine), atropos_cerebellum_prob)

    if not os.path.exists(atropos_wm_label):
        system('Atropos -d 3' +
                ' -a [' + fdwi + ',0.1]' +
                ' -a [' + ffa  + ',0.1]'+
                ' -x ' + fwm_mask +
                ' -i PriorProbabilityImages[3, ' + atropos_wm_base + '%02d' + fsl_ext() + ',0.1,0.0001]' +
                ' -m [0.3, 2x2x2] ' +
                ' -o ' + atropos_wm_label +
                ' -k HistogramParzenWindows[1.0,32]' + 
                ' -v')
    else:
        print("WM segmentation already completed")
    wm_label = nib.load(atropos_wm_label).get_data()
    brain_stem = wm_label == 2
    cerebellar_wm = wm_label == 3

    pseudo_white[cerebellar_gm] = 0
    pseudo_white[cerebellar_wm] = 0
    pseudo_white[ventricles] = 1
    pseudo_white[brain_stem] = 0

    # Resample segmentations to 0.75 mm isotropic
    old_vsize = seg_img.header.get_zooms()[:3]
    new_vsize = (0.75, 0.75, 0.75)
    seg_data, new_affine = reslice(seg_data, seg_img.affine, old_vsize, new_vsize, order=0)
    pseudo_white, new_affine = reslice(pseudo_white, seg_img.affine, old_vsize, new_vsize, order=0)
    internal_mask, new_affine = reslice(internal_mask, seg_img.affine, old_vsize, new_vsize, order=0)
    external_mask, new_affine = reslice(external_mask, seg_img.affine, old_vsize, new_vsize, order=0)
    thalamus_label, new_affine = reslice(thalamus_label, seg_img.affine, old_vsize, new_vsize, order=0)
    brain_stem, new_affine = reslice(brain_stem, seg_img.affine, old_vsize, new_vsize, order=0)
    ventricles, new_affine = reslice(ventricles, seg_img.affine, old_vsize, new_vsize, order=0)
    cerebellar_gm, new_affine = reslice(cerebellar_gm, seg_img.affine, old_vsize, new_vsize, order=0)
    cerebellar_wm, new_affine = reslice(cerebellar_wm, seg_img.affine, old_vsize, new_vsize, order=0)
    dwi_data_reslice, new_affine = reslice(dwi_img.get_data(), dwi_img.affine, old_vsize, new_vsize, order=3)
    
    fdwi_resamp = surfDir + thisSub + suffix + 'DWI_resamp' + fsl_ext()
    nib.save(nib.Nifti1Image(dwi_data_reslice, new_affine), fdwi_resamp)
    fdwi_neg = surfDir + thisSub + suffix + 'DWI_neg' + fsl_ext()
    nib.save(nib.Nifti1Image(-1 * dwi_data_reslice, new_affine), fdwi_neg)


    # Segmentation is not perfect at the moment, remove any pieces of the pseudo wm mask on islands 
    label_im, nb_labels = ndimage.label(pseudo_white == 1)
    sizes = ndimage.sum(pseudo_white == 1, label_im, range(nb_labels + 1))
    size_img = sizes[label_im]
    pseudo_white[size_img == max(sizes)] = 1
    pseudo_white[size_img != max(sizes)] = 0

    # Fill in holes from crossing fibre regions
    pseudo_white[binary_fill_holes(pseudo_white)] = 1
    pseudo_white[binary_closing(pseudo_white, iterations = 3)] = 1
    pseudo_white[binary_fill_holes(pseudo_white)] = 1

    fpseudo_white = surfDir + thisSub + suffix + '_initwm' + fsl_ext()
    nib.save(nib.Nifti1Image(np.int8(pseudo_white), new_affine), fpseudo_white)

    # Separate into left and right hemispheres based on the center of mass of the thalamus
    thalamus_center = np.uint8(np.round(center_of_mass(thalamus_label)))

    lh_pseudo_white = np.zeros(pseudo_white.shape)
    lh_pseudo_white[thalamus_center[0]+1:,:,:] = pseudo_white[thalamus_center[0]+1:,:,:]
    
    rh_pseudo_white = np.zeros(pseudo_white.shape)
    rh_pseudo_white[:thalamus_center[0]+1,:,:] = pseudo_white[:thalamus_center[0]+1,:,:]

    # Make interhemispheric mask
    inter_mask = np.ones(seg_data.shape)

    # Shift left hemisphere right/ right hemisphere left and fill in intersection
    lh_shifted = np.roll(lh_pseudo_white, -1, axis=0)
    rh_shifted = np.roll(rh_pseudo_white, 1, axis=0)
    wm_intersect = np.zeros(lh_shifted.shape)
    wm_intersect[np.logical_and(lh_shifted, rh_shifted)] = 1
    inter_mask[wm_intersect==1] = 0
    lh_pseudo_white[wm_intersect ==1] = 0
    rh_pseudo_white[wm_intersect ==1] = 0

    # fill 2 voxels left/right of intersect for inter hemispheric mask
    inter_mask[np.logical_or(np.roll(wm_intersect, -1, axis=0)==1, np.roll(wm_intersect, 1, axis=0)==1)] = 0
    inter_mask[np.logical_or(np.roll(wm_intersect, -2, axis=0)==1, np.roll(wm_intersect, 2, axis=0)==1)] = 0
    inter_mask[external_mask == 1] = 0
    inter_mask[brain_stem] = 0
    inter_mask[thalamus_label == 1] = 0
    inter_mask[ventricles] = 0

    # Segmentation is not perfect at the moment, remove any pieces of the interhempispheric mask on islands 
    label_im, nb_labels = ndimage.label(inter_mask == 0)
    sizes = ndimage.sum(inter_mask == 0, label_im, range(nb_labels + 1))
    size_img = sizes[label_im]
    inter_mask[size_img == max(sizes)] = 0
    inter_mask[size_img != max(sizes)] = 1

    fwm_lh = surfDir + thisSub + suffix + '_initwm_lh' + fsl_ext()
    nib.save(nib.Nifti1Image(np.int8(lh_pseudo_white), new_affine), fwm_lh)

    fwm_rh = surfDir + thisSub + suffix + '_initwm_rh' + fsl_ext()
    nib.save(nib.Nifti1Image(np.int8(rh_pseudo_white), new_affine), fwm_rh)

    # move inter mask a voxel to the left and right
    finter = surfDir + thisSub + suffix + '_inter_mask' + fsl_ext()
    nib.save(nib.Nifti1Image(np.int8(inter_mask), new_affine), finter)

    # Output distance maps for surface based cortical segmentation
    tissueDir = subDir + 'tissue_classification/'
    fwm_prob = tissueDir + 'tissue_prob_final_out_prob_WM' + fsl_ext()
    wm_prob_img = nib.load(fwm_prob)
    wm_prob = wm_prob_img.get_data()
    fgm_prob = tissueDir + 'tissue_prob_final_out_prob_GM' + fsl_ext()
    gm_prob = nib.load(fgm_prob).get_data()
    fcsf_prob = tissueDir + 'tissue_prob_final_out_prob_CSF' + fsl_ext()
    csf_prob = nib.load(fcsf_prob).get_data()

    # Resample tissue probability maps
    wm_prob, new_affine = reslice(wm_prob, seg_img.affine, old_vsize, new_vsize, order=3)
    gm_prob, new_affine = reslice(gm_prob, seg_img.affine, old_vsize, new_vsize, order=3)
    csf_prob, new_affine = reslice(csf_prob, seg_img.affine, old_vsize, new_vsize, order=3)

    # WM/cortex distance map
    wm_dist = -1 * wm_prob
    wm_dist = wm_dist + gm_prob
    wm_dist = wm_dist + csf_prob
    wm_dist[binary_dilation(binary_dilation(internal_mask == 1))] = -1
    wm_dist[binary_dilation(binary_dilation(ventricles))] = -1
    wm_dist[binary_dilation(binary_dilation(brain_stem))] = 1

    fwm_dist = surfDir + thisSub + suffix + '_wm_cortex_dist' + fsl_ext()
    nib.save(nib.Nifti1Image(wm_dist, new_affine), fwm_dist)
    
    # cortex/CSF distance map
    cortex_dist = np.zeros(seg_data.shape)
    cortex_dist[seg_data == 2] = -1
    cortex_dist[seg_data == 1] = -1
    cortex_dist[seg_data == 3] = 1
    cortex_dist[binary_dilation(binary_dilation(internal_mask == 1))] = -1
    cortex_dist[binary_dilation(binary_dilation(ventricles))] = -1
    cortex_dist[binary_dilation(binary_dilation(brain_stem))] = 1
    cortex_dist[cerebellar_gm] = 1
    cortex_dist[cerebellar_wm] = 1
    cortex_dist[binary_dilation(binary_dilation(external_mask == 1))] = 1

    fcortex_dist = surfDir + thisSub + suffix + '_cortex_csf_dist' + fsl_ext()
    nib.save(nib.Nifti1Image(cortex_dist, new_affine), fcortex_dist)
    
    ## Estimate boundary FA
    #fwm_edge = fpseudo_white.replace(fsl_ext(),'_edge' + fsl_ext())
    #os.system('fslmaths ' + fpseudo_white + ' -edge -bin -mas ' + fpseudo_white + ' ' + fwm_edge)

    #edge_img = nib.load(fwm_edge)
    #edge_data = edge_img.get_data()

    #ffa = subDir + 'DTI_maps/' + thisSub + suffix + '_FA' + fsl_ext()
    #fa_img = nib.load(ffa)
    #fa_data = fa_img.get_data()
    #new_vsize = (0.75, 0.75, 0.75)
    #fa_data, new_affine = reslice(fa_data, fa_img.affine, fa_img.header.get_zooms()[:3], new_vsize, order=1)
    #fa_target = np.percentile(fa_data[edge_data == 1],20)

    return fwm_lh, fwm_rh, finter, fwm_dist, fcortex_dist, fdwi_resamp, fdwi_neg

def generate_tensor_force_map(subDir, thisSub, preproc_suffix, suffix, surfDir, fa_target=0.2):

    ftb_force = surfDir + thisSub + suffix + '_tensor-based-force' + fsl_ext()

    fseg = subDir + 'voxel_based_segmentation/' + thisSub + suffix + '_tissue_seg_init' + fsl_ext()
    seg_img = nib.load(fseg)
    seg_data = seg_img.get_data()
    
    old_vsize = seg_img.header.get_zooms()[:3]
    new_vsize = (0.75, 0.75, 0.75)
    seg_data, new_affine = reslice(seg_data, seg_img.affine, old_vsize, new_vsize, order=0)

    ffa = subDir + 'DTI_maps/' + thisSub + suffix + '_FA' + fsl_ext()
    fa_img = nib.load(ffa)
    fa_data = fa_img.get_data()
    fa_data, new_affine = reslice(fa_data, fa_img.affine, old_vsize, new_vsize, order=1)
    fa_data[seg_data == 0] = 0

    if preproc_suffix == '':
        fdwi = subDir + 'meanDWI/' + thisSub + '_mean_b1000_n4' + fsl_ext()
    else:
        fdwi = subDir + 'meanDWI/' + thisSub + '_' + preproc_suffix + '_mean_b1000_n4' + fsl_ext()
    dwi_img = nib.load(fdwi)
    dwi_data = dwi_img.get_data()
    dwi_data, new_affine = reslice(dwi_data, dwi_img.affine, old_vsize, new_vsize, order=1)
    dwi_data[seg_data == 0] = 0 
    
    # remove cerrebellum and brain-stem from dwi image
    remove_ind = [10, 4, 9]
    # Segmentation is not perfect at the moment, remove any pieces of the individual segmentations on islands 
    for thisRemoveInd in remove_ind:
        label_im, nb_labels = ndimage.label(seg_data == thisRemoveInd)
        sizes = ndimage.sum(seg_data == thisRemoveInd, label_im, range(nb_labels + 1))
        size_img = sizes[label_im]
        dwi_data[size_img == max(sizes)] = 1.0
     
    finter = surfDir + thisSub + suffix + '_inter_mask' + fsl_ext()
    inter_img = nib.load(finter)
    inter_mask = inter_img.get_data()

    # tensor force-based map
    force_data = -fa_data + fa_target
    
    neg_force_ind = [11, 6, 5]
    for thisNegForceInd in neg_force_ind:
        #label_im, nb_labels = ndimage.label(seg_data == thisNegForceInd)
        #sizes = ndimage.sum(seg_data == thisNegForceInd, label_im, range(nb_labels + 1))
        #size_img = sizes[label_im]
        #force_data[seg_data == max(sizes)] = -1.0 # fill in ventricles
        force_data[seg_data == thisNegForceInd] = -1.0 

    pos_force_ind = [3,10,4,9] # fill in exterior CSF
    for thisPosForceInd in pos_force_ind:
        #label_im, nb_labels = ndimage.label(seg_data == thisPosForceInd)
        #sizes = ndimage.sum(seg_data == thisPosForceInd, label_im, range(nb_labels + 1))
        #size_img = sizes[label_im]
        #force_data[size_img == max(sizes)] = 1.0
        force_data[seg_data == thisPosForceInd] = 1.0

    inter_mask_resamp, new_affine = reslice(inter_mask, inter_img.affine, inter_img.header.get_zooms()[:3], new_vsize, order=0)
    force_data[inter_mask_resamp == 0] = -1.0
    #force_data[seg_data == 10] = 1.0 # fill in brain stem
    #force_data[seg_data == 4] = 1.0 # fill in cerrebellum GM
    #force_data[seg_data == 9] = 1.0 # fill in cerrebellum WM
    nib.save(nib.Nifti1Image(force_data, new_affine), ftb_force)

    fdwi_mask = surfDir + thisSub + suffix + '_meanb1000_n4_mask' + fsl_ext()
    nib.save(nib.Nifti1Image(dwi_data, new_affine), fdwi_mask)

    fdwi_neg = fdwi_mask.replace(fsl_ext(),'_neg' + fsl_ext())
    nib.save(nib.Nifti1Image(-dwi_data, new_affine), fdwi_neg)

    fWMGM_bound = surfDir + thisSub + suffix + '_wmgm_bound' + fsl_ext()
    wmgm_bound = np.zeros(force_data.shape)
    pseudo_wm_ind = [2,5,6,7,8,11]
    cortex_ind = [1]
    exterior_csf_ind = [3,4,9,10,0]
    
    wmgm_bound[np.isin(seg_data, pseudo_wm_ind)] = -1
    wmgm_bound[np.isin(seg_data, cortex_ind)] = 1
    wmgm_bound[np.isin(seg_data, exterior_csf_ind)] = 1
    nib.save(nib.Nifti1Image(wmgm_bound, new_affine), fWMGM_bound)

    fGMCSF_bound = surfDir + thisSub + suffix + '_gmcsf_bound' + fsl_ext()
    gmcsf_bound = np.zeros(force_data.shape)
    gmcsf_bound[np.isin(seg_data, pseudo_wm_ind)] = -1
    gmcsf_bound[np.isin(seg_data, cortex_ind)] = -1
    gmcsf_bound[np.isin(seg_data, exterior_csf_ind)] = 1
    nib.save(nib.Nifti1Image(gmcsf_bound, new_affine), fGMCSF_bound)

    fCSF_bound = surfDir + thisSub + suffix + '_csf_bound' + fsl_ext()
    csf_bound = np.zeros(force_data.shape)
    csf_bound[np.isin(seg_data, exterior_csf_ind)] = 1
    nib.save(nib.Nifti1Image(csf_bound, new_affine), fCSF_bound)

    return ftb_force, fdwi_mask, fdwi_neg, fWMGM_bound, fGMCSF_bound, fCSF_bound

def generate_surfaces_from_dwi(outDir, thisSub, preproc_suffix, shell_suffix, freesurf_subdir):
    print("Surfing: " + thisSub)
    subDir = outDir + '/' + thisSub + '/'
    surfDir = subDir + 'surf/'
    if not os.path.exists(surfDir):
        os.system('mkdir ' + surfDir)
   
    if preproc_suffix != '':
        suffix = '_' + preproc_suffix + '_' + shell_suffix 
    else:
        suffix = '_' + shell_suffix 

    wm_surf_fname = surfDir + thisSub + suffix + '_wm.vtk'
    
    wm_lh_fname, wm_rh_fname, finter, fwm_dist, fcortex_dist, fdwi_resamp, fdwi_neg = generate_initial_lr_wm(subDir, thisSub, suffix, surfDir)
    #ftb_force, fdwi_mask, fdwi_neg, fWMGM_bound, fGMCSF_bound, fCSF_bound = generate_tensor_force_map(subDir, thisSub, preproc_suffix, suffix, surfDir, fa_target = fa_target)

    # Output Composite MD map
    #dtiDir = subDir + 'DTI_maps/'
    #fmd = dtiDir + thisSub + suffix + '_MD' + fsl_ext()
    #md_img = nib.load(fmd)
    #md_data = md_img.get_data()

    #ffa = dtiDir + thisSub + suffix + '_FA' + fsl_ext()
    #fa_data = nib.load(ffa).get_data()

    #md_plus_fa = md_data*1000 + fa_data
    #md_plus_fa_neg = -1 * md_plus_fa

    #fmd_plus_fa = surfDir + thisSub + suffix + '_MDplusFA' + fsl_ext()
    #nib.save(nib.Nifti1Image(md_plus_fa, md_img.affine), fmd_plus_fa)

    #fmd_plus_fa_neg = surfDir + thisSub + suffix + '_MDplusFA_neg' + fsl_ext()
    #nib.save(nib.Nifti1Image(md_plus_fa_neg, md_img.affine), fmd_plus_fa_neg)

    #tissueDir = subDir + 'tissue_classification/'
    #fwm_prob = tissueDir + 'tissue_prob_final_out_prob_WM' + fsl_ext()
    #wm_prob_img = nib.load(fwm_prob)
    #wm_dist = -1 * wm_prob_img.get_data()
    #fgm_prob = tissueDir + 'tissue_prob_final_out_prob_GM' + fsl_ext()
    #wm_dist = wm_dist + nib.load(fgm_prob).get_data()
    #fcsf_prob = tissueDir + 'tissue_prob_final_out_prob_CSF' + fsl_ext()
    #wm_dist = wm_dist + nib.load(fcsf_prob).get_data()

    #fwm_dist = fwm_prob.replace(fsl_ext(), '_dist' + fsl_ext())
    #nib.save(nib.Nifti1Image(wm_dist, wm_prob_img.affine), fwm_dist)

    if not os.path.exists(wm_surf_fname):

        wm_lh_surf_fname = surfDir + thisSub + suffix + '_wm_lh.vtk'
        os.system('mirtk extract-surface ' + wm_lh_fname + ' ' + wm_lh_surf_fname + ' -isovalue 0.5')
        os.system('mirtk extract-connected-points ' + wm_lh_surf_fname + ' ' + wm_lh_surf_fname )
        os.system('mirtk remesh-surface ' + wm_lh_surf_fname + ' ' + wm_lh_surf_fname + ' -min-edgelength 0.5 -max-edgelength 1.0')
        
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
        
        wm_lh_fix_fname = freesurf_fix_topology(wm_lh_surf_fname, 'CORTEX_LEFT', freesurf_subdir) # Outputs wm_surf_fname_fix
        wm_lh_surf = sutil.read_surf_vtk(wm_lh_fix_fname)

        leftLabels =  np.ones((wm_lh_surf.GetNumberOfPoints(),))
        vtk_pntList = numpy_to_vtk(leftLabels)
        vtk_pntList.SetName("HemiLabels")
        wm_lh_surf.GetPointData().AddArray(vtk_pntList)

        wm_rh_surf_fname =surfDir + thisSub + suffix + '_wm_rh.vtk'
        os.system('mirtk extract-surface ' + wm_rh_fname + ' ' + wm_rh_surf_fname + ' -isovalue 0.5')
        os.system('mirtk extract-connected-points ' + wm_rh_surf_fname + ' ' + wm_rh_surf_fname )
        os.system('mirtk remesh-surface ' + wm_rh_surf_fname + ' ' + wm_rh_surf_fname + ' -min-edgelength 0.5 -max-edgelength 1.0')
        
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
        
        wm_rh_fix_fname = freesurf_fix_topology(wm_rh_surf_fname, 'CORTEX_RIGHT', freesurf_subdir)
        wm_rh_surf = sutil.read_surf_vtk(wm_rh_fix_fname) 

        rightLabels =  np.ones((wm_rh_surf.GetNumberOfPoints(),))*2
        vtk_pntList = numpy_to_vtk(rightLabels)
        vtk_pntList.SetName("HemiLabels")
        wm_rh_surf.GetPointData().AddArray(vtk_pntList)
        
        sutil.write_surf_vtk(wm_rh_surf, wm_rh_surf_fname.replace('.vtk','_test.vtk'))


        #combine left and right surfaces
        appender = vtk.vtkAppendPolyData()
        appender.AddInputData(wm_lh_surf)
        appender.AddInputData(wm_rh_surf)
        appender.Update()
        wm_surf = appender.GetOutput()
        sutil.write_surf_vtk(wm_surf, wm_surf_fname)

        # Initalize white matter surface with mask
        wm_surf = sutil.read_surf_vtk(wm_surf_fname)
        imaskDil_ImgVTK, imaskDil_HdrVTK, sformMat = sutil.read_image(finter)
        wm_surf = sutil.interpolate_voldata_to_surface(wm_surf, imaskDil_ImgVTK, sformMat, pntDataName='Mask',categorical=True)
        wm_surf = sutil.interpolate_voldata_to_surface(wm_surf, imaskDil_ImgVTK, sformMat, pntDataName='InitialStatus', categorical=True)
        sutil.write_surf_vtk(wm_surf, wm_surf_fname) 
       
        # Note that because the way 1's and 0's are used here, erode will make the mask bigger
        os.system('mirtk open-scalars ' + wm_surf_fname + ' ' + wm_surf_fname + ' -a Mask -n 5 ')
        os.system('mirtk dilate-scalars ' + wm_surf_fname + ' ' + wm_surf_fname + ' -a Mask -n 5 ')
        os.system('mirtk erode-scalars ' + wm_surf_fname + ' ' + wm_surf_fname + ' -a Mask -n 6 ')

        os.system('mirtk open-scalars ' + wm_surf_fname + ' ' + wm_surf_fname + ' -a InitialStatus -n 5 ')
        os.system('mirtk dilate-scalars ' + wm_surf_fname + ' ' + wm_surf_fname + ' -a InitialStatus -n 5 ')
        os.system('mirtk erode-scalars ' + wm_surf_fname + ' ' + wm_surf_fname + ' -a InitialStatus -n 6 ')

    # Refine WM Surface
    #wm_surf_refined_fname = wm_surf_fname.replace('.vtk', '_WithTBforce.vtk')
    #print("Generating WM Surface with tensor-based force: FA Target " + str(fa_target))
    wm_final_fname = wm_surf_fname.replace('.vtk','_final.vtk')
    if not os.path.exists(wm_final_fname):
        # With Tensor-based force
        #os.system('mirtk deform-mesh ' + wm_surf_fname + ' ' + wm_surf_refined_fname + ' -image ' + fdwi_mask + ' -edge-distance 1.6 -edge-distance-smoothing 1 -edge-distance-median 1 -edge-distance-averaging 4 2 1 -distance-image ' + ftb_force + ' -distance 1.6 -distance-smoothing 1 -distance-averaging 4 2 1 -distance-measure normal -optimizer EulerMethod -step 0.2 -steps 100 -epsilon 1e-6 -delta 0.001 -min-active 1% -reset-status -nointersection -fast-collision-test -min-width 0.1 -min-distance 0.1 -repulsion 4 -repulsion-distance 0.5 -repulsion-width 1.0 -curvature 4.0 -gauss-curvature 1.0 -gauss-curvature-minimum .1 -gauss-curvature-maximum .2 -gauss-curvature-outside 0.5 -edge-distance-type ClosestMaximum -threads 32 -remesh 1 -min-edge-length 0.5 -max-edge-length 1.0')
        
        ## without tensor-based force
        #os.system('mirtk deform-mesh ' + wm_surf_fname + ' ' + wm_surf_refined_fname.replace('.vtk','_noTensor.vtk') + ' -image ' + fwm_prob_neg + ' -edge-distance 2.0 -edge-distance-smoothing 1 -edge-distance-median 1 -edge-distance-averaging 4 2 1 -optimizer EulerMethod -step 0.4 -steps 100 -epsilon 1e-6 -delta 0.001 -min-active 1% -reset-status -nointersection -fast-collision-test -min-width 0.1 -min-distance 0.1 -repulsion 4 -repulsion-distance 0.5 -repulsion-width 1.0 -remesh 1 -min-edge-length 0.5 -max-edge-length 1.0 -curvature 4.0 -gauss-curvature 1.0 -gauss-curvature-minimum .1 -gauss-curvature-maximum .2 -gauss-curvature-outside 0.5 -edge-distance-type ClosestMaximum -threads 8')
        
        os.system('mirtk deform-mesh ' + wm_surf_fname + ' ' + wm_final_fname + ' -distance-image ' + fwm_dist +' -distance 1.6 -distance-smoothing 1 -distance-averaging 4 2 1 -distance-measure normal -optimizer EulerMethod -step 0.2 -steps 100 -epsilon 1e-6 -delta 0.001 -min-active 1% -reset-status -nointersection -fast-collision-test -min-width 0.1 -min-distance 0.1 -repulsion 4 -repulsion-distance 0.5 -repulsion-width 1.0 -remesh 1 -min-edge-length 0.5 -max-edge-length 1.0 -curvature 4.0 -gauss-curvature 1.0 -gauss-curvature-minimum .1 -gauss-curvature-maximum .2 -gauss-curvature-outside 0.5 -edge-distance-type ClosestMaximum -threads 8')
    else:
        print("white surface already refined")


    #print("Refining White Surface without distance force")
    #wm_final_fname = wm_surf_fname.replace('.vtk','_final.vtk')
    #if not os.path.exists(wm_final_fname):
    #    #os.system('mirtk deform-mesh ' + wm_surf_refined_fname + ' ' + wm_final_fname + ' -image ' + fdwi_mask + ' -edge-distance 1.6 -edge-distance-smoothing 1 -edge-distance-median 1 -edge-distance-averaging 1 -distance-image ' + fCSF_bound + ' -distance 1.6 -distance-smoothing 1 -distance-averaging 1 -distance-measure normal -optimizer EulerMethod -step 0.2 -steps 300 -epsilon 1e-6 -delta 0.001 -min-active 1% -reset-status -nointersection -fast-collision-test -min-width 0.1 -min-distance 0.1 -repulsion 4 -repulsion-distance 1.0 -repulsion-width 2.0 -curvature 4.0 -gauss-curvature 1.0 -gauss-curvature-minimum .1 -gauss-curvature-maximum .2 -gauss-curvature-outside 0.5 -edge-distance-type ClosestMaximum -threads 32 -remesh 1 -min-edge-length 0.5 -max-edge-length 1.0')

    
    ## Move surface outward edge based on CSF segmentation
    print("Expanding white surface Using GM/CSF Boundary")
    wm_expand_fname = wm_final_fname.replace('.vtk','_expandGMCSF.vtk')
    if not os.path.exists(wm_expand_fname):
        #os.system('mirtk deform-mesh ' + wm_final_fname + ' ' + wm_expand_fname + ' -normal 0.25 -step 0.1 -optimizer EulerMethod -steps 500 -max-displacement 1.5 -non-self-intersection True -nointersection -fast-collision-test -min-active 5% -delta 0.0001 -min-width 0.1 -min-distance 0.1 -repulsion 2.0 -repulsion-distance 0.5 -repulsion-width 1.0 -curvature 8 -gauss-curvature 0.8 -gauss-curvature-minimum .1 -gauss-curvature-maximum .4 -gauss-curvature-inside 2 -negative-gauss-curvature-action inflate  -threads 16')
        os.system('mirtk deform-mesh ' + wm_final_fname + ' ' + wm_expand_fname + ' -image ' + fdwi_neg + ' -edge-distance 0.5 -edge-distance-smoothing 1 -edge-distance-median 1 -edge-distance-averaging 4 2 1 -distance-image ' + fcortex_dist + ' -distance 1.0 -distance-smoothing 1 -distance-averaging 4 2 1 -distance-measure normal -optimizer EulerMethod -step 0.2 -steps 100 200 -epsilon 1e-6 -delta 0.001 -min-active 5% -reset-status -nointersection -fast-collision-test -min-width 0.1 -min-distance 0.1 -repulsion 2.0 -repulsion-distance 0.5 -repulsion-width 1.0 -curvature 4.0 -edge-distance-type ClosestMaximum -edge-distance-max-intensity -1 -gauss-curvature 1.6 -gauss-curvature-minimum .1 -gauss-curvature-maximum .4 -gauss-curvature-inside 2 -negative-gauss-curvature-action inflate  -threads 8')
    else:
        print("Expanded surface already generated")

    ## Generate pial surface
    print("Refining CSF/GREY border")
    pial_fname =  surfDir + thisSub + suffix + '_pial.vtk'
    if not os.path.exists(pial_fname):
        os.system('mirtk deform-mesh ' + wm_expand_fname + ' ' + pial_fname + ' -image ' + fdwi_neg + ' -edge-distance 1.0 -edge-distance-smoothing 1 -edge-distance-median 1 -edge-distance-averaging 1 -optimizer EulerMethod -step 0.2 -steps 300 -epsilon 1e-6 -delta 0.001 -min-active 5% -reset-status -nointersection -fast-collision-test -min-width 0.1 -min-distance 0.1 -repulsion 2.0 -repulsion-distance 0.5 -repulsion-width 1.0 -curvature 4.0 -edge-distance-type ClosestMaximum -edge-distance-max-intensity -1 -gauss-curvature 1.6 -gauss-curvature-minimum .1 -gauss-curvature-maximum .4 -gauss-curvature-inside 2 -negative-gauss-curvature-action inflate  -threads 32')
        #os.system('mirtk deform-mesh ' + wm_final_fname + ' ' + pial_fname + ' -distance-image ' + fcortex_dist +' -distance 1.0 -distance-smoothing 1 -distance-averaging 4 2 1 -distance-measure normal -optimizer EulerMethod -step 0.2 -steps 300 -epsilon 1e-6 -delta 0.001 -min-active 5% -reset-status -nointersection -fast-collision-test -min-width 0.1 -min-distance 0.1 -repulsion 2.0 -repulsion-distance 0.5 -repulsion-width 1.0 -curvature 4.0 -edge-distance-type ClosestMaximum -gauss-curvature 1.6 -gauss-curvature-minimum .1 -gauss-curvature-maximum .4 -gauss-curvature-inside 2 -negative-gauss-curvature-action inflate -threads 8')
    else:
        print("pial remesh already made")

    # Split WM surface
    wm_surf = sutil.read_surf_vtk(wm_final_fname)
    lh_wm_fname = wm_final_fname.replace('.vtk','_lh.vtk')
    rh_wm_fname = wm_final_fname.replace('.vtk','_rh.vtk')    
    if not os.path.exists(lh_wm_fname) or not os.path.exists(rh_wm_fname):
        [lh_wm,rh_wm] = sutil.split_surface_by_label(wm_surf)
        sutil.write_surf_vtk(lh_wm, lh_wm_fname)
        sutil.write_surf_vtk(rh_wm, rh_wm_fname)
    lh_wm_gii = vtktogii(lh_wm_fname, 'ANATOMICAL', 'GRAY_WHITE' , 'CORTEX_LEFT')
    rh_wm_gii = vtktogii(rh_wm_fname, 'ANATOMICAL', 'GRAY_WHITE' , 'CORTEX_RIGHT')

    # Split Pial surface
    pial_surf = sutil.read_surf_vtk(pial_fname)
    lh_pial_fname = pial_fname.replace('.vtk','_lh.vtk')
    rh_pial_fname = pial_fname.replace('.vtk','_rh.vtk')
    if not os.path.exists(lh_pial_fname) or not os.path.exists(rh_pial_fname):
        [lh_pial,rh_pial] = sutil.split_surface_by_label(pial_surf)
        sutil.write_surf_vtk(lh_pial, lh_pial_fname)
        sutil.write_surf_vtk(rh_pial, rh_pial_fname)
    lh_pial_gii = vtktogii(lh_pial_fname, 'ANATOMICAL', 'PIAL' , 'CORTEX_LEFT')
    rh_pial_gii = vtktogii(rh_pial_fname, 'ANATOMICAL', 'PIAL' , 'CORTEX_RIGHT')
    
    for hemi in ['lh','rh']:
        if hemi == 'lh':
            hemi_letter = 'L'
            C = 'CORTEX_LEFT'
            wm_vtk = lh_wm_fname
            pial_vtk = lh_pial_fname
            wm_gii = lh_wm_gii
            pial_gii = lh_pial_gii
        else:
            hemi_letter = 'R'
            C = 'CORTEX_RIGHT'
            wm_vtk = rh_wm_fname
            pial_vtk = rh_pial_fname
            wm_gii = rh_wm_gii
            pial_gii = rh_pial_gii

        # Export mask as gii file
        mask_gii = surfDir + thisSub + suffix + 'mask_' + hemi + '.native.shape.gii'
        giimap(wm_vtk, mask_gii, 'Mask', 'Mask', C, flip=False)
        os.system("wb_command -metric-math 'ceil(mask)' " + mask_gii + ' -var mask ' + mask_gii)

        # Curvature using White matter surface
        print("Processing Curvature on WM surface")
        curv_vtk = wm_vtk.replace('.vtk','.curvature.vtk')
        curv_gii = wm_gii.replace('.surf.gii','.curvature.native.shape.gii')
        if not os.path.exists(curv_vtk):
            os.system('mirtk calculate-surface-attributes ' + wm_vtk + ' ' + curv_vtk + ' -H Curvature -smooth-weighting Combinatorial -smooth-iterations 10 -vtk-curvatures') 
            os.system('mirtk calculate ' + curv_vtk + ' -mul -1 -scalars Curvature -out ' + curv_vtk)
            giimap(curv_vtk, curv_gii, 'Curvature', 'Curvature', C)
            os.system('wb_command -metric-dilate ' + curv_gii + ' ' + wm_gii + ' 10 ' + curv_gii + ' -nearest')
  
        # Get MidThickness Surface
        print("Getting mid thickness surface")
        midthick_vtk = pial_vtk.replace('pial','midthick')
        if not os.path.exists(midthick_vtk):
            os.system('mid-surface ' + wm_vtk + ' ' + pial_vtk + ' ' + midthick_vtk + ' -ascii')
            midthick_gii = vtktogii(midthick_vtk, 'ANATOMICAL', 'MIDTHICKNESS',C)
       
        # Calculating Cortical Thickness    
        print("Calculating Thickness")
        thick_vtk = wm_vtk.replace('.vtk','.thickness.vtk')
        dist1_vtk = wm_vtk.replace('.vtk','.dist1.vtk')
        dist2_vtk = wm_vtk.replace('.vtk','.dist2.vtk')
        thick_gii = wm_gii.replace('.surf.gii','.thickness.native.shape.gii')
        if not os.path.exists(thick_vtk):
            os.system('mirtk evaluate-distance ' + wm_vtk + ' ' + pial_vtk + ' ' + dist1_vtk + ' -name Thickness')
            os.system('mirtk evaluate-distance ' + pial_vtk + ' ' + wm_vtk + ' ' + dist2_vtk + ' -name Thickness')
            os.system('mirtk calculate ' + dist1_vtk + ' -scalars Thickness -add ' + dist2_vtk + ' Thickness -div 2 -o ' + thick_vtk)
            giimap(thick_vtk, thick_gii, 'Thickness', 'Thickness', C)
            os.system('wb_command -metric-dilate ' + thick_gii + ' ' + wm_gii + ' 10 ' + thick_gii + ' -nearest')
       
        print("Inflating WM surface for visualization")
        wm_smooth_gii = wm_gii.replace('.surf.gii','.gaussSmooth.surf.gii')
        veryinflate_gii = wm_gii.replace('.surf.gii','.veryinflated.surf.gii')
        if not os.path.exists(veryinflate_gii):
            os.system('mirtk smooth-surface ' + wm_gii + ' ' + wm_smooth_gii + ' -points -iterations 1000 -gaussian 10.0') 
            os.system('wb_command -set-structure ' + wm_smooth_gii + ' ' + C + ' -surface-type VERY_INFLATED -surface-secondary-type GRAY_WHITE')
            os.system('wb_command -surface-inflation ' + wm_gii + ' ' + wm_smooth_gii + ' 30 1.0 2 1.05 ' + veryinflate_gii)

        print("Inflating WM surface for registration")
        wm_inflate2_vtk = wm_vtk.replace('.vtk','.inflated_for_sphere.vtk')
        wm_sulc_gii = thick_gii.replace('thickness','sulc')
        if not os.path.exists(wm_inflate2_vtk):
            wm_nostatus_vtk = wm_vtk.replace('.vtk','.nostatus.vtk')
            os.system('mirtk delete-pointset-attributes ' + wm_vtk + ' ' + wm_nostatus_vtk + ' -name InitialStatus')
            os.system('mirtk deform-mesh ' + wm_nostatus_vtk + ' ' + wm_inflate2_vtk + ' -inflate-brain -track SulcalDepth')
            giimap(wm_inflate2_vtk, wm_sulc_gii, 'SulcalDepth', 'Sulc', C)
        wm_inflate2_gii = vtktogii(wm_inflate2_vtk, 'INFLATED', 'GRAY_WHITE',C)

        #print("Extracting Spherical Surface")
        #wm_sphere_gii = wm_inflate2_gii.replace('.surf.gii','.sphere.surf.gii')
        #if not os.path.exists(wm_sphere_gii):
        #    freesurfer_mris_sphere(wm_gii, wm_inflate2_gii, wm_sphere_gii, hemi, C, freesurf_subdir)
        #    
        #print("Register using MSM spherical registration FSL 6.0")
        #fs_LR_dir = '/home/graham/Software/NeuroSoftware/HCPpipelines-master/global/templates/standard_mesh_atlases/'
        #ref_sphere_gii = fs_LR_dir + 'fsaverage.' + hemi_letter + '_LR.spherical_std.164k_fs_LR.rotzyx.surf.gii'
        #ref_sulc_gii = fs_LR_dir +  hemi_letter + '.refsulc.164k_fs_LR.shape.gii'

        #out_base = surfDir + thisSub + suffix + hemi
        #out_reg = out_base + '.sphere.reg.surf.gii'

        #if not os.path.exists(out_reg):
        #
        #    # Then register using sulc maps using the above registration as starting positioni
        #    wm_sulc_mask_gii = wm_sulc_gii.replace('.native.shape.gii','-masked.native.shape.gii')
        #    os.system("wb_command -metric-math 'x*mask' " + wm_sulc_mask_gii + ' -var x ' + wm_sulc_gii + ' -var mask ' + mask_gii + ' -repeat')
        #    
        #    #wm_smooth_sulc_gii = wm_sulc_gii.replace('.native.shape.gii','-smoothed.native.shape.gii')
        #    #os.system('wb_command -metric-smoothing ' + wm_gii + ' ' + wm_sulc_mask_gii + ' 8 ' + wm_smooth_sulc_gii)

        #    #wm_sulc_only_gii = wm_sulc_gii.replace('.native.shape.gii','-sulcmask.native.shape.gii')
        #    #os.system("wb_command -metric-math 'x * (x < 0)' " + wm_sulc_only_gii + ' -var x ' + wm_smooth_sulc_gii)

        #    msm_reg(wm_sphere_gii, ref_sphere_gii, wm_sulc_gii, ref_sulc_gii, out_base, 'DiffusionSurfaceConfiguration')
        #    os.system('wb_command -set-structure ' + out_reg + ' ' + C + ' -surface-type SPHERICAL -surface-secondary-type GRAY_WHITE')
   
       
        #ATLASDIR = '/media/graham/DATA2/Cortical_Anisotropy/Data/Atlas/'
        ##atlas_fname =  ATLASDIR + 'fslr.' + hemi + '.aparc.label.gii '
        #atlas_list = ['','']
        #atlas_list[0] = ATLASDIR + 'Conte69_atlas-v2.LR.164k_fs_LR.wb/Conte69_atlas_164k_wb/parcellations_VGD11b.' + hemi_letter + '.164k_fs_LR.label.gii'
        #atlas_list[1] = ATLASDIR + 'Conte69_atlas-v2.LR.164k_fs_LR.wb/Conte69_atlas_164k_wb/brodmann-lobes.' + hemi_letter + '.164K_fs_LR.label.gii'
        #
        #out_parc_list = ['','']
        #out_parc_list[0] = out_base + hemi + '.parcellations_VGD11b.164k_fs_LR.label.gii'
        #out_parc_list[1] = out_base + hemi + '.brodmann-lobes.164K_fs_LR.label.gii'
        #for atlas_ind in range(0,len(atlas_list)):
        #    atlas_fname = atlas_list[atlas_ind]
        #    out_parc = out_parc_list[atlas_ind]
        #
        #    #if not os.path.exists(out_parc):
        #    os.system('wb_command -label-resample ' + atlas_fname + ' ' + ref_sphere_gii + ' ' + out_reg + ' BARYCENTRIC ' + out_parc + ' -largest')
        #    #os.system('msmresample ' + out_reg + ' ' + out_parc + ' -labels ' + atlas_fname + ' -project ' + wm_sphere_gii)
    
        #out_brodmann = out_parc_list[1]
        #out_brodmann_dilate = out_brodmann.replace('.label.gii','-dilate.label.gii')
        #os.system('wb_command -label-dilate ' + out_brodmann + ' ' + wm_gii + ' 20 ' + out_brodmann_dilate)
    
        #out_brodmann_dilate_mask = out_brodmann_dilate.replace('.label.gii','-mask.label.gii')
        #os.system("wb_command -label-mask " + out_brodmann_dilate + ' ' + mask_gii + ' ' + out_brodmann_dilate_mask)

    return

