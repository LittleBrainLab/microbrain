# microbrain
microbrain is a fully automated diffusion MRI analysis pipeline for measurement of grey matter microstructure.  This pipeline requires the installation of multiple neuroimaging analysis software packages, thus the environment is provided via a dockerfile in this repository. Importantly this is a VERY BETA Version use with extreme caution. For trouble shooting help contact Graham Little (graham.little.phd@gmail.com)

# References
Subcortical Gray Matter Segmentation - Little 2023 Bioarxiv, Automated Surface-Based Segmentation of Deep Gray Matter Regions Based on Diffusion Tensor Images Reveals Unique Age Trajectories Over the Healthy Lifespan

Cortical Gray Matter Segmentation - Little 2021 NeuroImage, Automated cerebral cortex segmentation based solely on diffusion tensor imaging for investigating cortical anisotropy

Caudolenticular Gray Matter Bridge Tractography - Little 2023 Bioarxiv, Mapping caudolenticular gray matter bridges in the human brain striatum through diffusion magnetic resonance imaging and tractography

# Installation
```
pip install -r requirements.txt
pip install -e .
```

# Example Usage
```
microbrain_setup.py -s path_to_subject_directory -i path_to_dicom_directory
microbrain_run.py -s path_to_subject_directory -b [0,1000] --gibbs --eddy --mbrain-seg --mbrain-cort
```
