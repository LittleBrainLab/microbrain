# microbrain
microbrain is a fully automated diffusion MRI analysis pipeline for measurement of grey matter microstructure.  Note this pipeline requires the installation of multiple neuroimaging/python analysis software packages (see setup_instructions.txt). Importantly this is a VERY BETA Version use with extreme caution. For trouble shooting help contact Graham Little (graham.little.phd@gmail.com)

Subcortical/Cortical segmentation methods (Little 2023 Bioarxiv, Little 2021 NeuroImage) will be added soon.

# Installation
pip install -r requirements.txt
pip install -e .

# Example Usage
microbrain_setup.py -s path_to_subject_directory -i path_to_dicom_directory
microbrain_run.py -s path_to_subject_directory -b [0,1000] --gibbs --eddy --mbrain-seg --mbrain-cort
