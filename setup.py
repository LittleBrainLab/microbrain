import glob
from setuptools import setup

opts = dict(scripts=glob.glob("scripts/*.py"),
            data_files=[('microbrain/data/tissuepriors',
                         ["microbrain/data/tissuepriors/avg152T1_gm_resampled.nii.gz",
                          "microbrain/data/tissuepriors/avg152T1_wm_resampled.nii.gz",
                          "microbrain/data/tissuepriors/avg152T1_csf_resampled.nii.gz"]),
                          ('microbrain/data/TEMP_FS',
                           glob.glob("microbrain/data/TEMP_FS/*")),
                        ('microbrain/data/fsl/',
                         ['microbrain/data/fsl/standard/FSL_HCP1065_FA_1mm.nii.gz',
                          'microbrain/data/fsl/atlases/HarvardOxford/HarvardOxford-sub-prob-1mm.nii.gz'
                          'microbrain/data/fsl/atlases/MNI/MNI-prob-1mm.nii.gz'
                          'microbrain/data/fsl/atlases/JHU/JHU-ICBM-labels-1mm.nii.gz'])],
            include_package_data=True)

if __name__ == '__main__':
    setup(**opts)
