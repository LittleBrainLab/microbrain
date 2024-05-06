import glob
from setuptools import setup

opts = dict(scripts=glob.glob("scripts/*.py"),
            data_files=[('microbrain/data/tissuepriors',
                         ["microbrain/data/tissuepriors/avg152T1_gm_resampled.nii.gz",
                          "microbrain/data/tissuepriors/avg152T1_wm_resampled.nii.gz",
                          "microbrain/data/tissuepriors/avg152T1_csf_resampled.nii.gz"]),
                          ('microbrain/data/TEMP_FS',
                           glob.glob("microbrain/data/TEMP_FS/*"))],
            include_package_data=True)

if __name__ == '__main__':
    setup(**opts)
