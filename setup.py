import glob
from setuptools import setup, find_packages

opts = dict(scripts=glob.glob("scripts/*.py"),
            packages=find_packages(),
            data_files=[('data/tissuepriors',
                         ["data/tissuepriors/avg152T1_gm_resampled.nii.gz",
                          "data/tissuepriors/avg152T1_wm_resampled.nii.gz",
                          "data/tissuepriors/avg152T1_csf_resampled.nii.gz"]),
                          ('data/TEMP_FS',
                           glob.glob("data/TEMP_FS/*"))],
            include_package_data=True)


if __name__ == '__main__':
    setup(**opts)
