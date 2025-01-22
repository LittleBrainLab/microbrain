# microbrain: Automated Gray Matter Analysis for Diffusion MRI

microbrain is an Python package for analyzing gray matter in diffusion MRI data. It provides a pipeline for preprocessing, modeling, segmentation, and tractography to study microstructural properties of the brain.&#x20;

---

## Key Features

- **Preprocessing**: Includes denoising, brain extraction, and distortion correction.
- **Modeling**: Automatically fits diffusion tensor imaging (DTI) maps with multi-shell data.
- **Segmentation**: Performs surface-based segmentation of cortical and subcortical gray matter on diffusion MRI in native imaging space.
- **Tractography**: Implements advanced methods to reconstruct gray matter cellular bridges.

---

## Quick Start with Docker

### Prerequisites

- Install [Docker](https://www.docker.com/).

### Running microbrain

1. Pull the Docker image:

   ```bash
   docker pull scilus/microbrain:2024
   ```

   Note the microbrain container is large requiring +16GB of RAM during extraction, consider allocating more memory if needed.

2. To run microbrain you need:

   - A valid FreeSurfer license file usually located at /usr/local/freesurfer/license.txt
   - A temporary folder (path/to/tmp)
   - A folder contain diffusion data (path/to/data)

3. Run the container:

   ```bash
   docker run --rm -it \
     -v /path/to/data:/data \
     -v /path/to/tmp:/tmp \
     -v /path/to/freesurfer/license.txt:/usr/local/freesurfer/license.txt \
     scilus/microbrain:2024
   ```

4. Inside the container, execute the pipeline:

   ```bash
   python microbrain_setup.py -s /data/subject_dir -i /data/dicom_dir
   python microbrain_run.py -s /data/subject_dir -b [0,1000] --gibbs --dmppca --eddy --mbrain-seg --mbrain-cort
   python microbrain_scil_track_cellular_bridges.py -s /data/subject_dir
   ```

   Note: the microbrain\_setup.py script is used if dicom data is used as input otherwise setup the subject folder structure manually (see below)

5. Retrieve the processed data from `/path/to/data` on your host machine.

---

## Setting Up the Subject Folder

The `microbrain_setup.py` script automates the creation of the required folder structure for processing from dicom images.  Alternatively, you can manually create the directory structure (see below).  Note that for microbrain to function correctly, the \<subject-directory> and \<subject-id> must be the same.

### Using `microbrain_setup.py`

Run the following command:

```bash
python microbrain_setup.py -s <subject-directory> -i <dicom-directory>
```

- `<subject-directory>`: Path where the subject's data will be organized.
- `<dicom-directory>`: Directory containing the DICOM files to be converted to NIFTI format.

This script will:

- Convert DICOM files to NIFTI format.
- Organize the data into an `orig` directory with the following structure and files:
  - `<subject-directory>/orig/<subject-id>.nii.gz`: The primary diffusion-weighted imaging (DWI) NIFTI file.
  - `<subject-directory>/orig/<subject-id>.bval`: The associated b-value file.
  - `<subject-directory>/orig/<subject-id>.bvec`: The associated b-vector file.
  - `<subject-directory>/orig/<subject-id>.json`: (optional) The JSON metadata file for the DWI scan.
- Ensure the `<subject-directory>` matches the `<subject-id>` for compatibility.
- Optionally handle reverse phase-encode or field map data if specified using additional flags.

### Manual Folder Structure

If you prefer to manually create the subject folder structure or you are working from NIFTI files, create the above data structure with the necessary files and the proper naming convention before running the pipeline.

---

## Running Outside of Docker

The `microbrain` pipeline relies on several third-party neuroimaging and scientific libraries to perform its functions. These include FSL for preprocessing and distortion correction, ANTs for image registration, FreeSurfer for cortical segmentation (requires a valid license file), DIPY for diffusion modeling, scilpy for tractography, and MIRTK for surface-based segmentation. To run the pipeline outside Docker, all dependencies must be installed and configured manually, including setting environment variables for FSL and FreeSurfer. For convenience, using the Docker container with pre-installed dependencies is recommended, however here commands for using `microbrain` in a native environment.

### Step 1: Setup

Prepare the subject directory:

```bash
python microbrain_setup.py -s <subject-directory> -i <dicom-directory>
```

### Step 2: Process and Segment Diffusion Data

Run preprocessing, diffusion modeling and gray matter segmentation:

```bash
python microbrain_run.py -s <subject-directory> -b [0,1000] --dmppca --eddy --mbrain-seg --mbrain-cort
```

### Step 3: Perform caudolenticular gray matter bridge (CLGB) tractography

Reconstruct CLGBs:

```bash
python microbrain_scil_track_cellular_bridges.py -s <subject-directory>
```

---

## Outputs

### Outputs folders from `microbrain_run.py`

- `<subject-directory>/preprocessed/`: denoised, gibbs corrected, eddy corrected images.&#x20;
- `<subject-directory>/DTI_maps/`: parameter maps output from diffusion tensor fitting.
- `<subject-directory>/meanDWI/`: N4-corrected Mean diffusion-weighted images for each shell.
- `<subject-directory>/registration/`: Files for registration from MNI to native imaging space using ANTs.
- `<subject-directory>/subcortical_segmentation/mesh_output/`: Final subcortical gray matter segmentations in mesh format.
- `<subject-directory>/surf/mesh_segmentation/`: Final cortical surface segmentations in mesh format.

### Outputs folders from `microbrain_scil_track_cellular_bridges.py`

- `<subject-directory>/tracking/fodf/`: files used for fiber orientation distribution function fitting and an fODF map.
- `<subject-directory>/tracking/tracking\_mask/`: a binary white matter mask generated from the microbrain segmentations computed earlier
- `<subject-directory>/tracking/gm\_cellular\_bridge\_tractography/`: all intermediary files for the analysis, importantly the CLGB tractography reconstructions for each hemisphere can be found in the tractograms\_gm\_cellular\_bridges folder.

---

## Support

For issues or feature requests, please create a GitHub issue or contact the author:

- Graham Little ([graham.little.phd@gmail.com](mailto\:graham.little.phd@gmail.com))

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.

