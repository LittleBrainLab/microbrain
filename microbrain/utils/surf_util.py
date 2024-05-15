import numpy as np
# import cupy as cp
import sys
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from vtk.util.numpy_support import numpy_to_vtk

output_stream = sys.stdout

def write_surf_vtp(surfVTK, fname):
    """
    writes a vtk polydata object to a vtp file

    Parameters
    ----------
    surfVTK: vtkPolyData object
    fname: string
        filename for the output vtp file

    Returns
    -------
    None
    """

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(fname)
    writer.SetInputData(surfVTK)
    writer.Write()

    return


def read_surf_vtp(fname):
    """
    reads a vtp file and returns a vtk polydata object

    Parameters
    ----------
    fname: string
        filename for the input vtp file

    Returns
    -------
    surfVTK: vtkPolyData object
    """

    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(fname)
    reader.Update()

    surfVTK = reader.GetOutput()

    return surfVTK


def read_surf_vtk(fname):
    """
    reads a vtk file and returns a vtk polydata object

    Parameters
    ----------
    fname: string
        filename for the input vtk file

    Returns
    -------
    surfVTK: vtkPolyData object
    """

    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(fname)
    reader.Update()

    surfVTK = reader.GetOutput()

    return surfVTK


def write_surf_vtk(surfVTK, fname):
    """
    writes a vtk polydata object to a vtk file

    Parameters
    ----------
    surfVTK: vtkPolyData object
    fname: string
        filename for the output vtk file

    Returns
    -------
    None
    """

    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(fname)
    writer.SetInputData(surfVTK)
    # version compatible multiple versions of MIRTK
    writer.SetFileVersion(42)
    writer.Write()

    return


def read_image(fname):
    """
    reads a nifti file and returns a vtk image object

    Parameters
    ----------
    fname: string
        filename for the input nifti file

    Returns
    -------
    imageVTK: vtkImageData object
    header: vtkNIFTIHeader object
    transVTK: vtkTransform object
    """

    reader = vtk.vtkNIFTIImageReader()
    reader.SetFileName(fname)
    reader.SetTimeAsVector(True)
    reader.Update()

    transVTK = vtk.vtkTransform()
    transVTK.SetMatrix(reader.GetSFormMatrix())
    transVTK.Update()

    imageVTK = reader.GetOutput()

    return imageVTK, reader.GetNIFTIHeader(), transVTK


def write_image(image, header, sformMat, fname):
    """
    writes a vtk image object to a nifti file

    Parameters
    ----------
    image: vtkImageData object
    header: vtkNIFTIHeader object
    sformMat: vtkTransform object
    fname: string
        filename for the output nifti file

    Returns
    -------
    None
    """

    writer = vtk.vtkNIFTIImageWriter()
    writer.SetInputData(image)
    writer.SetFileName(fname)
    writer.SetNIFTIHeader(header)

    # For some reason vtk writer ignores PixDim[0] if set to -1 causing output to flip
    if header.GetPixDim(0) == -1:
        writer.SetQFac(-1)
    elif header.GetPixDim(0) == 1:
        writer.SetQFac(1)
    else:
        writer.SetQFac(-1)

    writer.Write()

    return


def surf_to_volume_mask(fdwi, fmesh, inside_val, fout):
    """
    Outputs a voxel label for a given 3D mesh, label includes all voxels whose center is within the mesh

    Parameters
    ----------
    fdwi: string
        filename for 3Dnifti file which defines the voxel grid for the label
    fmesh: string
        vtk file for the mesh that will be converted to a voxel label
    inside_val: integer
        value assigned to the voxel label
    fout: string
        filename of output nifti file containing the label

    Returns
    -------
    None
    """

    # Transform the vtk mesh to native space
    surfVTK = read_surf_vtk(fmesh)
    vtkImage, vtkHeader, sformMat = read_image(fdwi)
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

    write_image(out_image, vtkHeader, sformMat, fout)

    return


def split_surface_by_label(surfVTK, label=None, label_name="HemiLabels"):
    """
    Splits a surface into multiple surfaces based on label

    Parameters
    ----------
    surfVTK: vtkPolyData object
        input surface
    label: list of integers
        list of labels on polydata to split the surface by
    label_name: string
        name of the label array in the polydata

    Returns
    -------
    surfList: list of vtkPolyData objects
        list of surfaces split by label
    """

    surfVTK.BuildLinks()

    label_array = vtk_to_numpy(surfVTK.GetPointData().GetArray(label_name))
    if label == None:
        label_list = np.unique(label_array)
    else:
        label_list = label

    surfList = []
    for thisLabel in label_list:
        thisSurf = vtk.vtkPolyData()
        # thisSurf.DeepCopy(surfVTK)
        # thisSurf.BuildLinks()
        thisSurf.Allocate(700000, 1000)
        pointIDs = np.array(range(0, label_array.shape[0]))
        newPointIds = pointIDs[np.array(label_array) == thisLabel]
        pointIdMap = np.array(range(0, newPointIds.shape[0]))

        pointIDoffset = np.min(newPointIds)
        newPointArray = vtk_to_numpy(surfVTK.GetPoints().GetData())
        newPointArray = newPointArray[newPointIds, :]

        new_vtkArray = numpy_to_vtk(newPointArray)
        new_vtkPoints = vtk.vtkPoints()
        new_vtkPoints.SetData(new_vtkArray)
        thisSurf.SetPoints(new_vtkPoints)

        for pointID in newPointIds:
            cellIds = vtk.vtkIdList()
            surfVTK.GetPointCells(pointID, cellIds)
            if pointID % 1000 == 0:
                output_stream.write(
                    "Splitting Surface Label " + str(thisLabel) + ": Vert " + str(pointID) + "    \r")
                output_stream.flush()
            sys.stdout.flush()

            for cellInd in range(0, cellIds.GetNumberOfIds()):
                thisCell = surfVTK.GetCell(cellIds.GetId(cellInd))
                thisSurf.Squeeze()

                # If all the points in the cell have the same label then add it
                cellPnts = np.array([thisCell.GetPointIds().GetId(
                    i) for i in range(0, thisCell.GetPointIds().GetNumberOfIds())])
                cell_label = label_array[cellPnts]

                if np.all(cell_label == thisLabel):
                    cellPnts = np.array(
                        [pointIdMap[cellPnt == newPointIds] for cellPnt in cellPnts])
                    cellPntIds = vtk.vtkIdList()
                    for i in range(0, cellPnts.shape[0]):
                        cellPntIds.InsertNextId(cellPnts[i, 0])
                    thisSurf.InsertNextCell(5, cellPntIds)
                else:
                    print("Found a boundary")

        output_stream.write('\n')

        arrayNum = surfVTK.GetPointData().GetNumberOfArrays()
        print("Transfering " + str(arrayNum) + " of Arrays")
        for arrayInd in range(0, arrayNum):
            oldPointData = surfVTK.GetPointData().GetArray(arrayInd)
            oldData = vtk_to_numpy(oldPointData)
            newPointData = numpy_to_vtk(oldData[newPointIds])
            newPointData.SetName(oldPointData.GetName())
            thisSurf.GetPointData().AddArray(newPointData)

        thisSurf.Squeeze()

        # clean mesh
        cleanFilter = vtk.vtkCleanPolyData()
        cleanFilter.SetInputData(thisSurf)
        cleanFilter.Update()
        thisSurf = cleanFilter.GetOutput()

        # remove duplicate polys
        removeFilter = vtk.vtkRemoveDuplicatePolys()
        removeFilter.SetInputData(thisSurf)
        removeFilter.Update()
        thisSurf = removeFilter.GetOutput()

        surfList = surfList + [thisSurf]

    return surfList


def compute_mid_surface(wmSurf, pialSurf):
    """
    Computes the mid surface between two surfaces

    Parameters
    ----------
    wmSurf: vtkPolyData object
        white matter surface
    pialSurf: vtkPolyData object
        pial surface

    Returns
    -------
    medialSurf: vtkPolyData object
        mid surface
    """

    medialSurf = vtk.vtkPolyData()
    medialSurf.DeepCopy(wmSurf)

    wmPnts = vtk_to_numpy(wmSurf.GetPoints().GetData())
    pialPnts = vtk_to_numpy(pialSurf.GetPoints().GetData())
    medialPnts = (pialPnts - wmPnts)/2.0 + wmPnts

    new_vtkArray = numpy_to_vtk(medialPnts)
    new_vtkPoints = vtk.vtkPoints()
    new_vtkPoints.SetData(new_vtkArray)
    medialSurf.SetPoints(new_vtkPoints)

    return medialSurf


def interpolate_voldata_to_surface(surfVTK, meanDiffVTK, sformMat, pntDataName='Labels', categorical=False):
    """
    Interpolates volume data to surface

    Parameters
    ----------
    surfVTK: vtkPolyData object
        input surface
    meanDiffVTK: vtkImageData object
        input volume data
    sformMat: vtkTransform object
        sform matrix for the volume data
    pntDataName: string
        name of the output point data array
    categorical: boolean
        flag for categorical data

    Returns
    -------
    surfVTK: vtkPolyData object
        surface with interpolated volume data
    """

    # Transform poly data to voxel_space
    sformInv = vtk.vtkTransform()
    sformInv.DeepCopy(sformMat)
    sformInv.Inverse()
    sformInv.Update()

    transformPD = vtk.vtkTransformPolyDataFilter()
    transformPD.SetTransform(sformInv)
    transformPD.SetInputData(surfVTK)
    transformPD.Update()
    surfVTK_voxel = transformPD.GetOutput()

    probe = vtk.vtkProbeFilter()
    probe.SetSourceData(meanDiffVTK)
    probe.SetInputData(surfVTK_voxel)
    if categorical == True:
        probe.CategoricalDataOn()
    probe.Update()
    surfVTK_interp_voxel = probe.GetOutput()
    interpData = surfVTK_interp_voxel.GetPointData().GetArray("NIFTI")

    # Needed to put in a tie break because probe was returning 0.5 decimal values in categorical case
    # Note that value is cast to int32 this is because numpy_to_vtk blew up with int8 input.
    if categorical:
        interpData = numpy_to_vtk(np.int32(np.ceil(vtk_to_numpy(interpData))))

    interpData.SetName(pntDataName)
    surfVTK.GetPointData().AddArray(interpData)

    return surfVTK
