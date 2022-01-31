import numpy as np
#import cupy as cp
import sys
import multiprocessing
import os
import nibabel as nib
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from vtk.util.numpy_support import numpy_to_vtk
import matplotlib.pyplot as plt
import scipy

#import pycuda.autoinit
#import pycuda.driver as drv

from skimage import measure
from functools import partial
from time import time
#from pycuda.compiler import SourceModule
#from mayavi import mlab
from multiprocessing import Pool

output_stream = sys.stdout
procDir = '../../Data/Test_Data/CB_BRAIN_040_v2_smoothed/'

globalSurf = []
globalCellLocator = []
# !!! have to change a bunch of code in here so that I don't change pointers :(
# Started to modify functions to take VTK structures as input instead, these saved a ton of disk space as well
def write_surf_vtp(surfVTK, fname):
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(fname)
    writer.SetInputData(surfVTK)
    writer.Write()

    return

def read_surf_vtp(fname):
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(fname)
    reader.Update()

    surfVTK = reader.GetOutput()

    return surfVTK

def read_surf_vtk(fname):
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(fname)
    reader.Update()

    surfVTK = reader.GetOutput()

    return surfVTK


def write_surf_vtk(surfVTK, fname):
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(fname)
    writer.SetInputData(surfVTK)
    writer.Write()

    return

def read_image(fname):

    reader = vtk.vtkNIFTIImageReader()
    reader.SetFileName(fname)
    reader.SetTimeAsVector(True) 
    reader.Update()
    
    transVTK = vtk.vtkTransform()
    transVTK.SetMatrix(reader.GetSFormMatrix())
    transVTK.Update()
    
    imageVTK = reader.GetOutput()

    return imageVTK, reader.GetNIFTIHeader(), transVTK

def add_initial_points(surfVTK):
    
    pointList = range(0,surfVTK.GetNumberOfPoints())
    vtk_pntList = numpy_to_vtk(pointList)
    vtk_pntList.SetName("InitialPoints")
    surfVTK.GetPointData().AddArray(vtk_pntList)
    
    return surfVTK

def split_surface_by_label(surfVTK,label=None):

    surfVTK.BuildLinks()

    label_array = vtk_to_numpy(surfVTK.GetPointData().GetArray("HemiLabels"))
    if label == None:
        label_list = np.unique(label_array)
    else:
        label_list = label

        
    print(label_list)
    surfList = []
    for thisLabel in label_list:
        thisSurf = vtk.vtkPolyData()
        #thisSurf.DeepCopy(surfVTK)
        #thisSurf.BuildLinks()
        thisSurf.Allocate(700000,1000)
        pointIDs = np.array(range(0, label_array.shape[0])) 
        newPointIds = pointIDs[np.array(label_array) == thisLabel]
        pointIdMap = np.array(range(0, newPointIds.shape[0]))

        pointIDoffset = np.min(newPointIds)
        newPointArray = vtk_to_numpy(surfVTK.GetPoints().GetData())
        newPointArray = newPointArray[newPointIds,:]

        new_vtkArray = numpy_to_vtk(newPointArray)
        new_vtkPoints = vtk.vtkPoints()
        new_vtkPoints.SetData(new_vtkArray)
        thisSurf.SetPoints(new_vtkPoints)
        
        for pointID in newPointIds:
            cellIds = vtk.vtkIdList()
            surfVTK.GetPointCells(pointID,cellIds)
            if pointID%1000 == 0:
                output_stream.write("Splitting Surface Label " + str(thisLabel) + ": Vert " + str(pointID) + "    \r")
                output_stream.flush()
            sys.stdout.flush()

            
            for cellInd in range(0,cellIds.GetNumberOfIds()):
                thisCell = surfVTK.GetCell(cellIds.GetId(cellInd))
                thisSurf.Squeeze()
                
                # If all the points in the cell have the same label then add it
                cellPnts = np.array([thisCell.GetPointIds().GetId(i) for i in range(0, thisCell.GetPointIds().GetNumberOfIds())])
                cell_label = label_array[cellPnts]
                
                if np.all(cell_label == thisLabel):
                    cellPnts = np.array([pointIdMap[cellPnt == newPointIds] for cellPnt in cellPnts])
                    cellPntIds = vtk.vtkIdList()
                    for i in range(0, cellPnts.shape[0]):
                            cellPntIds.InsertNextId(cellPnts[i])
                    thisSurf.InsertNextCell(5,cellPntIds)
                else:
                    print("Found a boundary")
        
        output_stream.write('\n')
        
        arrayNum = surfVTK.GetPointData().GetNumberOfArrays()
        print("Transfering " + str(arrayNum) + " of Arrays")
        for arrayInd in range(0,arrayNum):
            oldPointData = surfVTK.GetPointData().GetArray(arrayInd)      
            oldData = vtk_to_numpy(oldPointData)
            newPointData = numpy_to_vtk(oldData[newPointIds])
            newPointData.SetName(oldPointData.GetName())
            thisSurf.GetPointData().AddArray(newPointData)
            
        thisSurf.Squeeze()
        surfList = surfList + [thisSurf]
            
    return surfList

def get_medial_surf_from_wm_gm(wmSurf, pialSurf):
    medialSurf = vtk.vtkPolyData()
    medialSurf.DeepCopy(wmSurf)
    pointList = range(0,wmSurf.GetNumberOfPoints())

    wmPnts = vtk_to_numpy(wmSurf.GetPoints().GetData())
    pialPnts = vtk_to_numpy(pialSurf.GetPoints().GetData())
    medialPnts = np.zeros(wmPnts.shape)
    
    wmPntMap = vtk_to_numpy(wmSurf.GetPointData().GetArray("InitialPoints"))
    pialPntMap = vtk_to_numpy(pialSurf.GetPointData().GetArray("InitialPoints")) 
    print(np.min(wmPntMap), np.max(wmPntMap))
    print(np.min(pialPntMap), np.max(pialPntMap))

    for pointID in pointList:
        if pointID%1000 == 0:
            output_stream.write("Calculating Medial Surface: Vert " + str(pointID) + "    \r")
            output_stream.flush()
        
        sys.stdout.flush()
        thisGMpnt = np.mean(pialPnts[np.where(pialPntMap == pointID), :], axis=0)
        
        if thisGMpnt.shape == (3,):
            thisGMpnt = thisGMpnt[None,:]
        
        if  np.isnan(thisGMpnt).any() or thisGMpnt.shape[0] == 0:
            medialPnts[pointID,:] = wmPnts[pointID,:]
        else:
            allGMpnts = pialPnts[np.squeeze(np.where(pialPntMap == pointID)), :]
            if allGMpnts.shape == (3,):
                allGMpnts = allGMpnts[None,:]
            allWMpnts = np.repeat(wmPnts[pointID,:][None,:],allGMpnts.shape[0],axis=0)
            distance = np.sqrt(np.sum(np.square(allWMpnts - allGMpnts),axis=1))
            
            minPnt = allGMpnts[distance == np.min(distance),:]
            if minPnt.shape[0] == 2:
                minPnt = np.squeeze(minPnt[0,0:3])
            medialPnts[pointID,:] = wmPnts[pointID,:] + (minPnt - wmPnts[pointID,:])*0.5
            
            #distance = np.sqrt(np.sum(np.square(thisWMpnt -thisGMpnt),axis=1))
            #if np.mean(distance) < 7.0:
            #    medialPnts[pointID,:] = wmPnts[pointID,:] + (np.mean(thisGMpnt,axis=0) - wmPnts[pointID,:])/2
            #else:
            #    medialPnts[pointID,:] = wmPnts[pointID,:]

    output_stream.write('\n') 
    new_vtkArray = numpy_to_vtk(medialPnts)
    new_vtkPoints = vtk.vtkPoints()
    new_vtkPoints.SetData(new_vtkArray)
    medialSurf.SetPoints(new_vtkPoints)

    medialSurf = smooth_mesh(medialSurf, n_iter=20, relax = 0.2)

    return medialSurf
 
# Add in function to segment white matter here


# Given a WM segmentation (wm =1) return surface 
def generate_surface_from_wm(wm_img):

    wm_hdr = wm_img.get_header()
    wm_data = wm_img.get_data()

    wm_ind = 1

    surf_verts, surf_faces, surf_normals, values  = measure.marching_cubes_lewiner(wm_data, spacing=wm_hdr.get_zooms(), allow_degenerate=False)

    # Rescale surfaces back to voxel space before transforming into scanner space
    print("Putting surface in voxel space")
    pixdims = wm_hdr.get_zooms()
    surf_verts[:,0] = np.array([vert[0]/pixdims[0] for vert in surf_verts])
    surf_verts[:,1] = np.array([vert[1]/pixdims[1] for vert in surf_verts])
    surf_verts[:,2] = np.array([vert[2]/pixdims[2] for vert in surf_verts])

    # transform vertices using affine transform from header
    thisAffine = np.matrix(wm_img.affine)

    print("Transfoming Surface to Scanner Space")
    tmp_verts = np.transpose(surf_verts)
    tmp_verts = np.concatenate((tmp_verts,np.ones([1,tmp_verts.shape[1]])))
    tmp_verts = thisAffine * np.matrix(tmp_verts)
    surf_verts = np.array(np.transpose(tmp_verts[0:3,:]))

    # Left Hemisphere
    surf = vtk.vtkPolyData()

    #Left Vertices
    points = vtk.vtkPoints()
    for i in range(0,surf_verts.shape[0]):
        points.InsertNextPoint(surf_verts[i,0],surf_verts[i,1],surf_verts[i,2])
    surf.SetPoints(points)

    #Left Faces
    cells = vtk.vtkCellArray()
    for i in range(0,surf_faces.shape[0]):
        cell = vtk.vtkTriangle()
        Ids = cell.GetPointIds()
        for KId in range(surf_faces.shape[1]):
            Ids.SetId(KId, surf_faces[i,KId])
        cells.InsertNextCell(cell)
    surf.SetPolys(cells)

    return surf

# Given a WM segmentation (lh = 1, rh =2) return two surfaces (one for each hemisphere)
def generate_surface_from_wm_2surf(wm_img):

    wm_hdr = wm_img.get_header()
    wm_data = wm_img.get_data()

    rh_ind = 1
    lh_ind = 2

    lh_wm = np.zeros(wm_data.shape)
    lh_wm[wm_data == lh_ind] = 1
    lh_surf = {}
    lh_surf_verts, lh_surf_faces, lh_surf_normals, values  = measure.marching_cubes_lewiner(lh_wm, spacing=wm_hdr.get_zooms(), allow_degenerate=False)
    #lh_surf_verts, lh_surf_faces, lh_surf_normals, values  = measure.marching_cubes_lewiner(lh_wm, 0, spacing=wm_hdr.get_zooms())

    rh_wm = np.zeros(wm_data.shape)
    rh_wm[wm_data == rh_ind] = 1
    rh_surf = {}
    rh_surf_verts, rh_surf_faces, rh_surf_normals, values  = measure.marching_cubes_lewiner(rh_wm, spacing=wm_hdr.get_zooms(), allow_degenerate=False)
    #rh_surf_verts, rh_surf_faces, rh_surf_normals, values  = measure.marching_cubes_lewiner(rh_wm, 0, spacing=wm_hdr.get_zooms())

    # Rescale surfaces back to voxel space before transforming into scanner space
    print("Putting surfaces in voxel spce")
    pixdims = wm_hdr.get_zooms()
    lh_surf_verts[:,0] = np.array([vert[0]/pixdims[0] for vert in lh_surf_verts])
    lh_surf_verts[:,1] = np.array([vert[1]/pixdims[1] for vert in lh_surf_verts])
    lh_surf_verts[:,2] = np.array([vert[2]/pixdims[2] for vert in lh_surf_verts])

    rh_surf_verts[:,0] = np.array([vert[0]/pixdims[0] for vert in rh_surf_verts])
    rh_surf_verts[:,1] = np.array([vert[1]/pixdims[1] for vert in rh_surf_verts])
    rh_surf_verts[:,2] = np.array([vert[2]/pixdims[2] for vert in rh_surf_verts])
    
    # transform vertices using affine transform from header
    thisAffine = np.matrix(wm_img.affine)

    # Scaling already taken care of by the marching cubes algorithm, remove the scaling but leave reflections
    #thisAffine[0,0] = thisAffine[0,0]/np.absolute(thisAffine[0,0])
    #thisAffine[1,1] = thisAffine[1,1]/np.absolute(thisAffine[1,1])
    #thisAffine[2,2] = thisAffine[2,2]/np.absolute(thisAffine[2,2])

    print("Transfoming Left Surface to Scanner Space")
    tmp_lh_verts = np.transpose(lh_surf_verts)
    tmp_lh_verts = np.concatenate((tmp_lh_verts,np.ones([1,tmp_lh_verts.shape[1]])))
    tmp_lh_verts = thisAffine * np.matrix(tmp_lh_verts)
    lh_surf_verts = np.array(np.transpose(tmp_lh_verts[0:3,:]))

    #print("Updating Left Hemisphere Vertex Neighbors")
    #lh_surf = update_neighbors(lh_surf)

    print("Transfoming Right Surface to Scanner Space")  
    tmp_rh_verts = np.transpose(rh_surf_verts)
    tmp_rh_verts = np.concatenate((tmp_rh_verts,np.ones([1,tmp_rh_verts.shape[1]])))
    tmp_rh_verts = thisAffine * np.matrix(tmp_rh_verts)
    rh_surf_verts = np.array(np.transpose(tmp_rh_verts[0:3,:]))

    #print("Updating Right Hemisphere Vertex Neighbors")
    #rh_surf = update_neighbors(rh_surf)

    # Create VTK objects to store meshes
    # Left Hemisphere
    lh_surf = vtk.vtkPolyData()

    #Left Vertices
    lh_points = vtk.vtkPoints()
    for i in range(0,lh_surf_verts.shape[0]):
        lh_points.InsertNextPoint(lh_surf_verts[i,0],lh_surf_verts[i,1],lh_surf_verts[i,2])
    lh_surf.SetPoints(lh_points)

    #Left Faces
    lh_cells = vtk.vtkCellArray()
    for i in range(0,lh_surf_faces.shape[0]):
        cell = vtk.vtkTriangle()
        Ids = cell.GetPointIds()
        for KId in range(lh_surf_faces.shape[1]):
            Ids.SetId(KId, lh_surf_faces[i,KId])
        lh_cells.InsertNextCell(cell)
    lh_surf.SetPolys(lh_cells)


    # Right Hemisphere
    rh_surf = vtk.vtkPolyData()

    # Right Vertices
    rh_points = vtk.vtkPoints()
    for i in range(0, rh_surf_verts.shape[0]):
        rh_points.InsertNextPoint(rh_surf_verts[i,0], rh_surf_verts[i,1], rh_surf_verts[i,2])
    rh_surf.SetPoints(rh_points)

    # Right Faces
    rh_cells = vtk.vtkCellArray()
    for i in range(0, rh_surf_faces.shape[0]):
        cell = vtk.vtkTriangle()
        Ids = cell.GetPointIds()
        for KId in range(rh_surf_faces.shape[1]):
            Ids.SetId(KId, rh_surf_faces[i,KId])
        rh_cells.InsertNextCell(cell)
    rh_surf.SetPolys(rh_cells)

    return lh_surf, rh_surf

# ToDo: this function should calculate FA,MD data, register Harvard atlas/cerrebellum mask to native space
# Maybe: do shell averaging, n4 correct, masking here instead of in preprocessing script
def prepare_diffdata_for_surf_analysis():
    print("Need to make this function")
    return

#ToDo: run segmentation using FA/MD/meanDiff images generated from prepare_diffdata_for_surf_analysis
def segment_wm_on_meanDWI():
    print("Need to make this function")
    return

# translate in direction of normal (vtkWarpScalar wasn't working so I wrote my own)
def translate_surface(surfVTK, moveList=np.array([]), scale=1):

    #Copy the data so you don't modify the source
    surfVTK_warp =  vtk.vtkPolyData()
    surfVTK_warp.DeepCopy(surfVTK)

    # Get points/normals convert to numpy arrays
    if moveList.size == 0:
        moveList = vtk_to_numpy(surfVTK_warp.GetPointData().GetNormals())
    points = vtk_to_numpy(surfVTK_warp.GetPoints().GetData())

    # Update points in PolyData Object
    new_points = moveList*scale + points
    new_vtkArray = numpy_to_vtk(new_points)
    new_vtkPoints = vtk.vtkPoints()
    new_vtkPoints.SetData(new_vtkArray)
    surfVTK_warp.SetPoints(new_vtkPoints)

    return surfVTK_warp

def interpolate_voldata_to_surface(surfVTK, meanDiffVTK, sformMat,pntDataName='Labels', categorical=False):
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

    # Transform back to anatomical space            
    #transformPD = vtk.vtkTransformPolyDataFilter()
    #transformPD.SetTransform(sformMat)
    #transformPD.SetInputData(surfVTK_interp_voxel)
    #transformPD.Update()
    #surfVTK_interp = transformPD.GetOutput()

    #surfVTK_interp.GetPointData().GetArray("NIFTI").SetName(pntDataName) 
    
    return surfVTK

def calculate_gradient_image(vtkImage):
    imageDim = vtkImage.GetSpacing()

    resampler = vtk.vtkImageResample()
    resampler.SetInputData(vtkImage)
    resampler.SetOutputSpacing(imageDim[0], imageDim[0], imageDim[0])
    resampler.Update()
    resampled_image = resampler.GetOutput()

    imageGradient = vtk.vtkImageGradient()
    imageGradient.SetInputData(resampled_image)
    imageGradient.SetDimensionality(3)
    imageGradient.Update()
    gradient_image = imageGradient.GetOutput()

    return gradient_image

def get_surf_neighbors(surfVTK, pointList=[]):
    
    if not pointList:
        pointList = range(0,surfVTK.GetNumberOfPoints())
    
    neighbors = []
    hanging_verts = []
    isolated_verts = []
    for pointId in pointList:
        cellIds = vtk.vtkIdList()
        surfVTK.GetPointCells(pointId, cellIds)

        thisNeighborhood = []
        for i in range(0, cellIds.GetNumberOfIds()):
                cellPointIds = vtk.vtkIdList()
                surfVTK.GetCellPoints(cellIds.GetId(i), cellPointIds)

                # get the neighbors of the cell
                for j in range(0,cellPointIds.GetNumberOfIds()):
                     thisNeighborhood = thisNeighborhood + [cellPointIds.GetId(j)]                                                                             
        
        thisNeighborhood = list(set(thisNeighborhood))
        thisNeighborhood = [int(x) for x in thisNeighborhood if x != pointId]
        neighbors = neighbors + [thisNeighborhood]
        
        if len(thisNeighborhood) == 1 or len(thisNeighborhood) == 2:
            hanging_verts = hanging_verts + [pointId]
        
        if len(thisNeighborhood) == 0:
            isolated_verts = isolated_verts + [pointId]

    return neighbors, hanging_verts, isolated_verts

def vertex_proximity(point_array, k, locator):
    thisPnt = point_array[k,:].tolist()    
    pntList = vtk.vtkIdList()

    closestPnt = thisPnt
    n_closest = 1
    while closestPnt == thisPnt:
        locator.FindClosestNPoints(n_closest, thisPnt, pntList)
        closestPntId = int(pntList.GetId(pntList.GetNumberOfIds()-1))
        closestPnt = point_array[closestPntId,:].tolist()
        n_closest += 1
    
    thisPnt = np.array(thisPnt)
    closestPnt = np.array(closestPnt)

    dmin_to_other_vert = scipy.spatial.distance.cdist(thisPnt[:,None].T, closestPnt[:,None].T)
    
    return  dmin_to_other_vert

def cell_intersection(k, point_array, dir_array, cell_locator):
    tol0 = 0.00001
    thisPnt = point_array[k,:] + dir_array[k,:]/np.linalg.norm(dir_array[k,:], axis=0)*tol0
    #thisPnt = point_array[k,:]

    movedPnt = point_array[k,:] + dir_array[k,:] + dir_array[k,:]/np.linalg.norm(dir_array[k,:], axis=0)*0.01
    #movedPnt = point_array[k,:] + dir_array[k,:]

    t = vtk.mutable(0.0)
    pos = [0.0, 0.0, 0.0]
    pcoords = [0.0, 0.0, 0.0]
    subId = vtk.mutable(0)
    cell_locator.IntersectWithLine(thisPnt.tolist(), movedPnt.tolist(), tol0, t, pos, pcoords, subId)
    
    if t <= tol0 or t > 1.0 - tol0:
        no_intersection = 1 # no intersection, safe to moce
    else:
        #no_intersection = 0 # intersection detected
        for no_intersection in np.arange(1, 0, -0.05):
            #movedPnt = point_array[k,:] + dir_array[k,:]*no_intersection
            movedPnt = point_array[k,:] + dir_array[k,:]*no_intersection + dir_array[k,:]/np.linalg.norm(dir_array[k,:], axis=0)*0.01
            cell_locator.IntersectWithLine(thisPnt.tolist(), movedPnt.tolist(), tol0, t, pos, pcoords, subId)
            #print(t, no_intersection)
            # If no intersection then use this move instead
            no_intersection = no_intersection / 2.1 # Worst case scenario two vertices move towards each other at this distance, divide by 2 to avoid intersection
            if t <= tol0 or t > 1.0 - tol0:
                #print("No intersection found at: " +  str(no_intersection))
                break

    return no_intersection
# Given Two VTK surfaces Get a medial surface
def get_medial_surface(surfVTK1, surfVTK2):
    medialSurf = vtk.vtkPolyData()
    medialSurf.DeepCopy(surfVTK1)

    pntLocator = vtk.vtkCellLocator()
    pntLocator.SetDataSet(surfVTK2)
    pntLocator.BuildLocator()

    normalsAlg  = vtk.vtkPolyDataNormals()
    normalsAlg.ComputePointNormalsOn()
    normalsAlg.SetInputData(surfVTK1)
    normalsAlg.SetAutoOrientNormals(False)
    normalsAlg.NonManifoldTraversalOff()
    normalsAlg.ConsistencyOn()
    normalsAlg.SplittingOff()
    normalsAlg.Update()
    surfVTK_norm = normalsAlg.GetOutput()
    normal_array = vtk_to_numpy(surfVTK_norm.GetPointData().GetArray("Normals"))


    surf1_points = vtk_to_numpy(surfVTK1.GetPoints().GetData())
    surf2_points = vtk_to_numpy(surfVTK2.GetPoints().GetData())
    medial_points = np.zeros(surf1_points.shape)
    for i in range(0,surf1_points.shape[0]):
        thisPnt = surf1_points[i,:]
        #closestPnt = surf2_points[pntLocator.FindClosestPoint(thisPnt),:]
            
        movedPnt = thisPnt + (normal_array[i,:]*10)
        tol0 = 0
        tol1 = vtk.mutable(0)
        pos = [0.0, 0.0, 0.0]
        pcoords = [0.0, 0.0, 0.0]
        subId = vtk.mutable(0)
        pntLocator.IntersectWithLine(thisPnt.tolist(), movedPnt.tolist(), tol0, tol1, pos, pcoords, subId)
        closestPnt = pos

        if closestPnt ==[0,0,0]:
            cellId = vtk.mutable(0)
            closestPointDist2 = vtk.mutable(0)
            pntLocator.FindClosestPoint(thisPnt.tolist(),closestPnt, cellId, subId, closestPointDist2)

        if closestPnt == [0,0,0]:
            medial_points[i,:] = thisPnt
        elif np.linalg.norm((thisPnt-closestPnt)/2,axis=0) > 10:
            medial_points[i,:] = thisPnt
        else:
            medial_points[i,:] = thisPnt - (thisPnt - closestPnt)/2
    
    medial_vtkArray = numpy_to_vtk(medial_points)
    medial_vtkPoints = vtk.vtkPoints()
    medial_vtkPoints.SetData(medial_vtkArray)
    medialSurf.SetPoints(medial_vtkPoints)

    return medialSurf
    
def get_radial_surface(medialVTK, normalSurfVTK, vecImgVTK, sformMat):
    
    normalsAlg  = vtk.vtkPolyDataNormals()
    normalsAlg.ComputeCellNormalsOff()
    normalsAlg.ComputePointNormalsOn()
    normalsAlg.SetInputData(normalSurfVTK)
    normalsAlg.SetAutoOrientNormals(False)
    normalsAlg.NonManifoldTraversalOn()
    normalsAlg.ConsistencyOn()
    normalsAlg.SplittingOff()
    normalsAlg.Update()
    surfVTK_norm = normalsAlg.GetOutput()
    normal_array = vtk_to_numpy(surfVTK_norm.GetPointData().GetArray("Normals"))
    
    vecVTK = interpolate_voldata_to_surface(medialVTK, vecImgVTK, sformMat, pntDataName='EVECS')
    prim_evecs = vtk_to_numpy(vecVTK.GetPointData().GetArray("EVECS"))
    
    rad_points = np.absolute(np.einsum('ij,ij->i', prim_evecs, normal_array))
   
    rad_surf = vtk.vtkPolyData()
    rad_surf.DeepCopy(medialVTK)
    rad_vtkArray = numpy_to_vtk(rad_points[:,None])
    rad_vtkArray.SetName("Radiality")
    #rad_vtkPoints = vtk.vtkPoints()
    #rad_vtkPoints.SetData(rad_vtkArray)
    rad_surf.GetPointData().AddArray(rad_vtkArray)


    return rad_surf


def correct_surface_by_normal(surfVTK, targetImageVTK, gradImageVTK, sformMat, target_val, interMaskVTK, 
                            reverse_direction=False, n_iter=500, smooth_iter=15, move_dmax=0.06, output_itr_surface=True, 
                            cell_distmin=0.15, pnt_dmin = 0.1, limitImg=[], limit=[], lambda_I=1, target_nverts = 400000):
    
    global globalSurf
    global globalCellLocator
    

    #sum_bypoint = cp.ElementwiseKernel(
    #    'I thisPnt, I startID, I endID, raw F term2_bypnt',
    #    'F pntSum',
    #    '''
    #    pntSum = 0.0;
    #    for(int i=startID; i <= endID; i++)
    #    {
    #        pntSum += term2_bypnt[i];
    #    }
    #    ''',
    #    'sum_bypoint')

    #mean_bypoint = cp.ElementwiseKernel(
    #    'I thisPnt, I startID, I endID, raw F term2_bypnt',
    #    'F pntMean',
    #    '''
    #    float pntSum = 0.0;
    #    for(int i=startID; i <= endID; i++)
    #    {
    #        pntSum += term2_bypnt[i];
    #    }
    #    pntMean = pntSum / (endID - startID + 1)
    #    ''',
    #    'mean_bypoint')

    if isinstance(surfVTK, list):
        print("Multisurfing!")
        multisurf = True
    else:
        multisurf = False

    print("Correcting Surface by Searching Normals on MeanDWI - Target Value: " + str(target_val))
  
    if multisurf:
        multiSurfVTK1 = vtk.vtkPolyData()
        multiSurfVTK1.DeepCopy(surfVTK[0])

        multiSurfVTK2 = vtk.vtkPolyData()
        multiSurfVTK2.DeepCopy(surfVTK[1])

        multiSurfVTK = [multiSurfVTK1, multiSurfVTK2]
        surfNum = 2
    else:
        thisSurfVTK = vtk.vtkPolyData()
        thisSurfVTK.DeepCopy(surfVTK)
        surfNum = 1
    
    
    
    lambda_I = 0.0
    lambda_N = 0.1 # Restrict movement when curved
    lambda_S = 0.01 # smoothing

    for i in range(0, n_iter):
        for m in range(0,surfNum):
            if multisurf:
                if m == 0:
                    thisSurfVTK = multiSurfVTK[0]
                    otherSurfVTK = multiSurfVTK[1]
                    #surf_intensity = multisurf_intensity[0]
                    #surf_grad = multisurf_grad[0]
                else:
                    thisSurfVTK = multiSurfVTK[1]
                    otherSurfVTK = multiSurfVTK[0]
                    #surf_intensity = multisurf_intensity[1]
                    #surf_grad = multisurf_grad[1]
            
            curver = vtk.vtkCurvatures()
            curver.SetInputData(thisSurfVTK)
            curver.SetCurvatureType(3) # Minimum 3, Maximum 2, Mean 1, Gauss 0
            curver.Update()
            curvSurf = curver.GetOutput()
            curv_min = vtk_to_numpy(curvSurf.GetPointData().GetArray("Minimum_Curvature"))
            curver.SetCurvatureType(2) # Minimum 3, Maximum 2, Mean 1, Gauss 0
            curver.Update()
            curvSurf = curver.GetOutput()
            curv_max = vtk_to_numpy(curvSurf.GetPointData().GetArray("Maximum_Curvature"))
            curvVals = (0.25*curv_max + curv_min)/2
            print("Min Max Avg Curv", np.max(curvVals), np.min(curvVals), np.mean(curvVals))
            del curver
            #curv_stop = np.ones(curvVals.shape)
            #plt.hist(curvVals)
            #plt.show()
            
            #curvVals = curvVals / np.max(curvVals)
            #curvVals[curvVals > 1.0 ] = 1.0
            #curvVals[curvVals < -1.0] = -1.0
            #plt.hist(curvVals)
            #plt.show()

            #divider = vtk.vtkAdaptiveSubdivisionFilter()
            #divider.SetInputData(thisSurfVTK)
            #divider.SetMaximumEdgeLength(maxEdge)
            #divider.Update()
            #thisSurfVTK = divider.GetOutput()

            num_points = np.size(vtk_to_numpy(thisSurfVTK.GetPoints().GetData()), axis=0)
            target_intensity = np.ones((num_points,)) * target_val

            output_stream.write("Iteration: " + str(i) + "    \r")
            output_stream.flush()
            sys.stdout.flush()
                                    
            #pntLocator = vtk.vtkPointLocator()
            #pntLocator.SetDataSet(thisSurfVTK)
            #pntLocator.BuildLocator()
            
	    #cell2Pnts = vtk.vtkCellCenters()
	    #cell2Pnts.SetInputData(thisSurfVTK)
	    #cell2Pnts.VertexCellsOff()
	    #cell2Pnts.Update()
	    #cellSurf = cell2Pnts.GetOutput()
            
            if multisurf:
                appender = vtk.vtkAppendPolyData()
                appender.AddInputData(thisSurfVTK)
                appender.AddInputData(otherSurfVTK)
                appender.Update()
                intersectSurf = appender.GetOutput()
            else:
                intersectSurf = thisSurfVTK
            
            cellLocator = vtk.vtkCellLocator()
            cellLocator.SetDataSet(intersectSurf)
            cellLocator.SetNumberOfCellsPerBucket(1)
            cellLocator.SetTolerance(0.00000001)
            cellLocator.BuildLocator()

            # Calculating Point Normals
            normalsAlg  = vtk.vtkPolyDataNormals()
            normalsAlg.ComputeCellNormalsOff()
            normalsAlg.ComputePointNormalsOn()
            normalsAlg.SetInputData(thisSurfVTK)
            normalsAlg.SetAutoOrientNormals(True)
            normalsAlg.NonManifoldTraversalOn()
            normalsAlg.ConsistencyOn()
            normalsAlg.SplittingOff()
            normalsAlg.Update()
            surfVTK_norm = normalsAlg.GetOutput()

            #surfVTK_interp = interpolate_voldata_to_surface(thisSurfVTK, meanDiffVTK, sformMat)
            surfVTK_interp =  interpolate_voldata_to_surface(thisSurfVTK, targetImageVTK, sformMat)
            intensity_array = vtk_to_numpy(surfVTK_interp.GetPointData().GetArray("NIFTI"))

            # Interpolate gradient onto surface
            surfVTK_grad = interpolate_voldata_to_surface(thisSurfVTK, gradImageVTK, sformMat)
            gradient_array = vtk_to_numpy(surfVTK_grad.GetPointData().GetArray("NIFTI"))
            xscale = sformMat.GetMatrix().GetElement(0,0)
            yscale = sformMat.GetMatrix().GetElement(1,1)
            zscale = sformMat.GetMatrix().GetElement(2,2)

            xscale = xscale / np.absolute(xscale)
            yscale = yscale / np.absolute(yscale)
            zscale = zscale / np.absolute(zscale)

            gradient_array[:,0] = xscale * gradient_array[:,0]             
            gradient_array[:,1] = yscale * gradient_array[:,1]
            #gradient_array[:,2] = zscale * gradient_array[:,2]
             
            interMaskSurf = interpolate_voldata_to_surface(thisSurfVTK, interMaskVTK, sformMat)
            surfMask = vtk_to_numpy(interMaskSurf.GetPointData().GetArray("NIFTI"))

            normal_array = vtk_to_numpy(surfVTK_norm.GetPointData().GetArray("Normals"))
            point_array = vtk_to_numpy(surfVTK_norm.GetPoints().GetData()) 
            #cell_center_array = vtk_to_numpy(cellSurf.GetPoints().GetData())

            move_limits = np.ones((point_array.shape[0],))
            if limitImg:
                limitSurf = interpolate_voldata_to_surface(thisSurfVTK, limitImg, sformMat)
                limit_vals = vtk_to_numpy(limitSurf.GetPointData().GetArray("NIFTI"))
                move_limits[limit_vals < limit] = 0

            #show_normals(point_array, normal_array)
            e0_array = np.concatenate((np.expand_dims(-normal_array[:,2],axis=1), np.zeros((normal_array.shape[0],1)), np.expand_dims(normal_array[:,0],axis=1)), axis=1)
            e1_array = np.cross(normal_array,e0_array)

            # Part 1 of objective function
            
            # Based on this variable decide which way to move vertice along normal
            #intensity_scale = np.mean(target_intensity) * 110
            intensity_scale = 1
            if reverse_direction:
                intensity_diff = (intensity_array - target_intensity)[:,None] / intensity_scale # Use this one for FA (If using B0 gradient)
            else:
                intensity_diff = (target_intensity - intensity_array)[:,None] / intensity_scale # Use this one for MD (If using B0 gradient)
            
            direction_array = normal_array # For ISMRM 2019 moved vertices in direction of normal
            #direction_array = gradient_array / np.linalg.norm(gradient_array,axis=1)[:,None] # Normalize the gradient direction because of hyper intensities on B0 image
            #direction_array = gradient_array
            
            term1 = lambda_I * direction_array * intensity_diff 
            move_amp_array = intensity_diff / np.absolute(intensity_diff)
            
            # Part 2 of Term 
            #point_prox = np.zeros((point_array.shape[0],))
            #cell_prox = np.zeros((point_array.shape[0],))
            #self_prox = np.zeros((point_array.shape[0],))
            #self_dmin = np.zeros((point_array.shape[0],))
            cell_dmin = np.ones((point_array.shape[0],))
            #dir_arry = np.ones((point_array.shape[0],))*0.1
            #old_cell_dmin = cell_dmin
            #cell_intersect = np.zeros((point_array.shape[0],))

            #term2 = cp.zeros((point_array.shape[0],3))
            #term2 = np.zeros((point_array.shape[0],))
            #term2_p1 = cp.zeros((point_array.shape[0],3))
            #term2_e0 = cp.zeros((point_array.shape[0],3))
            #term2_e1 = cp.zeros((point_array.shape[0],3))
            #term2_p2 = cp.zeros((point_array.shape[0],3))
           
            #gpu_point_array = cp.array(point_array[pointID_flat_ind,:])
            #gpu_neighbor_array = cp.array(point_array[neighborID_flat_ind,:])
            #gpu_normal_array = cp.array(normal_array[pointID_flat_ind,:])
            #gpu_e0_array = cp.array(e0_array[pointID_flat_ind,:])
            #gpu_e1_array = cp.array(e1_array[pointID_flat_ind,:])

            #normal_dot = cp.einsum('ij,ij->i', 
            #    gpu_normal_array, 
            #    gpu_point_array - gpu_neighbor_array)

            #e0_dot = cp.einsum('ij,ij->i',
            #    gpu_e0_array, 
            #    gpu_point_array - gpu_neighbor_array)

            #e1_dot = cp.einsum('ij,ij->i',
            #    gpu_e1_array,                          
            #    gpu_point_array - gpu_neighbor_array)
           
            #term2_pntwise =   lambda_N * -gpu_normal_array * cp.repeat(cp.einsum('ij,ij->i', gpu_normal_array, gpu_point_array - gpu_neighbor_array)[:,None],3,axis=1)\
            #                + lambda_E * -gpu_e0_array * cp.repeat(cp.einsum('ij,ij->i', gpu_e0_array, gpu_point_array - gpu_neighbor_array)[:,None],3,axis=1) \
            #                + lambda_E * -gpu_e1_array * cp.repeat(cp.einsum('ij,ij->i', gpu_e1_array, gpu_point_array - gpu_neighbor_array)[:,None],3,axis=1)
            
            #term2_p1_pntwise = lambda_N * -gpu_normal_array * cp.repeat(normal_dot[:,None],3,axis=1)
            #term2_e0_pntwise = gpu_e0_array * cp.repeat(e0_dot[:,None],3,axis=1)
            #term2_e1_pntwise = gpu_e1_array * cp.repeat(e1_dot[:,None],3,axis=1)
            #term2_p2_pntwise = gpu_e0_array * cp.repeat(e0_dot[:,None],3,axis=1)  + gpu_e1_array * cp.repeat(e1_dot[:,None],3,axis=1) 
            
            #term2_pntwise = cp.asnumpy(term2_pntwise)
            #term2_pntwise = cp.asnumpy(term2_p1_pntwise)
            #term2 = np.array([np.sum(term2_pntwise[pointID_flat_ind == k,:], axis=0) for k in range(0,point_array.shape[0])])
            
            #sum_bypoint(pointIDs, start_pointIDs, end_pointIDs, term2_pntwise[:,0].astype(np.float64),gpu_term2[:,0])
            #sum_bypoint(pointIDs, start_pointIDs, end_pointIDs, term2_pntwise[:,1].astype(np.float64),gpu_term2[:,1])
            #sum_bypoint(pointIDs, start_pointIDs, end_pointIDs, term2_pntwise[:,2].astype(np.float64),gpu_term2[:,2])
            
            #sum_bypoint(pointIDs, start_pointIDs, end_pointIDs, term2_pntwise[:,0].astype(np.float64),gpu_term2[:,0])
            #sum_bypoint(pointIDs, start_pointIDs, end_pointIDs, term2_pntwise[:,1].astype(np.float64),gpu_term2[:,1])
            #sum_bypoint(pointIDs, start_pointIDs, end_pointIDs, term2_pntwise[:,2].astype(np.float64),gpu_term2[:,2])
            #term2 = cp.asnumpy(gpu_term2)

            #mean_bypoint(pointIDs, start_pointIDs, end_pointIDs, (gpu_neighbor_array[:,0] - gpu_point_array[:,0]).astype(np.float64), gpu_term2_smooth[:,0])
            #mean_bypoint(pointIDs, start_pointIDs, end_pointIDs, (gpu_neighbor_array[:,1] - gpu_point_array[:,1]).astype(np.float64), gpu_term2_smooth[:,1])
            #mean_bypoint(pointIDs, start_pointIDs, end_pointIDs, (gpu_neighbor_array[:,2] - gpu_point_array[:,2]).astype(np.float64), gpu_term2_smooth[:,2])
           
            #taubin_l = 0.5 # Shrinking
            #taubin_m = -0.1 # Expanding
            #taubin_l = 1.0 # when these are set to one becomes one iteration of laplacian smoothing
            #taubin_m = 1.0
            #if i%2 == 0:
            #    term2_smooth = cp.asnumpy(taubin_m*gpu_term2_smooth)
            #else:
            #    term2_smooth = cp.asnumpy(taubin_l*gpu_term2_smooth)

            dir_array = surfMask[:,None] * (term1 + lambda_S * -normal_array * curvVals[:,None])
            #dir_array = surfMask[:,None] * term1

            #dir_array = dir_array / np.linalg.norm(dir_array,axis=1)[:,None]

            #print "Calculating Intersection Criteria Per Vertex"
           # for k in range(0, point_array.shape[0]):
           #     if k%10000 == 0:
           #         output_stream.write("Calculating Part 2 of objective: Vert " + str(k) + "    \r")
           #         output_stream.flush()
           #     
           #     sys.stdout.flush()
           #    neighbor_ind = neighbors[k]
            #    neighbor_points = np.array(point_array[neighbor_ind,:])
            #    
            #    if len(neighbor_ind) < 3:
            #        print "Short on neighbors ", len(neighbor_ind)

            #    k_points = np.repeat(np.expand_dims(np.array(point_array[k,:]),axis=1),len(neighbor_ind),axis=1).T
            #    k_normal = np.repeat(np.expand_dims(np.array(normal_array[k,:]),axis=1),len(neighbor_ind),axis=1).T

            #    k_e0 = np.repeat(np.expand_dims(np.array(e0_array[k,:]),axis=1),len(neighbor_ind),axis=1).T
            #    k_e1 = np.repeat(np.expand_dims(np.array(e1_array[k,:]),axis=1),len(neighbor_ind),axis=1).T


            #    term2_p1[k,:] = -np.array(normal_array[k,:]) * np.sum(np.einsum('ij,ij->i', k_normal, k_points-neighbor_points))
            #    term2_e0[k,:] = -np.array(e0_array[k,:]) * np.sum(np.einsum('ij,ij->i', k_e0, k_points-neighbor_points))
            #    term2_e1[k,:] = -np.array(e1_array[k,:]) * np.sum(np.einsum('ij,ij->i', k_e1, k_points-neighbor_points))

            #    term2[k,:] = lambda_N * term2_p1[k] + term2_e0[k] + term2_e1[k]
            #    
                # Test term2 by simply moving vertice to centroid
                #term2[k,:] = np.mean(neighbor_points,axis=0) - point_array[k,:]
            
                #term2[k,:] = [1.0, 1.0, 1.0]
                #if k == 1906:
                #if np.isnan(term2[k,:]).any():
                #    print "Vertice ", k
                #    print "Point ", point_array[k,:]
                #    print term2[k,:]
                #    print np.linalg.norm(term2[k,:],axis=0)
                #    print term2[k,:].dtype
                #    print "Normal Array"
                #    print -normal_array[k,:]
                #    print "E0 Array"
                #    print -e0_array[k,:]
                #    print "E1 Array"
                #    print -e1_array[k,:]
                #    print "Neighbor Points"
                #    print neighbor_points
                #    print "K Points"
                #    print k_points.dtype
                #    print "Centroid Difference"
                #    print np.mean(neighbor_points,axis=0) - point_array[k,:]

                # self proximity terms that should prevent self-intersection
                #self_dmin[k] =  vertex_proximity(point_array, k, pntLocator)
                #glopbalSurf = thisSurfVTK
                #glopbalCellLocator = cellLocator
                #poopl = Pool()
                #cell_dmin = pool.map(partial(cell_intersection_parallel, 
                #                               point_array=point_array,
                #                               normal_array= normal_array,
                #                               move_max = move_dmax,
                #                               dir_array = dir_array), range(0,num_points))
                
                ## Below is the only line used for ISMRM Abstract For Self Intersection (Also used curvature)
                #k_move_vert = surfMask[k,None] * lambda_S*(term1[k,:] + term2_p1[k,:]) # Test for intersection for the components along the expansion direction
             #   cell_dmin[k] = cell_intersection(k, point_array, dir_array, cellLocator)
                 
                # Cell Proximity term (minimum of all neighboring cells)
                #cellIds = vtk.vtkIdList()
                #thisSurfVTK.GetPointCells(k, cellIds) # Gets the cells using point k
                #cell_prox_list = [vertex_proximity(cellSurf, cell_dmin, cellId, cellLocator) for cellId in range(0, cellIds.GetNumberOfIds())]	
                #cell_prox[k] = np.min(cell_prox_list)
                #cell_prox[k] = 1
                #self_prox[k] = point_prox[k] * cell_prox[k] # Note if either one of these is close to zero the pnt vertice won't move
            
            #term2 = cp.asnumpy(term2)
            #if np.isnan(term2).any():
            #    print "Detected NaN!!!!!!"

            #cell_dmin[old_cell_dmin == 0] = 0

            #point_prox[:] = scipy.special.expit(1000*(self_dmin - pnt_dmin))
            #cell_prox[:] = scipy.special.expit(1000*(cell_dmin - cell_distmin))
            #self_prox = point_prox

            #output_stream.write('\n')
            #move_vert = surfMask[:,None] * self_prox[:,None] * (term1 - term2)
            #self_prox[self_prox < 0.0001] = 0
            #move_vert = curv_stop[:,None] * move_limits[:,None] * surfMask[:,None] * cell_dmin[:,None] * (term1) # ISMRM Abstract Objective
            
            move_vert = surfMask[:,None] * cell_dmin[:,None] * (term1 + lambda_S * -normal_array * curvVals[:,None])
            #move_vert = surfMask[:,None] * (cell_dmin[:,None] * term1 + lambda_S * term2)
            
            #move_vector = np.linalg.norm(move_vert,axis=1)
            #print move_vector[move_vector>3]
            #print np.array(range(0,len(move_vector)))[move_vector>3]
            #move_vert = surfMask[:,None] * lambda_S * (cell_dmin[:,None] * term1 + np.repeat(term2[:,None],3,axis=1))

            # Don't move vertices more than move_dmax per iteration
            #move_vert[np.linalg.norm(move_vert,axis=1) > move_dmax,:] = move_vert[np.linalg.norm(move_vert,axis=1) > move_dmax,:] / np.linalg.norm(move_vert,axis=1)[np.linalg.norm(move_vert,axis=1)> move_dmax,None] * move_dmax
    
           # sample_pnt = 1906
           # print "Moving Vertice By", move_vert[sample_pnt,:]
           # print "Movement Magnitude", np.linalg.norm(move_vert[sample_pnt,:])
           # #print "Term 1", term1[sample_pnt,:]
           # print "Direction", direction_array[sample_pnt,:]
	   # print "Intensity"
           # print intensity_array[sample_pnt]
	    #print("Target Intenstity: " +  str(target_intensity))
            #print "Cell Dmin: " + str(cell_dmin[sample_pnt])
            #print "Surf Mask: " + str(surfMask[sample_pnt])
            #print "Move Limit: " + str(move_limits[sample_pnt])
            #if limitImg:
            #    print "Limit Values: " + str(limit_vals[sample_pnt])
            oldSurfVTK = vtk.vtkPolyData()
            oldSurfVTK.DeepCopy(thisSurfVTK)

            thisSurfVTK = translate_surface(thisSurfVTK, move_vert)
            
            if smooth_iter > 0:
                print("Smoothing Iteration")
                points_presmooth = vtk_to_numpy(thisSurfVTK.GetPoints().GetData())
                thisSurfVTK = smooth_mesh(thisSurfVTK, n_iter=smooth_iter, convergence = smooth_converge, relax=smooth_relax)
            
                points_postsmooth = vtk_to_numpy(thisSurfVTK.GetPoints().GetData())
            
                # Don't move masked points
                #points_postsmooth[surfMask == 0,:] = points_presmooth[surfMask == 0,:] # Do not smooth white matter surface
            
                # Restrict smoothing based on self proximity
                #points_postsmooth[:,:] = points_presmooth + (points_postsmooth-points_presmooth)*self_prox[:,None] # Restrict smoothing based on self proximity
            
                # Don't move points if they are going to intersect the surface
                points_postsmooth[:,:] = points_presmooth + (points_postsmooth-points_presmooth)*cell_dmin[:,None] # Restrict smoothing based on self proximity
            else:
                points_postsmooth = vtk_to_numpy(thisSurfVTK.GetPoints().GetData())

            new_vtkArray = numpy_to_vtk(points_postsmooth)
            new_vtkPoints = vtk.vtkPoints()
            new_vtkPoints.SetData(new_vtkArray)
            thisSurfVTK.SetPoints(new_vtkPoints)

            #vtk_move_vert = numpy_to_vtk(move_vert)
            #vtk_move_vert.SetName("Move Vert")

            #vtk_point_dmin = numpy_to_vtk(self_dmin)
            #vtk_point_dmin.SetName("Point Dmin")
            #            
            #vtk_point_prox = numpy_to_vtk(point_prox)
            #vtk_point_prox.SetName("Point Proxy")

            vtk_cell_dmin = numpy_to_vtk(cell_dmin)
            vtk_cell_dmin.SetName("Cell Dmin")
            #            
            #vtk_cell_prox = numpy_to_vtk(cell_prox)
            #vtk_cell_prox.SetName("Cell Proxy")
            #        
            vtk_surfMask = numpy_to_vtk(surfMask)
            vtk_surfMask.SetName("Surf Mask")

            #vtk_target = numpy_to_vtk(target_intensity)
            #vtk_target.SetName("Target")
            #
            vtk_curv = numpy_to_vtk(curvVals)
            vtk_curv.SetName("Curv Vals")
           
            #vtk_norm_xdir = numpy_to_vtk(normal_array[:,0])
            #vtk_norm_xdir.SetName("Normal_X_Dir")
            #vtk_norm_ydir = numpy_to_vtk(normal_array[:,1])
            #vtk_norm_ydir.SetName("Normal_Y_Dir")
            #vtk_norm_zdir = numpy_to_vtk(normal_array[:,2])
            #vtk_norm_zdir.SetName("Normal Z_Dir")

            #vtk_xdir = numpy_to_vtk(direction_array[:,0])
            #vtk_xdir.SetName("X_Dir")
            #vtk_ydir = numpy_to_vtk(direction_array[:,1])
            #vtk_ydir.SetName("Y_Dir")
            #vtk_zdir = numpy_to_vtk(direction_array[:,2])
            #vtk_zdir.SetName("Z_Dir")
            #
            #vtk_term1_xdir = numpy_to_vtk(term1[:,0])
            #vtk_term1_xdir.SetName("Term1 X")
            #vtk_term1_ydir = numpy_to_vtk(term1[:,1])
            #vtk_term1_ydir.SetName("Term1 Y")
            #vtk_term1_zdir = numpy_to_vtk(term1[:,2])
            #vtk_term1_zdir.SetName("Term1 Z")
            #
            #
            #vtk_term2_p1_xdir = numpy_to_vtk(term2[:,0])
            #vtk_term2_p1_xdir.SetName("Term2 P1 X")
            #vtk_term2_p1_ydir = numpy_to_vtk(term2[:,1])
            #vtk_term2_p1_ydir.SetName("Term2 P1 Y")
            #vtk_term2_p1_zdir = numpy_to_vtk(term2[:,2])
            #vtk_term2_p1_zdir.SetName("Term2 P1 Z")

            #vtk_term2_e0_xdir = numpy_to_vtk(term2_e0[:,0])
            #vtk_term2_e0_xdir.SetName("Term2 E0 X")
            #vtk_term2_e0_ydir = numpy_to_vtk(term2_e0[:,1])
            #vtk_term2_e0_ydir.SetName("Term2 E0 Y")
            #vtk_term2_e0_zdir = numpy_to_vtk(term2_e0[:,2])
            #vtk_term2_e0_zdir.SetName("Term2 E0 Z")

            #vtk_term2_p2_xdir = numpy_to_vtk(term2_p2[:,0])
            #vtk_term2_p2_xdir.SetName("Term2 P2 X")
            #vtk_term2_p2_ydir = numpy_to_vtk(term2_p2[:,1])
            #vtk_term2_p2_ydir.SetName("Term2 P2 Y")
            #vtk_term2_p2_zdir = numpy_to_vtk(term2_p2[:,2])
            #vtk_term2_p2_zdir.SetName("Term2 P2 Z")

            #vtk_term2_e1_xdir = numpy_to_vtk(term2_e1[:,0])
            #vtk_term2_e1_xdir.SetName("Term2 E1 X")
            #vtk_term2_e1_ydir = numpy_to_vtk(term2_e1[:,1])
            #vtk_term2_e1_ydir.SetName("Term2 E1 Y")
            #vtk_term2_e1_zdir = numpy_to_vtk(term2_e1[:,2])
            #vtk_term2_e1_zdir.SetName("Term2 E1 Z")
            #
            #vtk_term2_p1_xdir = numpy_to_vtk(term2_p1[:,0])
            #vtk_term2_p1_xdir.SetName("Term2 P1 X")
            #vtk_term2_p1_ydir = numpy_to_vtk(term2_p1[:,1])
            #vtk_term2_p1_ydir.SetName("Term2 P1 Y")
            #vtk_term2_p1_zdir = numpy_to_vtk(term2_p1[:,2])
            #vtk_term2_p1_zdir.SetName("Term2 P1 Z")

            ##vtk_term2_p1 = numpy_to_vtk(term2_p1)
            ##vtk_term2_p1.SetName("term2_p1")
            ##vtk_term2_p2 = numpy_to_vtk(term2_p2)
            ##vtk_term2_p2.SetName("term2_p2")

            #vtk_term2 = numpy_to_vtk(term2)
            #vtk_term2.SetName("term2")

            ##thisSurfVTK.GetPointData().AddArray(vtk_move_vert)
            ##thisSurfVTK.GetPointData().AddArray(vtk_term1)
            ##thisSurfVTK.GetPointData().AddArray(vtk_term2)
            ##thisSurfVTK.GetPointData().AddArray(vtk_point_dmin)
            ##thisSurfVTK.GetPointData().AddArray(vtk_point_prox)
            oldSurfVTK.GetPointData().AddArray(vtk_cell_dmin)
            oldSurfVTK.GetPointData().AddArray(vtk_surfMask)
            oldSurfVTK.GetPointData().AddArray(vtk_curv)
            #thisSurfVTK.GetPointData().AddArray(vtk_norm_xdir)
            #thisSurfVTK.GetPointData().AddArray(vtk_norm_ydir)
            #thisSurfVTK.GetPointData().AddArray(vtk_norm_zdir)

            #thisSurfVTK.GetPointData().AddArray(vtk_xdir)
            #thisSurfVTK.GetPointData().AddArray(vtk_ydir)
            #thisSurfVTK.GetPointData().AddArray(vtk_zdir)
            #
            #thisSurfVTK.GetPointData().AddArray(vtk_term1_xdir)
            #thisSurfVTK.GetPointData().AddArray(vtk_term1_ydir)
            #thisSurfVTK.GetPointData().AddArray(vtk_term1_zdir)
            #
            #oldSurfVTK.GetPointData().AddArray(vtk_term2_p1_xdir)
            #oldSurfVTK.GetPointData().AddArray(vtk_term2_p1_ydir)
            #oldSurfVTK.GetPointData().AddArray(vtk_term2_p1_zdir)
            #
            #thisSurfVTK.GetPointData().AddArray(vtk_term2_e0_xdir)
            #thisSurfVTK.GetPointData().AddArray(vtk_term2_e0_ydir)
            #thisSurfVTK.GetPointData().AddArray(vtk_term2_e0_zdir)

            #thisSurfVTK.GetPointData().AddArray(vtk_term2_p2_xdir)
            #thisSurfVTK.GetPointData().AddArray(vtk_term2_p2_ydir)
            #thisSurfVTK.GetPointData().AddArray(vtk_term2_p2_zdir)

            #thisSurfVTK.GetPointData().AddArray(vtk_term2_e1_xdir)
            #thisSurfVTK.GetPointData().AddArray(vtk_term2_e1_ydir)
            #thisSurfVTK.GetPointData().AddArray(vtk_term2_e1_zdir)
            #
            #thisSurfVTK.GetPointData().AddArray(vtk_term2_p1_xdir)
            #thisSurfVTK.GetPointData().AddArray(vtk_term2_p1_ydir)
            #thisSurfVTK.GetPointData().AddArray(vtk_term2_p1_zdir)
            
            output_every_itr = 20
            if output_itr_surface:
                if i% output_every_itr == 0:
                    if multisurf:
                        write_surf_vtk(oldSurfVTK, procDir + "wm_itr" + str(i) + '_surf_' + str(m) + ".vtk")
                    else:
                        write_surf_vtk(oldSurfVTK, procDir + "wm_itr" + str(i)  
                            + '_fa' + str(target_val).replace('.','p')
                            + '_LI' + str(lambda_I).replace('.','p')
                            + '_LN' + str(lambda_N).replace('.','p')
                            + '_LS' + str(lambda_S).replace('.','p') + '.vtk')
   
            if multisurf:
                if m == 0:
                    multiSurfVTK[0] = thisSurfVTK
                else:
                    multiSurfVTK[1] = thisSurfVTK

    if multisurf:
        thisSurfVTK = multiSurfVTK
    output_stream.write('\n')
    return thisSurfVTK

# Visualize surface using mayavi
#def show_surface(surfDict):
#    mlab.triangular_mesh([vert[0] for vert in surfDict['verts']], [vert[1] for vert in surfDict['verts']], [vert[2] for vert in surfDict['verts']], surfDict['faces']) 
#    #if os.fork() == 0:
#    #    try:
#    mlab.show() 
#    #finally:
#    #        os._exit(os.EX_OK)
#
#    return

#def show_normals(point_array, normal_array):
#    mlab.quiver3d(point_array[:,0],
#                  point_array[:,1],
#                  point_array[:,2],
#                  normal_array[:,0],
#                  normal_array[:,1],
#                  normal_array[:,2],
#                  opacity=0.2,
#            line_width=0.2,
#            color=(0, 0, 1))
#    mlab.show()
#    return

# Smoothng functions 
def get_neighbors_ind(faces, vert_ind):

    face_ind = np.logical_or(np.logical_or(vert_ind == faces[:,0], vert_ind == faces[:,1]), vert_ind == faces[:,2])
    nb_ind = np.unique(faces[face_ind,:])
    nb_ind = np.delete(nb_ind, np.array(nb_ind == vert_ind))

    return nb_ind


def update_neighbors(surfDict):
    neighbors = np.empty((len(surfDict['verts']),),dtype=object)
    for i in range(0,len(surfDict['verts'])):
        neighbors[i] = get_neighbors_ind(surfDict['faces'], i).tolist()
    surfDict['neighbors'] = neighbors

    return surfDict

def get_laplace_displacement(vert_ind, verts, neighbors):

    nb_ind = neighbors[vert_ind]
    nb_vert = verts[nb_ind,:]
    displace = np.mean(nb_vert,axis = 0) - verts[vert_ind,:]

    return displace

def smooth_mesh(surfVTK, n_iter=100, convergence=0, relax = 0.01):
    ##clean data (!!!This breaks the triangularization of the data)
    #cleaner = vtk.vtkCleanPolyData()
    #cleaner.SetInputData(surfVTK)
    ##cleaner.ConvertLinesToPointsOn()
    ##cleaner.ConvertPolysToLinesOff()
    ##cleaner.ConvertStripsToPolysOff()
    #cleaner.Update()
    #print cleaner
    #print "Cleaned Data"
    #print cleaner.GetOutput()

    #write_surf(cleaner.GetOutput(), "clean_surf.vtp")

    smoother = vtk.vtkSmoothPolyDataFilter()
    #smoother = vtk.vtkWindowedSincPolyDataFilter()
    smoother.SetInputData(surfVTK)
    if convergence > 0:
        smoother.SetConvergence(convergence)
        smoother.SetNumberOfIterations(10000)
    else:
        smoother.SetNumberOfIterations(n_iter)
   # smoother.NormalizeCoordinatesOn()
    smoother.SetRelaxationFactor(relax)
    #smoother.FeatureEdgeSmoothingOn()
    #smoother.SetPassBand(0.2)
    smoother.Update()
    #print smoother

    #print smoother.GetOutput()
    #cleaner = vtk.vtkCleanPolyData()
    #cleaner.SetInputData(smoother.GetOutput())
    #cleaner.ConvertLinesToPointsOff()
    #cleaner.ConvertPolysToLinesOff()
    #cleaner.ConvertStripsToPolysOff()
    #cleaner.PieceInvariantOff()
    #cleaner.Update()
    #print cleaner
    #clean_surf = cleaner.GetOutput()
    #print clean_surf

    return smoother.GetOutput()


# calculate the norms at each vertex given a mesh
def calculate_normals(verts, faces):

    # Normalize an array of vectors (i.e. magnitude of 1)
    def normalize_v3(arr):
        ''' Normalize a numpy array of 3 component vectors shape=(n,3) '''
        lens = np.sqrt( arr[:,0]**2 + arr[:,1]**2 + arr[:,2]**2 )
        arr[:,0] /= lens
        arr[:,1] /= lens
        arr[:,2] /= lens                

        return arr

    #Create a zeroed array with the same type and shape as our vertices i.e., per vertex normal
    norms = np.zeros( verts.shape, dtype=verts.dtype )
    # Create an indexed view into the vertex array using the array of three indices for triangles
    tris = verts[faces]
    #Calculate the normal for all the triangles, by taking the cross product of the vectors v1-v0, and v2-v0 in each triangle             
    n = np.cross( tris[::,1 ] - tris[::,0]  , tris[::,2 ] - tris[::,0] )
    # n is now an array of normals per triangle. The length of each normal is dependent the vertices, 
    # we need to normalize these, so that our next step weights each normal equally.
    normalize_v3(n)
    # now we have a normalized array of normals, one per triangle, i.e., per triangle normals.
    # But instead of one per triangle (i.e., flat shading), we add to each vertex in that triangle, 
    # the triangles' normal. Multiple triangles would then contribute to every vertex, so we need to normalize again afterwards.
    # The cool part, we can actually add the normals through an indexed view of our (zeroed) per vertex normal array
    norms[ faces[:,0] ] += n
    norms[ faces[:,1] ] += n
    norms[ faces[:,2] ] += n
    norms = normalize_v3(norms)

    return norms

def trilinear_interpolation():
    print("Need to make this function")
    return


# Display Bad Edges
def display_bad_edges(surfITK):

    featureEdges = vtk.vtkFeatureEdges()
    featureEdges.SetInputData(surfITK)
    featureEdges.BoundaryEdgesOn()
    featureEdges.SetFeatureAngle(90)
    featureEdges.FeatureEdgesOn()
    featureEdges.BoundaryEdgesOff()
    featureEdges.ManifoldEdgesOff()
    featureEdges.NonManifoldEdgesOff()
    featureEdges.Update()
    print(featureEdges)

    edgeMapper = vtk.vtkPolyDataMapper()
    edgeMapper.SetInputConnection(featureEdges.GetOutputPort())
    edgeActor = vtk.vtkActor()
    edgeActor.SetMapper(edgeMapper)

    diskMapper = vtk.vtkPolyDataMapper()
    diskMapper.SetInputData(surfITK)
    diskActor = vtk.vtkActor()
    diskActor.SetMapper(diskMapper)

    #Create a renderer, render window, and interactor
    renderer = vtk.vtkRenderer()
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.AddRenderer(renderer)
    renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderWindow)     
    renderer.AddActor(edgeActor)
    renderer.AddActor(diskActor)
    renderer.SetBackground(.3, .6, .3)

    renderWindow.Render()
    renderWindowInteractor.Start()

    return






















































































































