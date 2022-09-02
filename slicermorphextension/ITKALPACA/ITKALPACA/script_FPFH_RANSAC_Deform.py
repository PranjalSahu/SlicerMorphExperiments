#!/usr/bin/env python
# coding: utf-8

# In[ ]:

# Import all packages

from re import A
import sys


sys.path.insert(
    0, "/data/SlicerMorph/ITKALPACA-python-dependencies/lib/python3.9/site-packages/"
)

WRITE_PATH = "/data/Apedata/Slicer-cli-outputs/"

import numpy as np
import vtk
import itk
import math
import time
import copy
from vtk.util import numpy_support
from vtk.util.numpy_support import numpy_to_vtk
from scipy.spatial import cKDTree
from cpdalp import DeformableRegistration

# Install Dependencies using
# export WHEELS_PATH=/data/SlicerMorph/AllPythonWheels/
# export PREFIX_PATH=/data/SlicerMorph/ITKALPACA-python-dependencies
# ~/Slicer-5.0.3-linux-amd64/bin/PythonSlicer -m pip install --prefix=$PREFIX_PATH --no-index --find-links=$WHEELS_PATH --no-deps --force-reinstall --no-cache-dir -r requirements.txt

def getFiducialPoints(fiducialNumpyPoints):
    points = vtk.vtkPoints()
    for i in range(fiducialNumpyPoints.shape[0]):
      point = fiducialNumpyPoints[i]
      points.InsertNextPoint(point)
    return points

def applyTPSTransform(sourcePoints, targetPoints, inputPolyData, nodeName):
    transform = vtk.vtkThinPlateSplineTransform()
    transform.SetSourceLandmarks( sourcePoints)
    transform.SetTargetLandmarks( targetPoints )
    transform.SetBasisToR() # for 3D transform

    transformFilter = vtk.vtkTransformPolyDataFilter()
    transformFilter.SetInputData(inputPolyData)
    transformFilter.SetTransform(transform)
    transformFilter.Update()

    warpedPolyData = transformFilter.GetOutput()
    return warpedPolyData

def projectPointsPolydata(sourcePolydata, targetPolydata, originalPoints, rayLength):
    print("original points: ", originalPoints.GetNumberOfPoints())
    #set up polydata for projected points to return
    projectedPointData = vtk.vtkPolyData()
    projectedPoints = vtk.vtkPoints()
    projectedPointData.SetPoints(projectedPoints)

    #set up locater for intersection with normal vector rays
    obbTree = vtk.vtkOBBTree()
    obbTree.SetDataSet(targetPolydata)
    obbTree.BuildLocator()

    #set up point locator for finding surface normals and closest point
    pointLocator = vtk.vtkPointLocator()
    pointLocator.SetDataSet(sourcePolydata)
    pointLocator.BuildLocator()

    targetPointLocator = vtk.vtkPointLocator()
    targetPointLocator.SetDataSet(targetPolydata)
    targetPointLocator.BuildLocator()

    #get surface normal from each landmark point
    rayDirection=[0,0,0]
    normalArray = sourcePolydata.GetPointData().GetArray("Normals")
    if(not normalArray):
      print("no normal array, calculating....")
      normalFilter=vtk.vtkPolyDataNormals()
      normalFilter.ComputePointNormalsOn()
      normalFilter.SetInputData(sourcePolydata)
      normalFilter.Update()
      normalArray = normalFilter.GetOutput().GetPointData().GetArray("Normals")
      if(not normalArray):
        print("Error: no normal array")
        return projectedPointData
    for index in range(originalPoints.GetNumberOfPoints()):
      originalPoint= originalPoints.GetPoint(index)
      # get ray direction from closest normal
      closestPointId = pointLocator.FindClosestPoint(originalPoint)
      rayDirection = normalArray.GetTuple(closestPointId)
      rayEndPoint=[0,0,0]
      for dim in range(len(rayEndPoint)):
        rayEndPoint[dim] = originalPoint[dim] + rayDirection[dim]* rayLength
      intersectionIds=vtk.vtkIdList()
      intersectionPoints=vtk.vtkPoints()
      obbTree.IntersectWithLine(originalPoint,rayEndPoint,intersectionPoints,intersectionIds)
      #if there are intersections, update the point to most external one.
      if intersectionPoints.GetNumberOfPoints() > 0:
        exteriorPoint = intersectionPoints.GetPoint(intersectionPoints.GetNumberOfPoints()-1)
        projectedPoints.InsertNextPoint(exteriorPoint)
      #if there are no intersections, reverse the normal vector
      else:
        for dim in range(len(rayEndPoint)):
          rayEndPoint[dim] = originalPoint[dim] + rayDirection[dim]* -rayLength
        obbTree.IntersectWithLine(originalPoint,rayEndPoint,intersectionPoints,intersectionIds)
        if intersectionPoints.GetNumberOfPoints()>0:
          exteriorPoint = intersectionPoints.GetPoint(0)
          projectedPoints.InsertNextPoint(exteriorPoint)
        #if none in reverse direction, use closest mesh point
        else:
          closestPointId = targetPointLocator.FindClosestPoint(originalPoint)
          rayOrigin = targetPolydata.GetPoint(closestPointId)
          projectedPoints.InsertNextPoint(rayOrigin)
    return projectedPointData

def cpd_registration(
    targetArray,
    sourceArray,
    CPDIterations,
    CPDTolerance,
    alpha_parameter,
    beta_parameter,
):
    print('-----------------------------------------------------------')
    print("Before Deformable Distance ", get_euclidean_distance(targetArray, sourceArray))

    cloudSize = np.max(targetArray, 0) - np.min(targetArray)
    targetArray = targetArray*25/cloudSize
    sourceArray = sourceArray*25/cloudSize

    output = DeformableRegistration(
        **{
            "X": targetArray,
            "Y": sourceArray,
            "max_iterations": CPDIterations,
            "tolerance": CPDTolerance,
            "low_rank": True,
        },
        alpha=alpha_parameter,
        beta=beta_parameter,
    )
    deformed_array, _ = output.register()
    deformed_array = deformed_array*cloudSize/25
    targetArray = targetArray*cloudSize/25

    print("After Deformable Distance ", get_euclidean_distance(targetArray, deformed_array))
    print('-----------------------------------------------------------')
    return deformed_array


def get_euclidean_distance(input_fixedPoints, input_movingPoints):
    mesh_fixed = itk.Mesh[itk.D, 3].New()
    mesh_moving = itk.Mesh[itk.D, 3].New()

    mesh_fixed.SetPoints(itk.vector_container_from_array(input_fixedPoints.flatten()))
    mesh_moving.SetPoints(itk.vector_container_from_array(input_movingPoints.flatten()))

    MetricType = itk.EuclideanDistancePointSetToPointSetMetricv4.PSD3
    metric = MetricType.New()
    metric.SetMovingPointSet(mesh_moving)
    metric.SetFixedPointSet(mesh_fixed)
    metric.Initialize()

    return metric.GetValue()


def scale_mesh(input_mesh, scale_factor):
    """
    Scale the input_mesh by the given scale_factor iso-tropically
    """
    mesh_points = itk.array_from_vector_container(input_mesh.GetPoints())
    mesh_points = mesh_points * scale_factor
    input_mesh.SetPoints(itk.vector_container_from_array(mesh_points.flatten()))
    return input_mesh


# returns the points in numpy array
def subsample_points_poisson(inputMesh, radius=4.5):
    """
    Return sub-sampled points as numpy array.
    The radius might need to be tuned as per the requirements.
    """
    import vtk
    from vtk.util import numpy_support

    f = vtk.vtkPoissonDiskSampler()
    f.SetInputData(inputMesh)
    f.SetRadius(radius)
    f.Update()

    sampled_points = f.GetOutput()
    points = sampled_points.GetPoints()
    pointdata = points.GetData()
    as_numpy = numpy_support.vtk_to_numpy(pointdata)
    return as_numpy


def subsample_points_poisson_polydata(inputMesh, radius):
    '''
        Subsamples the points and returns vtk points so 
        that it has normal data also in it.
    '''
    import vtk
    from vtk.util import numpy_support

    f = vtk.vtkPoissonDiskSampler()
    f.SetInputData(inputMesh)
    f.SetRadius(radius)
    f.Update()
    sampled_points = f.GetOutput()
    return sampled_points


def write_vtk(vtk_polydata, filename):
    a = vtk.vtkPolyDataWriter()
    a.SetFileName(filename)
    a.SetInputData(vtk_polydata)
    a.SetFileVersion(42)
    a.SetFileTypeToBinary()
    a.Update()
    return


def readvtk(filename):
    a = vtk.vtkPolyDataReader()
    a.SetFileName(filename)
    a.Update()
    m1 = a.GetOutput()
    return m1

def readply(filename):
    a = vtk.vtkPLYReader()
    a.SetFileName(filename)
    a.Update()
    m1 = a.GetOutput()
    return m1

def readvtp(filename):
    a = vtk.vtkXMLPolyDataReader()
    a.SetFileName(filename)
    a.Update()
    m1 = a.GetOutput()
    return m1


def readvtkdata(filename):
    a = vtk.vtkDataReader()
    a.SetFileName(filename)
    a.Update()
    m1 = a.GetOutput()
    return m1


def read_landmarks(filename):
    """
    Reads the landmarks fscv file and converts the points to correct coordinate system.
    Returns array of size N x 3
    """
    import pandas as pd

    df = pd.read_csv(filename, comment="#", header=None)

    points_x = np.expand_dims(-1 * np.array(df[1].tolist()), -1)
    points_y = np.expand_dims(-1 * np.array(df[2].tolist()), -1)
    points_z = np.expand_dims(np.array(df[3].tolist()), -1)

    points = np.concatenate([points_x, points_y, points_z], axis=1)

    return points


def write_itk_mesh(input_mesh, filename):
    cells_array = itk.array_from_vector_container(input_mesh.GetCellsArray())
    w1 = itk.MeshFileWriter[type(input_mesh)].New()
    w1.SetFileName(filename)
    # w1.SetFileTypeAsBINARY()
    w1.SetInput(input_mesh)
    w1.Update()
    return


def getnormals(inputmesh):
    """
    To obtain the normal for each point from the triangle mesh.
    """
    normals = vtk.vtkTriangleMeshPointNormals()
    normals.SetInputData(inputmesh)
    normals.Update()
    return normals.GetOutput()


# def getnormals_pcl(inputPoints):
#     import open3d as o3d
#     A_pcd = o3d.geometry.PointCloud()
#     A_pcd.points = o3d.utility.Vector3dVector(inputPoints)
#     voxel_size = 5
#     radius_normal = voxel_size * 2
#     A_pcd.estimate_normals(
#         o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
#     return np.array(A_pcd.normals)


def getnormals_pca(movingMesh):
    """
    To get normals for each point in the pointset.
    """
    normals = vtk.vtkPCANormalEstimation()
    normals.SetSampleSize(30)
    normals.SetFlipNormals(True)
    # normals.SetNormalOrientationToPoint()
    normals.SetNormalOrientationToGraphTraversal()
    normals.SetInputData(movingMesh)
    normals.Update()
    as_numpy = numpy_support.vtk_to_numpy(
        normals.GetOutput().GetPointData().GetArray(0)
    )
    return as_numpy


def extract_normal_from_tuple(input_mesh):
    """
    Extracts the normal data from the sampled points.
    Returns points and normals for the points.
    """
    t1 = input_mesh.GetPointData().GetArray("Normals")
    n1_array = []
    for i in range(t1.GetNumberOfTuples()):
        n1_array.append(t1.GetTuple(i))
    n1_array = np.array(n1_array)

    points = input_mesh.GetPoints()
    pointdata = points.GetData()
    as_numpy = numpy_support.vtk_to_numpy(pointdata)

    return as_numpy, n1_array


def final_iteration(fixedPoints, movingPoints, transform_type):
    """
    Perform the final iteration of alignment.

    Args:
        fixedPoints, movingPoints, transform_type: 0 or 1 or 2
    Returns:
        (tranformed movingPoints, tranform)
    """
    mesh_fixed = itk.Mesh[itk.D, 3].New()
    mesh_moving = itk.Mesh[itk.D, 3].New()
    mesh_fixed.SetPoints(itk.vector_container_from_array(fixedPoints.flatten()))
    mesh_moving.SetPoints(itk.vector_container_from_array(movingPoints.flatten()))

    if transform_type == 0:
        TransformType = itk.Euler3DTransform[itk.D]
    elif transform_type == 1:
        TransformType = itk.ScaleVersor3DTransform[itk.D]
    elif transform_type == 2:
        TransformType = itk.Rigid3DTransform[itk.D]

    transform = TransformType.New()
    transform.SetIdentity()

    MetricType = itk.EuclideanDistancePointSetToPointSetMetricv4.PSD3
    metric = MetricType.New()
    metric.SetMovingPointSet(mesh_moving)
    metric.SetFixedPointSet(mesh_fixed)
    metric.SetMovingTransform(transform)
    metric.Initialize()

    number_of_epochs = 1000
    optimizer = itk.GradientDescentOptimizerv4Template[itk.D].New()
    optimizer.SetNumberOfIterations(number_of_epochs)
    optimizer.SetLearningRate(0.00001)
    optimizer.SetMinimumConvergenceValue(0.0)
    optimizer.SetConvergenceWindowSize(number_of_epochs)
    optimizer.SetMetric(metric)

    def print_iteration():
        if optimizer.GetCurrentIteration()%100 == 0:
            print(
                f"It: {optimizer.GetCurrentIteration()}"
                f" metric value: {optimizer.GetCurrentMetricValue():.6f} "
            )
    optimizer.AddObserver(itk.IterationEvent(), print_iteration)
    optimizer.StartOptimization()

    # Get the correct transform and perform the final alignment
    current_transform = metric.GetMovingTransform().GetInverseTransform()
    itk_transformed_mesh = itk.transform_mesh_filter(
        mesh_moving, transform=current_transform
    )

    return (
        itk.array_from_vector_container(itk_transformed_mesh.GetPoints()),
        current_transform,
    )

# RANSAC using package
def ransac_using_package(
    movingMeshPoints,
    fixedMeshPoints,
    movingMeshFeaturePoints,
    fixedMeshFeaturePoints,
    number_of_iterations,
    number_of_ransac_points,
    inlier_value):

    def GenerateData(data, agreeData):
        """
            In current implmentation the agreedata contains two corressponding 
            points from moving and fixed mesh. However, after the subsampling step the 
            number of points need not be equal in those meshes. So we randomly sample 
            the points from larger mesh.
        """
        data.reserve(movingMeshFeaturePoints.shape[0])
        for i in range(movingMeshFeaturePoints.shape[0]):
            point1 = movingMeshFeaturePoints[i]
            point2 = fixedMeshFeaturePoints[i]
            input_data = [point1[0], point1[1], point1[2], point2[0], point2[1], point2[2]]
            input_data = [float(x) for x in input_data]
            data.push_back(input_data)
    
        count_min = int(np.min([movingMeshPoints.shape[0], fixedMeshPoints.shape[0]]))

        mesh1_points  = copy.deepcopy(movingMeshPoints)
        mesh2_points  = copy.deepcopy(fixedMeshPoints)

        agreeData.reserve(count_min)
        for i in range(count_min):
            point1 = mesh1_points[i]
            point2 = mesh2_points[i]
            input_data = [point1[0], point1[1], point1[2], point2[0], point2[1], point2[2]]
            input_data = [float(x) for x in input_data]
            agreeData.push_back(input_data)
        return

    data = itk.vector[itk.Point[itk.D, 6]]()
    agreeData = itk.vector[itk.Point[itk.D, 6]]()
    GenerateData(data, agreeData)

    transformParameters = itk.vector.D()
    bestTransformParameters = itk.vector.D()

    maximumDistance = inlier_value
    RegistrationEstimatorType = itk.Ransac.LandmarkRegistrationEstimator[6]
    registrationEstimator = RegistrationEstimatorType.New()
    registrationEstimator.SetMinimalForEstimate(number_of_ransac_points)
    registrationEstimator.SetAgreeData(agreeData)
    registrationEstimator.SetDelta(maximumDistance)
    registrationEstimator.LeastSquaresEstimate(data, transformParameters)

    bestPercentage = 0
    bestParameters = None

    desiredProbabilityForNoOutliers = 0.99
    RANSACType = itk.RANSAC[itk.Point[itk.D, 6], itk.D]
    ransacEstimator = RANSACType.New()
    ransacEstimator.SetData(data)
    ransacEstimator.SetAgreeData(agreeData)
    ransacEstimator.SetMaxIteration(number_of_iterations)
    ransacEstimator.SetNumberOfThreads(16)
    ransacEstimator.SetParametersEstimator(registrationEstimator)
    
    for i in range(1):    
        percentageOfDataUsed = ransacEstimator.Compute( transformParameters, desiredProbabilityForNoOutliers )
        print(i, ' Percentage of data used is ', percentageOfDataUsed)
        if percentageOfDataUsed > bestPercentage:
            bestPercentage = percentageOfDataUsed
            bestParameters = transformParameters

    print("Percentage of points used ", bestPercentage)
    transformParameters = bestParameters
    transform = itk.Similarity3DTransform.D.New()
    p = transform.GetParameters()
    f = transform.GetFixedParameters()
    for i in range(7):
        p.SetElement(i, transformParameters[i])
    counter = 0
    for i in range(7, 10):
        f.SetElement(counter, transformParameters[i])
        counter = counter + 1
    transform.SetParameters(p)
    transform.SetFixedParameters(f)
    return itk.dict_from_transform(transform), bestPercentage, bestPercentage

# RANSAC with VTK Landmark Transform
def ransac_icp_parallel_vtk(
    movingMeshPoints,
    fixedMeshPoints,
    number_of_iterations,
    number_of_ransac_points,
    transform_type,
    inlier_value,
):
    """
    Perform Ransac by doing parallel iterations for different samples.
    """
    import numpy as np

    np.random.seed(0)

    all_points1 = movingMeshPoints
    all_points2 = fixedMeshPoints

    def process_task(
        i,
        number_of_ransac_points,
        return_result,
        inlier_points=None,
    ):
        """
        Args:
            i: input seed used to sample the points.
            number_of_ransac_points: Number of random points selected to perform the registration.
            return_result: Whether to return the transformed mesh.

        Returns: (Best Value, seed) or the transformed_points depending on return_result flag.

        """

        # Create Mesh inside the method to tackle the problem of serialization
        import itk
        import vtk
        from vtk.util import numpy_support

        def itk_transform_from_vtk(vtk_transform):
            def vtkmatrix_to_numpy(matrix):
                """
                Copies the elements of a vtkMatrix4x4 into a numpy array.

                :param matrix: The matrix to be copied into an array.
                :type matrix: vtk.vtkMatrix4x4
                :rtype: numpy.ndarray
                """
                import numpy as np

                m = np.ones((4, 4))
                for i in range(4):
                    for j in range(4):
                        m[i, j] = matrix.GetElement(i, j)
                return np.array(m)

            m = vtk_transform.GetMatrix()
            tp = vtkmatrix_to_numpy(m)

            itks = itk.Similarity3DTransform.New()
            itks.Translate(list(tp[:3, 3]))
            itks.SetMatrix(itk.matrix_from_array(tp[:3, :3]))

            return itks

        """ Create small point sets """
        np.random.seed(i)

        if inlier_points is None:
            random_indices = np.random.choice(
                all_points1.shape[0], size=number_of_ransac_points
            )
            A_corr_temp = all_points1[random_indices, :]
            B_corr_temp = all_points2[random_indices, :]
        else:
            A_corr_temp = all_points1[inlier_points, :]
            B_corr_temp = all_points2[inlier_points, :]

        """ Perform vtkLandMarkTransform based registration using selected points """
        source_points = vtk.vtkPoints()
        source_points.SetData(numpy_support.numpy_to_vtk(A_corr_temp, deep=True))

        target_points = vtk.vtkPoints()
        target_points.SetData(numpy_support.numpy_to_vtk(B_corr_temp, deep=True))

        landmarkTransform = vtk.vtkLandmarkTransform()
        landmarkTransform.SetModeToSimilarity()
        # landmarkTransform.SetModeToRigidBody()
        landmarkTransform.SetSourceLandmarks(source_points)
        landmarkTransform.SetTargetLandmarks(target_points)
        landmarkTransform.Update()

        def convertpoint(inputarray):
            inputarray = inputarray.astype("float64")
            vector_container = itk.VectorContainer[itk.UL, itk.Point.D3].New()
            vector_container.Reserve(inputarray.shape[0])
            for i in range(inputarray.shape[0]):
                vector_container.SetElement(i, list(inputarray[i]))
            return vector_container

        """ For converting VTK Transform to ITK Transform """
        current_transform = itk_transform_from_vtk(landmarkTransform)
        """ For transforming the pointset using the obtained transform """
        temp_pointset = itk.Mesh[itk.D, 3].New()
        temp_pointset.SetPoints(
            itk.vector_container_from_array(all_points1.flatten().astype("float32"))
        )

        new_pointset = itk.transform_mesh_filter(
            temp_pointset, transform=current_transform
        )
        new_pointset = itk.array_from_vector_container(new_pointset.GetPoints())

        # Get the distance of each landmark and get count of inliers
        diff_numpy = np.square(new_pointset - all_points2)
        diff_numpy = np.sqrt(np.sum(diff_numpy, 1))

        inlier_array = diff_numpy < inlier_value
        current_value = np.sum(inlier_array)
        inlier_array = np.nonzero(inlier_array)[0]

        if return_result:
            return itk.dict_from_transform(current_transform)
        else:
            return (current_value, inlier_array)

    # Spawn multiple jobs to utilize all cores
    # Test code for not using parallel threads
    results = []
    for i in range(number_of_iterations):
        results.append(process_task(i, number_of_ransac_points, 0))

    # results = Parallel(n_jobs=-1)(
    #    delayed(process_task)(i, mesh_sub_sample_points, number_of_ransac_points, 0)
    #    for i in range(number_of_iterations))

    results_values = []
    for k in results:
        results_values.append(k[0])

    # Sort the results and get the best one i.e. the lowest one
    index = np.argmax(results_values)
    value = results_values[index]

    final_result = process_task(index, number_of_ransac_points, 1, results[index][1])

    return final_result, index, value


def orient_points(input_points, x, y, z):
    """
    Orients the input points based on the x, y and z orientations given.
    """
    output_points = copy.deepcopy(input_points)
    output_points[:, 0] = output_points[:, 0] * x
    output_points[:, 1] = output_points[:, 1] * y
    output_points[:, 2] = output_points[:, 2] * z

    return output_points


def orient_points_in_mesh(input_mesh, x, y, z):
    input_points = itk.array_from_vector_container(input_mesh.GetPoints())
    oriented_points = orient_points(input_points, x, y, z)
    input_mesh.SetPoints(itk.vector_container_from_array(oriented_points.flatten()))
    return


def get_fpfh_feature(points_np, normals_np, radius, neighbors):
    pointset = itk.PointSet[itk.F, 3].New()
    pointset.SetPoints(
        itk.vector_container_from_array(points_np.flatten().astype("float32"))
    )

    normalset = itk.PointSet[itk.F, 3].New()
    normalset.SetPoints(
        itk.vector_container_from_array(normals_np.flatten().astype("float32"))
    )

    fpfh = itk.Fpfh.PointFeature.MF3MF3.New()
    fpfh.ComputeFPFHFeature(pointset, normalset, int(radius), int(neighbors))
    result = fpfh.GetFpfhFeature()

    fpfh_feats = itk.array_from_vector_container(result)
    fpfh_feats = np.reshape(fpfh_feats, [33, pointset.GetNumberOfPoints()]).T
    return fpfh_feats


def get_euclidean_distance(fixed_points_np, moving_points_np):
    fixed_mesh = itk.Mesh[itk.D, 3].New()
    moving_mesh = itk.Mesh[itk.D, 3].New()
    fixed_mesh.SetPoints(itk.vector_container_from_array(fixed_points_np.flatten().astype('float32')))
    moving_mesh.SetPoints(itk.vector_container_from_array(moving_points_np.flatten().astype('float32')))
    MetricType = itk.EuclideanDistancePointSetToPointSetMetricv4.PSD3
    metric = MetricType.New()
    metric.SetMovingPointSet(moving_mesh)
    metric.SetFixedPointSet(fixed_mesh)
    metric.Initialize()
    return metric.GetValue()


# Convert from numpy to vtk points
def numpy_to_vtk_points(input_points):
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_support.numpy_to_vtk(input_points, deep=True))
    return vtk_points


# Convert from numpy to vtkPolyData
def numpy_to_vtk_polydata(input_points):
    vtk_polydata = vtk.vtkPolyData()
    vtk_points = numpy_to_vtk_points(input_points)
    vtk_polydata.SetPoints(vtk_points)
    return vtk_polydata


# Convert from vtk points to numpy
def vtk_points_to_numpy(input_vtk_points):
    input_vtk_points = input_vtk_points.GetPoints()
    input_vtk_points = input_vtk_points.GetData()
    input_vtk_points = numpy_support.vtk_to_numpy(input_vtk_points)
    return input_vtk_points


def transform_numpy_points(points_np, transform):
    mesh = itk.Mesh[itk.F, 3].New()
    mesh.SetPoints(itk.vector_container_from_array(points_np.flatten().astype('float32')))
    transformed_mesh = itk.transform_mesh_filter(mesh, transform=transform)
    points_tranformed = itk.array_from_vector_container(transformed_mesh.GetPoints())
    points_tranformed = np.reshape(points_tranformed, [-1, 3])
    return points_tranformed


def find_knn_cpu(feat0, feat1, knn=1, return_distance=False):
    feat1tree = cKDTree(feat1)
    dists, nn_inds = feat1tree.query(feat0, k=knn)
    if return_distance:
        return nn_inds, dists
    else:
        return nn_inds


def find_correspondences(feats0, feats1, mutual_filter=True):
    '''
        Using the FPFH features find noisy corresspondes.
        These corresspondes will be used inside the RANSAC.
    '''
    nns01, dists1 = find_knn_cpu(feats0, feats1, knn=1, return_distance=True)
    corres01_idx0 = np.arange(len(nns01))
    corres01_idx1 = nns01

    if not mutual_filter:
        return corres01_idx0, corres01_idx1

    nns10, dists2 = find_knn_cpu(feats1, feats0, knn=1, return_distance=True)
    corres10_idx1 = np.arange(len(nns10))
    corres10_idx0 = nns10

    mutual_filter = corres10_idx0[corres01_idx1] == corres01_idx0
    corres_idx0 = corres01_idx0[mutual_filter]
    corres_idx1 = corres01_idx1[mutual_filter]

    return corres_idx0, corres_idx1


def get_numpy_points_from_vtk(vtk_polydata):
    """
    Returns the points as numpy from a vtk_polydata
    """
    points = vtk_polydata.GetPoints()
    pointdata = points.GetData()
    points_as_numpy = numpy_support.vtk_to_numpy(pointdata)
    return points_as_numpy


def set_numpy_points_in_vtk(vtk_polydata, points_as_numpy):
    """
    Sets the numpy points to a vtk_polydata
    """
    vtk_data_array = numpy_support.numpy_to_vtk(
        num_array=points_as_numpy, deep=True, array_type=vtk.VTK_FLOAT
    )
    points2 = vtk.vtkPoints()
    points2.SetData(vtk_data_array)
    vtk_polydata.SetPoints(points2)
    return


def transform_points_in_vtk(vtk_polydata, itk_transform):
    points_as_numpy = get_numpy_points_from_vtk(vtk_polydata)
    transformed_points = transform_numpy_points(points_as_numpy, itk_transform)
    set_numpy_points_in_vtk(vtk_polydata, transformed_points)
    return


def add_offset_to_itk_mesh(input_itk_mesh, offset):
    points = itk.array_from_vector_container(input_itk_mesh.GetPoints())
    points = points + offset
    input_itk_mesh.SetPoints(
        itk.vector_container_from_array(points.flatten().astype("float32"))
    )
    return


def add_offset_to_vtk_mesh(input_vtk_mesh, offset):
    numpy_points = get_numpy_points_from_vtk(input_vtk_mesh)
    numpy_points = numpy_points + offset
    set_numpy_points_in_vtk(input_vtk_mesh, numpy_points)
    return


"""
    Given the path of the meshes perform the registration.
    Args:
        target: Mesh which needs to be landmarked.
        source: Atlas mesh which will be deformed to align it with target mesh.
"""


def process(
    target,
    source,
    subsample_radius,
    number_of_ransac_points,
    inlier_value,
    fpfh_radius,
    fpfh_neighbors,
    deform_sigma,
    deform_neighbourhood,
    bspline_grid,
    deformable_iterations,
    ransac_iterations,
    alpha_parameter,
    beta_parameter,
    landmark_file,
    cpd_flag
):
    landmark_points = read_landmarks(landmark_file)
    print('Landmark points are ', landmark_points.shape)
    
    paths = [target, source]

    # Write the meshes in vtk format so that they can be read in ITK
    vtk_meshes = list()
    filetype = None
    for path in paths:
        print("Reading file ", path)
        if path.endswith('.vtp'):
            vtk_mesh = readvtp(path)
            filetype = '.vtp'
        elif path.endswith('.ply'):
            vtk_mesh = readply(path)
            filetype = '.ply'
        else:
            vtk_mesh = readvtk(path)
            filetype = '.vtk'

        # Get largest connected component and clean the mesh to remove un-used points
        connectivityFilter = vtk.vtkPolyDataConnectivityFilter()
        connectivityFilter.SetInputData(vtk_mesh)
        connectivityFilter.SetExtractionModeToLargestRegion()
        connectivityFilter.Update()
        vtk_mesh_out = connectivityFilter.GetOutput()

        filter1 = vtk.vtkCleanPolyData()
        filter1.SetInputData(vtk_mesh_out)
        filter1.Update()
        vtk_mesh_out_clean = filter1.GetOutput()

        triangle_filter = vtk.vtkTriangleFilter()
        triangle_filter.SetInputData(vtk_mesh_out_clean)
        triangle_filter.SetPassLines(False)
        triangle_filter.SetPassVerts(False)
        triangle_filter.Update()
        vtk_mesh_out_clean = triangle_filter.GetOutput()

        vtk_meshes.append(vtk_mesh_out_clean)

    # Scale the mesh and the landmark points
    box_filter = vtk.vtkBoundingBox()
    box_filter.SetBounds(vtk_meshes[0].GetBounds())
    fixedlength = box_filter.GetDiagonalLength()

    box_filter = vtk.vtkBoundingBox()
    box_filter.SetBounds(vtk_meshes[1].GetBounds())
    movinglength = box_filter.GetDiagonalLength()

    print("Scale length are  ", fixedlength, movinglength)
    scale_factor = fixedlength / movinglength

    points = vtk_meshes[1].GetPoints()
    pointdata = points.GetData()
    points_as_numpy = numpy_support.vtk_to_numpy(pointdata)
    points_as_numpy = points_as_numpy * scale_factor

    landmark_points = landmark_points * scale_factor
    set_numpy_points_in_vtk(vtk_meshes[1], points_as_numpy)

    # Get the offsets to align the minimum of moving and fixed mesh
    vtk_mesh_offsets = []
    for i, mesh in enumerate(vtk_meshes):
        mesh_points = get_numpy_points_from_vtk(mesh)
        vtk_mesh_offset = np.min(mesh_points, axis=0)
        vtk_mesh_offsets.append(vtk_mesh_offset)

    # Only move the moving mesh
    mesh_points = get_numpy_points_from_vtk(vtk_meshes[1])
    offset_amount = (vtk_mesh_offsets[1] - vtk_mesh_offsets[0])
    mesh_points = mesh_points - offset_amount
    set_numpy_points_in_vtk(vtk_meshes[1], mesh_points)
    # Move the moving landmark_points by same amount
    landmark_points = landmark_points - offset_amount

    fixedMesh = vtk_meshes[0]
    movingMesh = vtk_meshes[1]

    # Write back out to a filetype supported by ITK
    vtk_paths = [path.strip(filetype) + ".vtk" for path in paths]
    casenames = []
    for idx, _ in enumerate(vtk_meshes):
        casename = vtk_paths[idx].split("/")[-1].split(".")[0].split("_")[0]
        casenames.append(casename)
    
    for idx, mesh in enumerate(vtk_meshes):
        if idx == 1:
            write_path = WRITE_PATH + casenames[0] + "_movingMesh.vtk"
        else:
            write_path = WRITE_PATH + casenames[0] + "_fixedMesh.vtk"
        print("Writing mesh ", write_path)
        write_vtk(mesh, write_path)

    # Write the cleaned Initialized meshes
    movingMeshPath = WRITE_PATH + casenames[0] + "_movingMesh.vtk"
    fixedMeshPath = WRITE_PATH + casenames[0] + "_fixedMesh.vtk"

    # For performing RANSAC in parallel
    movingMesh_vtk = readvtk(movingMeshPath)
    fixedMesh_vtk = readvtk(fixedMeshPath)

    movingFullMesh_vtk = getnormals(movingMesh_vtk)
    fixedFullMesh_vtk = getnormals(fixedMesh_vtk)

    # Sub-Sample the points for rigid refinement and deformable registration
    subsample_radius_moving = subsample_radius
    subsample_radius_fixed = subsample_radius
    while (True):
        print('Resampling moving with radius = ', subsample_radius_moving)
        movingMesh_vtk = subsample_points_poisson_polydata(
            movingFullMesh_vtk, radius=subsample_radius_moving
        )
        if movingMesh_vtk.GetNumberOfPoints() < 5000:
            break
        else:
            subsample_radius_moving = subsample_radius_moving + 0.25
    
    while (True):
        print('Resampling fixed with radius = ', subsample_radius_fixed)
        fixedMesh_vtk = subsample_points_poisson_polydata(
            fixedFullMesh_vtk, radius=subsample_radius_fixed
        )
        if fixedMesh_vtk.GetNumberOfPoints() < 5000:
            break
        else:
            subsample_radius_fixed = subsample_radius_fixed + 0.25

    movingMeshPoints, movingMeshPointNormals = extract_normal_from_tuple(movingMesh_vtk)
    fixedMeshPoints, fixedMeshPointNormals = extract_normal_from_tuple(fixedMesh_vtk)

    print('------------------------------------------------------------')
    print("movingMeshPoints.shape ", movingMeshPoints.shape)
    print("movingMeshPointNormals.shape ", movingMeshPointNormals.shape)
    print("fixedMeshPoints.shape ", fixedMeshPoints.shape)
    print("fixedMeshPointNormals.shape ", fixedMeshPointNormals.shape)
    print('------------------------------------------------------------')

    # New FPFH Code
    pcS = np.expand_dims(fixedMeshPoints, -1)
    normal_np_pcl = fixedMeshPointNormals
    fixed_feats = get_fpfh_feature(pcS, normal_np_pcl, fpfh_radius, fpfh_neighbors)

    pcS = np.expand_dims(movingMeshPoints, -1)
    normal_np_pcl = movingMeshPointNormals
    moving_feats = get_fpfh_feature(pcS, normal_np_pcl, fpfh_radius, fpfh_neighbors)

    # Establish correspondences by nearest neighbour search in feature space
    corrs_A, corrs_B = find_correspondences(
        fixed_feats, moving_feats, mutual_filter=True
    )

    fixedMeshPoints = fixedMeshPoints.T
    movingMeshPoints = movingMeshPoints.T

    fixed_corr = fixedMeshPoints[:, corrs_A]    # np array of size 3 by num_corrs
    moving_corr = movingMeshPoints[:, corrs_B]  # np array of size 3 by num_corrs

    num_corrs = fixed_corr.shape[1]
    print(f"FPFH generates {num_corrs} putative correspondences.")

    print(fixed_corr.shape, moving_corr.shape)
    
    write_vtk(numpy_to_vtk_polydata(moving_corr.T), WRITE_PATH + casenames[0] + "_moving_corr.vtk")
    write_vtk(numpy_to_vtk_polydata(fixed_corr.T), WRITE_PATH + casenames[0] + "_fixed_corr.vtk")
    write_vtk(numpy_to_vtk_polydata(movingMeshPoints.T), WRITE_PATH + casenames[0] + "_movingMeshPointsBefore.vtk")
    write_vtk(numpy_to_vtk_polydata(fixedMeshPoints.T), WRITE_PATH + casenames[0] + "_fixedMeshPoints.vtk")

    import time
    bransac = time.time()
    print('Before RANSAC ', bransac)
    # Perform Initial alignment using Ransac parallel iterations
    transform_matrix, _, value = ransac_using_package(
        movingMeshPoints=movingMeshPoints.T,
        fixedMeshPoints=fixedMeshPoints.T,
        movingMeshFeaturePoints=moving_corr.T,
        fixedMeshFeaturePoints=fixed_corr.T,
        number_of_iterations=ransac_iterations,
        number_of_ransac_points=number_of_ransac_points,
        inlier_value=inlier_value,
    )
    aransac = time.time()
    print('After RANSAC ', aransac)
    print('Total Duraction ', aransac - bransac)
    
    transform_matrix = itk.transform_from_dict(transform_matrix)
    
    transform_points_in_vtk(movingMesh, transform_matrix)
    write_vtk(movingMesh, WRITE_PATH + casenames[0] + "_movingMeshRANSAC.vtk")

    movingMeshPoints = movingMeshPoints.T
    fixedMeshPoints = fixedMeshPoints.T

    print("movingMeshPoints.shape ", movingMeshPoints.shape)
    print("fixedMeshPoints.shape ", fixedMeshPoints.shape)

    landmark_points = transform_numpy_points(landmark_points, transform_matrix)
    movingMeshPoints = transform_numpy_points(movingMeshPoints, transform_matrix)

    print('-----------------------------------------------------------')
    print("Starting Rigid Refinement")
    print("Before Distance ", get_euclidean_distance(fixedMeshPoints, movingMeshPoints))
    transform_type = 0
    final_mesh_points, second_transform = final_iteration(
        fixedMeshPoints, movingMeshPoints, transform_type
    )
    print("After Distance ", get_euclidean_distance(fixedMeshPoints, final_mesh_points))

    write_vtk(numpy_to_vtk_polydata(movingMeshPoints), WRITE_PATH + casenames[0] + "_movingMeshPoints.vtk")
    write_vtk(numpy_to_vtk_polydata(final_mesh_points), WRITE_PATH + casenames[0] + "_final_mesh_points.vtk")

    landmark_points = transform_numpy_points(landmark_points, second_transform)
    transform_points_in_vtk(movingMesh, second_transform)
    write_vtk(movingMesh, WRITE_PATH + casenames[0] + "_movingMeshRigidRegistered.vtk")
    
    print("Completed Rigid Refinement")
    print('-----------------------------------------------------------')

    print('cpd_flag is ', cpd_flag)
    if int(cpd_flag) == 1:
        combined_points = np.concatenate([landmark_points, final_mesh_points], axis=0)
        print('Combined points shape ', combined_points.shape, landmark_points.shape, final_mesh_points.shape)
        CPDIterations = deformable_iterations
        CPDTolerance = 0.001
        outputArray = cpd_registration(fixedMeshPoints,
            combined_points,
            CPDIterations,
            CPDTolerance,
            alpha_parameter,
            beta_parameter)
        print('CPD Result ', outputArray.shape)

        np.save(WRITE_PATH + casenames[0] + "_cpdResultLandmarks.npy", outputArray[:landmark_points.shape[0], :])
        np.save(WRITE_PATH + casenames[0] + "_cpdResult.npy", outputArray[landmark_points.shape[0]:, :])

        outputPoints_vtk = getFiducialPoints(outputArray[:landmark_points.shape[0], :])
        inputPoints_vtk = getFiducialPoints(landmark_points)

        nodeName = 'TPS Warped Model'
        warpedModel = applyTPSTransform(inputPoints_vtk, outputPoints_vtk, movingMesh, nodeName)
        write_vtk(warpedModel, WRITE_PATH + casenames[0] + "_movingMeshFinalRegistered_CPD.vtk")

        maxProjectionFactor = 1/100.0
        maxProjection = (fixedFullMesh_vtk.GetLength()) * maxProjectionFactor
        print("Max Projection is ", maxProjection)
        projectedPoints = projectPointsPolydata(warpedModel, fixedFullMesh_vtk, outputPoints_vtk, maxProjection)
        projectedPoints = projectedPoints.GetPoints().GetData()
        projectedPoints = numpy_support.vtk_to_numpy(projectedPoints)
        print('projectedPoints shape is ', projectedPoints.shape)
        #projectedPoints = getFiducialPoints(projectedPoints)
        #write_vtk(projectedPoints, WRITE_PATH + casename + "_cpdResultLandmarksProjected.vtk")
        np.save(WRITE_PATH + casenames[0] + "_cpdResultLandmarksProjected.npy", projectedPoints)
        return
    else:
        # [STAR] Expectation Based PointSetToPointSetMetricv4 Registration
        imageDiagonal = 100
        PixelType = itk.D
        Dimension = 3

        FixedImageType = itk.Image[PixelType, Dimension]

        # Create PointSets for registration
        movingPS = itk.PointSet[itk.D, Dimension].New()
        fixedPS = itk.PointSet[itk.D, Dimension].New()

        movingPS.SetPoints(itk.vector_container_from_array(final_mesh_points.flatten()))
        fixedPS.SetPoints(itk.vector_container_from_array(fixedMeshPoints.flatten()))

        # For getting the Bounding Box
        ElementIdentifierType = itk.UL
        CoordType = itk.F
        Dimension = 3

        VecContType = itk.VectorContainer[
            ElementIdentifierType, itk.Point[CoordType, Dimension]
        ]
        bounding_box = itk.BoundingBox[
            ElementIdentifierType, Dimension, CoordType, VecContType
        ].New()

        bounding_box.SetPoints(movingPS.GetPoints())
        bounding_box.ComputeBoundingBox()

        minBounds = np.array(bounding_box.GetMinimum())
        maxBounds = np.array(bounding_box.GetMaximum())

        spacing = np.sqrt(bounding_box.GetDiagonalLength2()) / imageDiagonal
        diff = maxBounds - minBounds

        print("Spacing ", spacing)
        print("minBounds ", minBounds)
        print("maxBounds ", maxBounds)

        fixedImageSize = [0] * 3
        fixedImageSize[0] = math.ceil(1.25 * diff[0] / spacing)
        fixedImageSize[1] = math.ceil(1.25 * diff[1] / spacing)
        fixedImageSize[2] = math.ceil(1.25 * diff[2] / spacing)

        fixedImageOrigin = [0] * 3
        fixedImageOrigin[0] = minBounds[0] - 0.25 * diff[0]
        fixedImageOrigin[1] = minBounds[1] - 0.25 * diff[1]
        fixedImageOrigin[2] = minBounds[2] - 0.25 * diff[2]

        fixedImageSpacing = np.ones(3) * spacing
        fixedImageDirection = np.identity(3)

        fixedImage = FixedImageType.New()
        fixedImage.SetRegions(fixedImageSize)
        fixedImage.SetOrigin(fixedImageOrigin)
        fixedImage.SetDirection(fixedImageDirection)
        fixedImage.SetSpacing(fixedImageSpacing)
        fixedImage.Allocate()

        # Create BSpline Transformation object and initialize the parameters
        SplineOrder = 3
        TransformType = itk.BSplineTransform[itk.D, Dimension, SplineOrder]
        InitializerType = itk.BSplineTransformInitializer[TransformType, FixedImageType]

        transform = TransformType.New()

        numberOfGridNodesInOneDimension = bspline_grid
        transformInitializer = InitializerType.New()
        transformInitializer.SetTransform(transform)
        transformInitializer.SetImage(fixedImage)
        transformInitializer.SetTransformDomainMeshSize(
            numberOfGridNodesInOneDimension - SplineOrder
        )
        transformInitializer.InitializeTransform()

        # Registration Loop
        numOfIterations = deformable_iterations
        maxStep = 0.1
        learningRate = 0.1

        # Good combinations
        # 10000
        # sigma: 3, Kneighbourhood 20, bspline: 4 -> 3.72
        # sigma: 3, Kneighbourhood 20, bspline: 8 -> 3.77
        # sigma: 3, Kneighbourhood 20, bspline: 6 -> 3.42
        # sigma: 3, Kneighbourhood 20, bspline: 5 -> 3.70 -> best (by qualitative comparison)

        MetricType = itk.ExpectationBasedPointSetToPointSetMetricv4[type(movingPS)]
        metric = MetricType.New()
        metric.SetFixedPointSet(movingPS)
        metric.SetMovingPointSet(fixedPS)
        metric.SetPointSetSigma(deform_sigma)
        metric.SetEvaluationKNeighborhood(deform_neighbourhood)
        metric.SetMovingTransform(transform)
        metric.Initialize()

        print("Metric Created")

        optimizer = itk.RegularStepGradientDescentOptimizerv4.D.New()
        optimizer.SetNumberOfIterations(numOfIterations)
        optimizer.SetMaximumStepSizeInPhysicalUnits(maxStep)
        optimizer.SetLearningRate(learningRate)
        optimizer.SetMinimumConvergenceValue(-100)
        optimizer.SetConvergenceWindowSize(numOfIterations)
        optimizer.SetMetric(metric)

        def iteration_update():
            if optimizer.GetCurrentIteration() % 100 == 0:
                print(
                    f"It: {optimizer.GetCurrentIteration()}"
                    f" metric value: {optimizer.GetCurrentMetricValue():.6f} "
                )
            return

        iteration_command = itk.PyCommand.New()
        iteration_command.SetCommandCallable(iteration_update)
        optimizer.AddObserver(itk.IterationEvent(), iteration_command)
        optimizer.StartOptimization()

        # Transform the point set using the final transform
        final_transform = metric.GetMovingTransform()

        e_metric = itk.EuclideanDistancePointSetToPointSetMetricv4.PSD3.New()
        e_metric.SetFixedPointSet(fixedPS)
        e_metric.SetMovingPointSet(movingPS)
        print("Euclidean Metric Before TSD Deformable Registration ", e_metric.GetValue())

        movingPSNew = itk.PointSet[itk.D, 3].New()
        numberOfPoints = movingPS.GetNumberOfPoints()

        for n in range(0, numberOfPoints):
            movingPSNew.SetPoint(n, final_transform.TransformPoint(movingPS.GetPoint(n)))

        e_metric = itk.EuclideanDistancePointSetToPointSetMetricv4.PSD3.New()
        e_metric.SetFixedPointSet(fixedPS)
        e_metric.SetMovingPointSet(movingPSNew)
        print("Euclidean Metric After TSD Deformable Registration ", e_metric.GetValue())

        # Write the Displacement Field
        write_displacement_field = False
        if write_displacement_field:
            convertFilter = itk.TransformToDisplacementFieldFilter.IVF33D.New()
            convertFilter.SetTransform(final_transform)
            convertFilter.UseReferenceImageOn()
            convertFilter.SetReferenceImage(fixedImage)
            convertFilter.Update()
            field = convertFilter.GetOutput()
            field = np.array(field)
            np.save(WRITE_PATH + "displacement_field.npy", field)

        # Write the final registered mesh
        movingMeshPath = WRITE_PATH + casenames[0] + "_movingMeshRigidRegistered.vtk"
        print('movingMeshPath is ', movingMeshPath)
        movingMesh = readvtk(movingMeshPath)
        
        landmark_points_transformed = []
        for i in range(len(landmark_points)):
            landmark_points_transformed.append( final_transform.TransformPoint(itk.Point.F3([float(landmark_points[i][0]), float(landmark_points[i][1]), float(landmark_points[i][2])])))
        landmark_points_transformed = np.array(landmark_points_transformed)

        outputPoints_vtk = getFiducialPoints(landmark_points_transformed)
        inputPoints_vtk = getFiducialPoints(landmark_points)

        nodeName = 'TPS Warped Model'
        warpedModel = applyTPSTransform(inputPoints_vtk, outputPoints_vtk, movingMesh, nodeName)
        write_vtk(warpedModel, WRITE_PATH + casenames[0] + "_movingMeshFinalRegisteredITK_TPS.vtk")
        return


# # # refinement transform
# # movingLandmarkMesh = itk.transform_mesh_filter(
# #     movingLandmarkMesh, transform=second_transform
# # )

# # # deformable transform
# # movingLandmarkMesh = itk.transform_mesh_filter(
# #     movingLandmarkMesh, transform=final_transform
# # )

# # moving_landmark_points = itk.array_from_vector_container(movingLandmarkMesh.GetPoints())
# # fixed_landmark_points = itk.array_from_vector_container(fixedLandmarkMesh.GetPoints())

# # np.save(
# #     "/data/Apedata/Outputs/" + casename + "_moving_landmark.npy", moving_landmark_points
# # )
# # np.save(
# #     "/data/Apedata/Outputs/" + casename + "_fixed_landmark.npy", fixed_landmark_points
# # )

# # # Get the difference between the landmarks
# # diff = np.square(moving_landmark_points - fixed_landmark_points)
# # diff = np.sqrt(np.sum(diff, 1))
# # np.save("/data/Apedata/Outputs/" + casename + "_diff_landmark.npy", diff)

if __name__ == "__main__":
    print("Testing Module Loaded")
