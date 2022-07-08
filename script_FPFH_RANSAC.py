#!/usr/bin/env python
# coding: utf-8

# In[ ]:

# Import all packages

import numpy as np
from sklearn.preprocessing import scale
import itk
import vtk
import itkwidgets
import math
import matplotlib.pyplot as plt
import sys
from joblib import Parallel, delayed
import time
import copy

from vtk.util import numpy_support
from vtk.util.numpy_support import numpy_to_vtk

# To enable itkwidgets window
# from google.colab import output
# output.enable_custom_widget_manager()


# In[39]:


# Helper Functions

class FPFH(object):
    """Parent class for PFH"""

    def __init__(self, e, div, nneighbors, radius):
        """Pass in parameters """
        self._e = e
        self._div = div
        self._nneighbors = nneighbors
        self._radius = radius

        self._error_list = []
        self._Rlist = []
        self._tlist = []

    def step(self, si, fi):
        """Helper function for calc_pfh_hist. Depends on selection of div
        :si: TODO
        :fi: TODO
        :returns: TODO
        """
        if self._div == 2:
            if fi < si[0]:
                result = 0
            else:
                result = 1
        elif self._div == 3:
            if fi < si[0]:
                result = 0
            elif fi >= si[0] and fi < si[1]:
                result = 1
            else:
                result = 2
        elif self._div == 4:
            if fi < si[0]:
                result = 0
            elif fi >= si[0] and fi < si[1]:
                result = 1
            elif fi >= si[1] and fi < si[2]:
                result = 2
            else:
                result = 3
        elif self._div == 5:
            if fi < si[0]:
                result = 0
            elif fi >= si[0] and fi < si[1]:
                result = 1
            elif fi >= si[1] and fi < si[2]:
                result = 2
            elif fi >= si[2] and fi < si[3]:
                result = 3
            else:
                result = 4
        return result

    def calc_thresholds(self):
        """
        :returns: 3x(div-1) array where each row is a feature's thresholds
        """
        delta = 2. / self._div
        s1 = np.array([-1 + i * delta for i in range(1, self._div)])

        delta = 2. / self._div
        s3 = np.array([-1 + i * delta for i in range(1, self._div)])

        delta = (np.pi) / self._div
        s4 = np.array([-np.pi / 2 + i * delta for i in range(1, self._div)])

        s = np.array([s1, s3, s4])
        return s

    def calc_pfh_hist(self, f):
        """Calculate histogram and bin edges.
        :f: feature vector of f1,f3,f4 (Nx3)
        :returns:
            pfh_hist - array of length div^3, represents number of samples per bin
            bin_edges - range(0, 1, 2, ..., (div^3+1)) 
        """
        # preallocate array sizes, create bin_edges
        pfh_hist, bin_edges = np.zeros(self._div**3), np.arange(
            0, self._div**3 + 1)

        # find the division thresholds for the histogram
        s = self.calc_thresholds()

        # Loop for every row in f from 0 to N
        for j in range(0, f.shape[0]):
            # calculate the bin index to increment
            index = 0
            for i in range(1, 4):
                index += self.step(s[i - 1, :], f[j, i - 1]) * (self._div
                                                                **(i - 1))

            # Increment histogram at that index
            pfh_hist[index] += 1

        return pfh_hist, bin_edges

    def convert_pc_to_matrix(self, pc):
        """Coverts a point cloud to a numpy matrix.
        Inputs:
            pc - a list of 3 by 1 numpy matrices.
        outputs:
            numpy_pc - a 3 by n numpy matrix where each column is a point.
        """
        numpy_pc = np.matrix(np.zeros((3, len(pc))))

        for index, pt in enumerate(pc):
            numpy_pc[0:3, index] = pt

        return numpy_pc

    def convert_matrix_to_pc(self, numpy_pc):
        """Coverts a numpy matrix to a point cloud (useful for plotting).
        Inputs:
            numpy_pc - a 3 by n numpy matrix where each column is a point.
        outputs:
            pc - a list of 3 by 1 numpy matrices.
        """
        pc = []

        for i in range(0, numpy_pc.shape[1]):
            pc.append((numpy_pc[0:3, i]))

        return pc

    def getNeighbors(self, pq, tree):
        """Get k nearest neighbors of the query point pq from pc, within the radius
            :pq: TODO
            :pc: TODO
            :returns: TODO
            """
        k = self._nneighbors
        neighbors = []

        dist, ind = tree.query(pq, k=k + 1)
        dist_flag = dist < self._radius
        ind = ind[dist_flag]
        #print('neigbours are ', ind)
        #print(dist)
        #print(ind)
        #         for i in range(len(pc)):
        #             dist = np.linalg.norm(pq - pc[i])
        #             if dist <= self._radius:  #0.005
        #                 neighbors.append((dist, i))
        #         #print("Found {} neighbors".format(len(neighbors)))
        #         neighbors.sort(key=lambda x: x[0])
        #         neighbors.pop(0)
        neighbors = ind[1:self._nneighbors + 1]
        return neighbors

    def calc_normals(self, pc):
        """TODO: Docstring for calc_normals.
        :pc: TODO
        :returns: TODO
        """
        print("\tCalculating surface normals. \n")
        normals = []
        ind_of_neighbors = []
        N = len(pc)

        #print(pc.shape)
        from sklearn.neighbors import KDTree
        tree = KDTree(pc[:, :, 0], leaf_size=5)

        for i in range(N):
            # Get the indices of neighbors, it is a list of tuples (dist, indx)
            #print('check shape is ', pc[i].shape)
            indN = self.getNeighbors(pc[i].T, tree)  #<- old code
            #indN = list((neigh.kneighbors(pc[i].reshape(1, -1), return_distance=False)).flatten())
            #indN.pop(0)

            # Breakout just the indices
            #indN = [indN[i][1] for i in range(len(indN))]  #<- old code
            ind_of_neighbors.append(indN)
            '''
            # PCA
            X = pc[:, :, 0].T[:, indN]
            #print(X.shape)

            X = X - np.expand_dims(np.mean(X, axis=1), -1)
            if X.shape[1] == 0:
                print(X.shape)
            cov = np.matmul(X, X.T) / (len(indN))
            _, _, Vt = np.linalg.svd(cov)
            normal = Vt[2, :]

            # Re-orient normal vectors
            if np.matmul(normal, -1. * (pc[i])) < 0:
                normal = -1. * normal
            normals.append(normal)
            '''

        return normals, ind_of_neighbors

    def calcHistArray(self, pc, norm, indNeigh):
        """Overriding base PFH to FPFH"""

        print("\tCalculating histograms fast method \n")
        N = len(pc)
        histArray = np.zeros((N, self._div**3))
        distArray = np.zeros((self._nneighbors))
        distList = []
        for i in range(N):
            u = np.asarray(norm[i].T).squeeze()

            features = np.zeros((len(indNeigh[i]), 3))
            for j in range(len(indNeigh[i])):
                pi = pc[i]
                pj = pc[indNeigh[i][j]]
                if np.arccos(np.dot(norm[i], pj - pi)) <= np.arccos(
                        np.dot(norm[j], pi - pj)):
                    ps = pi
                    pt = pj
                    ns = np.asarray(norm[i]).squeeze()
                    nt = np.asarray(norm[indNeigh[i][j]]).squeeze()
                else:
                    ps = pj
                    pt = pi
                    ns = np.asarray(norm[indNeigh[i][j]]).squeeze()
                    nt = np.asarray(norm[i]).squeeze()

                u = ns
                difV = pt - ps
                dist = np.linalg.norm(difV)
                difV = difV / dist
                difV = np.asarray(difV).squeeze()
                v = np.cross(difV, u)
                w = np.cross(u, v)

                alpha = np.dot(v, nt)
                phi = np.dot(u, difV)
                theta = np.arctan(np.dot(w, nt) / np.dot(u, nt))

                features[j, 0] = alpha
                features[j, 1] = phi
                features[j, 2] = theta
                distArray[j] = dist

            distList.append(distArray)
            pfh_hist, bin_edges = self.calc_pfh_hist(features)
            histArray[i, :] = pfh_hist / (len(indNeigh[i]))

        fast_histArray = np.zeros_like(histArray)
        for i in range(N):
            k = len(indNeigh[i])
            for j in range(k):
                spfh_sum = histArray[indNeigh[i][j]] * (1 / distList[i][j])

            fast_histArray[i, :] = histArray[i, :] + (1 / k) * spfh_sum
        #print('checking feature ', len(fast_histArray), fast_histArray[0])
        return fast_histArray

    def findMatches(self, pcS, pcT):
        """Find matches from source to target points
        :pcS: Source point cloud
        :pcT: Target point cloud
        :returns: TODO
        """
        print("...Finding correspondences. \n")
        numS = len(pcS)
        numT = len(pcT)

        print("...Processing source point cloud...\n")
        normS, indS = self.calc_normals(pcS)
        ''' TODO: implement the different histograms '''
        #histS = calcHistArray_naive(pcT, normS, indS, div, nneighbors)
        #histS = calcHistArray_simple(pcT, normS, indS, div, nneighbors)
        histS = self.calcHistArray(pcS, normS, indS)

        print("...Processing target point cloud...\n")
        ''' TODO: implement the different histograms '''
        normT, indT = self.calc_normals(pcT)
        #histT = calcHistArray_naive(pcT, normT, indT, div, nneighbors)
        #histT = calcHistArray_simple(pcT, normT, indT, div, nneighbors)

        histT = self.calcHistArray(pcT, normT, indT)

        distance = []
        dist = []
        for i in range(numS):
            for j in range(numT):
                #appending the l2 norm and j
                dist.append((np.linalg.norm(histS[i] - histT[j]), j))
            dist.sort(
                key=lambda x: x[0])  #To sort by first element of the tuple
            distance.append(dist)
            dist = []
        return distance

def get_euclidean_distance(input_fixedPoints, input_movingPoints):
    mesh_fixed = itk.Mesh[itk.D, 3].New()
    mesh_moving = itk.Mesh[itk.D, 3].New()

    mesh_fixed.SetPoints(itk.vector_container_from_array(
        input_fixedPoints.flatten()))
    mesh_moving.SetPoints(
        itk.vector_container_from_array(input_movingPoints.flatten()))
    
    MetricType = itk.EuclideanDistancePointSetToPointSetMetricv4.PSD3
    metric = MetricType.New()
    metric.SetMovingPointSet(mesh_moving)
    metric.SetFixedPointSet(mesh_fixed)
    metric.Initialize()
    
    return metric.GetValue()

def scale_mesh(input_mesh, scale_factor):
    '''
        Scale the input_mesh by the given scale_factor iso-tropically
    '''
    mesh_points = itk.array_from_vector_container(input_mesh.GetPoints())
    mesh_points = mesh_points * scale_factor
    input_mesh.SetPoints(itk.vector_container_from_array(
        mesh_points.flatten()))
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

# Returns vtk points so that it has normal data also in it
def subsample_points_poisson_polydata(inputMesh, radius):
    import vtk
    from vtk.util import numpy_support
    f = vtk.vtkPoissonDiskSampler()
    f.SetInputData(inputMesh)
    f.SetRadius(radius)
    f.Update()
    sampled_points = f.GetOutput()
    return sampled_points

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


# In[34]:


# File paths

# moving mesh will remain constant
FIXED_MESH_FILE = sys.argv[1]
FIXED_LANDMARK_FILE = sys.argv[2]
MOVING_MESH_FILE = sys.argv[3]
MOVING_LANDMARK_FILE = sys.argv[4]

casename = FIXED_MESH_FILE.split("/")[-1].split(".")[0]
paths = [FIXED_MESH_FILE, MOVING_MESH_FILE]


# Read the landmarks and create mesh from them
fixed_landmark = read_landmarks(FIXED_LANDMARK_FILE)
moving_landmark = read_landmarks(MOVING_LANDMARK_FILE)

fixed_landmark_mesh = itk.Mesh[itk.D, 3].New()
moving_landmark_mesh = itk.Mesh[itk.D, 3].New()

fixed_landmark_mesh.SetPoints(
    itk.vector_container_from_array(fixed_landmark.flatten().astype("float32"))
)
moving_landmark_mesh.SetPoints(
    itk.vector_container_from_array(moving_landmark.flatten().astype("float32"))
)

itk_landmarks = [fixed_landmark_mesh, moving_landmark_mesh]


# Write the meshes in vtk format so that they can be read in ITK
vtk_meshes = list()

for path in paths:
    reader = vtk.vtkPLYReader()
    reader.SetFileName(path)
    reader.Update()

    vtk_mesh = reader.GetOutput()

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

    vtk_meshes.append(vtk_mesh_out_clean)

# Scale mesh before doing moment based initialization
box_filter = vtk.vtkBoundingBox()
box_filter.SetBounds(vtk_meshes[0].GetBounds())
fixedlength = box_filter.GetDiagonalLength()

box_filter = vtk.vtkBoundingBox()
box_filter.SetBounds(vtk_meshes[1].GetBounds())
movinglength = box_filter.GetDiagonalLength()

scale_factor = fixedlength / movinglength

points = vtk_meshes[1].GetPoints()
pointdata = points.GetData()
points_as_numpy = numpy_support.vtk_to_numpy(pointdata)
points_as_numpy = points_as_numpy*scale_factor

vtk_data_array = numpy_support.numpy_to_vtk(num_array=points_as_numpy, deep=True, array_type=vtk.VTK_FLOAT)
points2 = vtk.vtkPoints()
points2.SetData(vtk_data_array)
vtk_meshes[1].SetPoints(points2)

# scale the landmark moving mesh also
itk_landmarks[1] = scale_mesh(itk_landmarks[1], scale_factor)

# Write back out to a filetype supported by ITK
vtk_paths = [path.strip(".ply") + ".vtk" for path in paths]
for idx, mesh in enumerate(vtk_meshes):
    writer = vtk.vtkPolyDataWriter()
    writer.SetInputData(mesh)
    writer.SetFileVersion(42)
    writer.SetFileTypeToBinary()
    writer.SetFileName(vtk_paths[idx])
    writer.Update()

itk_meshes = [itk.meshread(path, pixel_type=itk.D) for path in vtk_paths]


# In[36]:


# Convert meshes to images for performing moment based initialization
#print("Starting moment based initialization")
itk_transformed_meshes = []

itk_images = []
for i, mesh in enumerate(itk_meshes):
    # Make all the points to positive coordinates
    mesh_points = itk.array_from_vector_container(mesh.GetPoints())
    m = np.min(mesh_points, 0)
    mesh_points = mesh_points - m
    mesh.SetPoints(itk.vector_container_from_array(mesh_points.flatten()))

    # Apply same subtraction to landmark points
    landmark_points = itk.array_from_vector_container(itk_landmarks[i].GetPoints())
    landmark_points = landmark_points - m
    itk_landmarks[i].SetPoints(
        itk.vector_container_from_array(landmark_points.flatten())
    )

    itk_transformed_meshes.append(mesh)
    itk_image = itk.triangle_mesh_to_binary_image_filter(
        mesh, origin=[0, 0, 0], spacing=[1, 1, 1], size=[250, 250, 250]
    )
    itk_images.append(itk_image)

itk_transforms = list()

for image in itk_images:
    calculator = itk.ImageMomentsCalculator[type(image)].New()
    calculator.SetImage(image)
    calculator.Compute()
    itk_transforms.append(calculator.GetPhysicalAxesToPrincipalAxesTransform())

# Transform the mesh and landmarks using the moment based initialized transform
itk_transformed_meshes = [
    itk.transform_mesh_filter(mesh, transform=itk_transforms[idx])
    for idx, mesh in enumerate(itk_meshes)
]

itk_transformed_landmarks = [
    itk.transform_mesh_filter(mesh, transform=itk_transforms[idx])
    for idx, mesh in enumerate(itk_landmarks)
]

fixedMesh = itk_transformed_meshes[0]
movingMesh = itk_transformed_meshes[1]

# Write the cleaned Moment Initialized meshes
movingMeshPath = "/data/Apedata/Outputs_RANSAC/" + casename + "_movingMesh.vtk"
fixedMeshPath = "/data/Apedata/Outputs_RANSAC/" + casename + "_fixedMesh.vtk"

w1 = itk.MeshFileWriter[type(fixedMesh)].New()
w1.SetFileName(fixedMeshPath)
w1.SetFileTypeAsBINARY()
w1.SetInput(fixedMesh)
w1.Update()

w1 = itk.MeshFileWriter[type(movingMesh)].New()
w1.SetFileName(movingMeshPath)
w1.SetFileTypeAsBINARY()
w1.SetInput(movingMesh)
w1.Update()

fixedLandmarkMesh = itk_transformed_landmarks[0]
movingLandmarkMesh = itk_transformed_landmarks[1]

# For performing RANSAC in parallel
import vtk
from vtk.util import numpy_support
from vtk.util.numpy_support import numpy_to_vtk

'''
    To obtain the normal for each point from the mesh
'''
def getnormals(inputmesh):
    normals = vtk.vtkTriangleMeshPointNormals()
    normals.SetInputData(inputmesh)
    normals.Update()
    return normals.GetOutput()

def getnormals_pca(movingMesh):
    normals = vtk.vtkPCANormalEstimation()
    normals.SetSampleSize(10)
    normals.SetFlipNormals(True)
    normals.SetNormalOrientationToPoint()
    normals.SetInputData(movingMesh)
    normals.Update()
    as_numpy = numpy_support.vtk_to_numpy(normals.GetOutput().GetPointData().GetArray(0))
    return as_numpy

''' 
    Extracts the normal data from the sampled points
'''
def extract_normal_from_tuple(input_mesh):
    t1 = input_mesh.GetPointData().GetArray("Normals")
    n1_array = []
    for i in range(t1.GetNumberOfTuples()):
        n1_array.append(t1.GetTuple(i))
    n1_array = np.array(n1_array)
    
    points = input_mesh.GetPoints()
    pointdata = points.GetData()
    as_numpy = numpy_support.vtk_to_numpy(pointdata)
    
    return as_numpy, n1_array


"""
    Perform the final iteration of alignment.

    Args:
        fixedPoints, movingPoints, transform_type: 0 or 1 or 2
    Returns:
        (tranformed movingPoints, tranform)
"""
def final_iteration(fixedPoints, movingPoints, transform_type):
    
    mesh_fixed = itk.Mesh[itk.D, 3].New()
    mesh_moving = itk.Mesh[itk.D, 3].New()

    mesh_fixed.SetPoints(itk.vector_container_from_array(fixedPoints.flatten()))
    mesh_moving.SetPoints(itk.vector_container_from_array(movingPoints.flatten()))

    if transform_type == 0:
        TransformType = itk.Euler3DTransform[itk.D]
    elif transform_type == 1:
        TransformType = itk.ScaleVersor3DTransform[itk.D]
    elif transform_type == 2:
        TransformType = itk.Similarity3DTransform[itk.D]

    transform = TransformType.New()
    transform.SetIdentity()

    MetricType = itk.EuclideanDistancePointSetToPointSetMetricv4.PSD3
    # MetricType = itk.PointToPlanePointSetToPointSetMetricv4.PSD3
    metric = MetricType.New()
    metric.SetMovingPointSet(mesh_moving)
    metric.SetFixedPointSet(mesh_fixed)
    metric.SetMovingTransform(transform)
    metric.Initialize()

    # print('Initial Value ', metric.GetValue())
    number_of_epochs = 5000
    optimizer = itk.GradientDescentOptimizerv4Template[itk.D].New()
    optimizer.SetNumberOfIterations(number_of_epochs)
    optimizer.SetLearningRate(0.0001)
    optimizer.SetMinimumConvergenceValue(0.0)
    optimizer.SetConvergenceWindowSize(number_of_epochs)
    optimizer.SetMetric(metric)

    def print_iteration():
        print(
            f"It: {optimizer.GetCurrentIteration()}"
            f" metric value: {optimizer.GetCurrentMetricValue():.6f} "
        )

    #optimizer.AddObserver(itk.IterationEvent(), print_iteration)
    optimizer.StartOptimization()

    print("Distance after Refinement ", metric.GetValue())

    # Get the correct transform and perform the final alignment
    current_transform = metric.GetMovingTransform().GetInverseTransform()
    itk_transformed_mesh = itk.transform_mesh_filter(
        mesh_moving, transform=current_transform
    )

    return (
        itk.array_from_vector_container(itk_transformed_mesh.GetPoints()),
        current_transform,
    )

# RANSAC with VTK Landmark Transform
def ransac_icp_parallel_vtk(movingMeshPoints, fixedMeshPoints,
                        number_of_iterations, mesh_sub_sample_points,
                        number_of_ransac_points, transform_type,
                        inlier_value):
    '''
        Perform Ransac by doing parallel iterations for different samples.
    '''
    import numpy as np
    np.random.seed(0)

    all_points1 = movingMeshPoints
    all_points2 = fixedMeshPoints

    def process(i, mesh_sub_sample_points, number_of_ransac_points,
                return_result, inlier_points=None):
        '''
        Args:
            i: input seed used to sample the points.
            number_of_ransac_points: Number of random points selected to perform the registration.
            mesh_sub_sample_points: Number of points used to calculate the Euclidean distance for entire mesh.
            return_result: Whether to return the transformed mesh.
        
        Returns: (Best Value, seed) or the transformed_points depending on return_result flag.
            
        '''

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


            m = landmarkTransform.GetMatrix()
            tp = vtkmatrix_to_numpy(m)

            itks = itk.Similarity3DTransform.New()
            itks.Translate(list(tp[:3, 3]))
            itks.SetMatrix(itk.matrix_from_array(tp[:3, :3]))

            return itks
        
        ''' Create small point sets '''
        np.random.seed(i)
        
        if inlier_points is None:
            random_indices = np.random.choice(all_points1.shape[0],
                                              size=number_of_ransac_points)
            A_corr_temp = all_points1[random_indices, :]
            B_corr_temp = all_points2[random_indices, :]
        else:
            A_corr_temp = all_points1[inlier_points, :]
            B_corr_temp = all_points2[inlier_points, :]
        
        
        ''' Perform vtkLandMarkTransform based registration using selected points '''
        source_points = vtk.vtkPoints()
        source_points.SetData(numpy_support.numpy_to_vtk(A_corr_temp, deep=True))

        target_points = vtk.vtkPoints()
        target_points.SetData(numpy_support.numpy_to_vtk(B_corr_temp, deep=True))

        landmarkTransform = vtk.vtkLandmarkTransform()
        landmarkTransform.SetModeToSimilarity()
        #landmarkTransform.SetModeToRigidBody()
        landmarkTransform.SetSourceLandmarks(source_points)
        landmarkTransform.SetTargetLandmarks(target_points)
        landmarkTransform.Update()
        
        
        ''' For converting VTK Transform to ITK Transform '''
        current_transform = itk_transform_from_vtk(landmarkTransform)
        
        
        ''' For transforming the pointset using the obtained transform '''
        temp_pointset = itk.Mesh[itk.D, 3].New()
        temp_pointset.SetPoints(itk.vector_container_from_array(all_points1.flatten().astype('float32')))

        new_pointset = itk.transform_mesh_filter(temp_pointset, transform=current_transform)
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
#     results = []
#     for i in range(number_of_iterations):
#         results.append(process(i, mesh_sub_sample_points, number_of_ransac_points, 0))
    
    results = Parallel(n_jobs=-1)(
        delayed(process)(i, mesh_sub_sample_points, number_of_ransac_points, 0)
        for i in range(number_of_iterations))
    
    results_values =  []
    for k in results:
        results_values.append(k[0])
    
    # Sort the results and get the best one i.e. the lowest one
    index = np.argmax(results_values)
    value = results_values[index]
    
    final_result = process(index, mesh_sub_sample_points,
                          number_of_ransac_points, 1, results[index][1])
    
    return final_result, index, value

def orient_points(input_points, x, y, z):
    '''
        Orients the input points based on the x, y and z orientations given.
    '''
    output_points = copy.deepcopy(input_points)
    output_points[:, 0] = output_points[:, 0]*x
    output_points[:, 1] = output_points[:, 1]*y
    output_points[:, 2] = output_points[:, 2]*z

    return output_points

def orient_points_in_mesh(input_mesh, x, y, z):
    input_points = itk.array_from_vector_container(input_mesh.GetPoints())
    oriented_points = orient_points(input_points, x, y, z)
    input_mesh.SetPoints(itk.vector_container_from_array(oriented_points.flatten()))
    return

#print("Starting Ransac")
import time

number_of_iterations = 50000
number_of_ransac_points = 250
inlier_value = 25
transform_type = 0

movingMesh_vtk = readvtk(movingMeshPath)
fixedMesh_vtk = readvtk(fixedMeshPath)

# Sub-Sample the points for rigid refinement and deformable registration
# radius = 5.5 for gorilla
# radius = 4.5 for Pan
# radius = 4 for Pongo
movingMeshPoints = subsample_points_poisson(movingMesh_vtk, radius=5.5)
fixedMeshPoints  = subsample_points_poisson(fixedMesh_vtk, radius=5.5)


#vtk_points_to_numpy(movingMeshPoints_vtk)
# Orient the points
best_value = 5000
best_points = None
best_orientation = [1, 1, 1]
# Loop through all the combinations of orientations
for x in [-1, 1]:
    for y in [-1, 1]:
        for z in [-1, 1]:
            oriented_points = orient_points(movingMeshPoints, x, y, z)
            #print(x, y, z)
            value = get_euclidean_distance(fixedMeshPoints, oriented_points)
            if value < best_value:
                best_points = oriented_points
                best_value = value
                best_orientation = [x, y, z]

# Orient the sub-sampled points and landmark points for the moving mesh
movingMeshPoints = orient_points(movingMeshPoints, best_orientation[0], best_orientation[1], best_orientation[2])
orient_points_in_mesh(movingLandmarkMesh, best_orientation[0], best_orientation[1], best_orientation[2])
orient_points_in_mesh(movingMesh, best_orientation[0], best_orientation[1], best_orientation[2])

w1 = itk.MeshFileWriter[type(movingMesh)].New()
w1.SetFileName(movingMeshPath.replace('movingMesh', 'movingMeshOriented'))
w1.SetFileTypeAsBINARY()
w1.SetInput(movingMesh)
w1.Update()

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

movingMeshPoints = numpy_to_vtk_polydata(movingMeshPoints)
fixedMeshPoints = numpy_to_vtk_polydata(fixedMeshPoints)

movingMeshPointNormals = getnormals_pca(movingMeshPoints)
fixedMeshPointNormals = getnormals_pca(fixedMeshPoints)

movingMeshPoints = vtk_points_to_numpy(movingMeshPoints)
fixedMeshPoints = vtk_points_to_numpy(fixedMeshPoints)

# Obtain normals from the sub-sampled points
#movingMeshPoints, movingMeshPointNormals = extract_normal_from_tuple(movingMeshPoints)
#fixedMeshPoints, fixedMeshPointNormals = extract_normal_from_tuple(fixedMeshPoints)


A_xyz = fixedMeshPoints.T
B_xyz = movingMeshPoints.T

print("A_xyz ", A_xyz.shape)
print("B_xyz ", B_xyz.shape)

np.save("/data/Apedata/Outputs_RANSAC/" + casename + '_fixedMesh_landmarks.npy', itk.array_from_vector_container(fixedLandmarkMesh.GetPoints()))
np.save("/data/Apedata/Outputs_RANSAC/" + casename + '_movingMesh_landmarks.npy', itk.array_from_vector_container(movingLandmarkMesh.GetPoints()))

np.save("/data/Apedata/Outputs_RANSAC/" + casename + '_movingMeshPoints.npy', movingMeshPoints)
np.save("/data/Apedata/Outputs_RANSAC/" + casename + '_fixedMeshPoints.npy', fixedMeshPoints)

# Extract FPFH feature
import open3d as o3d

def extract_open3d_fpfh(pcd, voxel_size):
    radius_normal = voxel_size * 2
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature,
                                             max_nn=100))
    return np.array(fpfh.data).T

if True:
    et = 0.1
    div = 3
    nneighbors = 25 # (voxel_size * 2)
    rad = 35        # (voxel_size * 5)

    fpfh = FPFH(et, div, nneighbors, rad)
    pcS = np.expand_dims(A_xyz.T, -1)
    _, indS = fpfh.calc_normals(pcS)
    normS = fixedMeshPointNormals
    A_feats = fpfh.calcHistArray(A_xyz.T, normS, indS)

    fpfh = FPFH(et, div, nneighbors, rad)
    pcS = np.expand_dims(B_xyz.T, -1)
    _, indS = fpfh.calc_normals(pcS)
    normS = movingMeshPointNormals
    B_feats = fpfh.calcHistArray(B_xyz.T, normS, indS)
else:
    VOXEL_SIZE = 7
    A_pcd_raw = o3d.geometry.PointCloud()
    A_pcd_raw.points = o3d.utility.Vector3dVector(fixedMeshPoints)
    B_pcd_raw = o3d.geometry.PointCloud()
    B_pcd_raw.points = o3d.utility.Vector3dVector(movingMeshPoints)
    A_pcd = A_pcd_raw
    B_pcd = B_pcd_raw
    A_feats = extract_open3d_fpfh(A_pcd, VOXEL_SIZE)
    B_feats = extract_open3d_fpfh(B_pcd, VOXEL_SIZE)

print(A_feats.shape, B_feats.shape)

# Establish correspondences by nearest neighbour search in feature space
from scipy.spatial import cKDTree

def find_knn_cpu(feat0, feat1, knn=1, return_distance=False):
    feat1tree = cKDTree(feat1)
    dists, nn_inds = feat1tree.query(feat0, k=knn, n_jobs=-1)
    if return_distance:
        return nn_inds, dists
    else:
        return nn_inds


def find_correspondences(feats0, feats1, mutual_filter=True):
    nns01 = find_knn_cpu(feats0, feats1, knn=1, return_distance=False)
    corres01_idx0 = np.arange(len(nns01))
    corres01_idx1 = nns01

    if not mutual_filter:
        return corres01_idx0, corres01_idx1

    nns10 = find_knn_cpu(feats1, feats0, knn=1, return_distance=False)
    corres10_idx1 = np.arange(len(nns10))
    corres10_idx0 = nns10

    mutual_filter = (corres10_idx0[corres01_idx1] == corres01_idx0)
    corres_idx0 = corres01_idx0[mutual_filter]
    corres_idx1 = corres01_idx1[mutual_filter]

    return corres_idx0, corres_idx1


corrs_A, corrs_B = find_correspondences(A_feats, B_feats, mutual_filter=True)

A_corr = A_xyz[:, corrs_A]  # np array of size 3 by num_corrs
B_corr = B_xyz[:, corrs_B]  # np array of size 3 by num_corrs

num_corrs = A_corr.shape[1]
print(f'FPFH generates {num_corrs} putative correspondences.')


# Perform Initial alignment using Ransac parallel iterations
transform_matrix, index, value = ransac_icp_parallel_vtk(movingMeshPoints = A_corr.T, 
                                                    fixedMeshPoints = B_corr.T,
                                                    number_of_iterations = 50000,
                                                    mesh_sub_sample_points = 500,
                                                    number_of_ransac_points = 250, 
                                                    transform_type = 3,
                                                    inlier_value = 35)

print('Best Combination ', index, value)
transform_matrix = itk.transform_from_dict(transform_matrix)


#movingMesh  = itk.meshread(fixedMeshPath)
fixedMesh_RANSAC = itk.transform_mesh_filter(fixedMesh, transform=transform_matrix)

w1 = itk.MeshFileWriter[type(fixedMesh)].New()
w1.SetFileName("/data/Apedata/Outputs_RANSAC/" + casename + "_fixedMeshRANSAC.vtk")
w1.SetFileTypeAsBINARY()
w1.SetInput(fixedMesh_RANSAC)
w1.Update()

exit(0)
# #print("Starting Rigid Refinement")
# # Perform final alignment using the Euler3DTransform
# transform_type = 0
# final_mesh_points, second_transform = final_iteration(
#     fixedMeshPoints, itk_transformed_points, transform_type
# )

# np.save('fixedMeshPoints.npy', fixedMeshPoints)
# np.save('final_mesh_points.npy', final_mesh_points)

# # Write the sub-sampled moving mesh points
# rigidRegisteredPoints = itk.Mesh.D3.New()
# rigidRegisteredPoints.SetPoints(itk.vector_container_from_array(final_mesh_points.flatten()))

# w1 = itk.MeshFileWriter[type(rigidRegisteredPoints)].New()
# w1.SetFileName("/data/Apedata/Outputs_RANSAC/" + casename + "_rigidRegisteredPoints.vtk")
# w1.SetFileTypeAsBINARY()
# w1.SetInput(rigidRegisteredPoints)
# w1.Update()

# # Transform the full mesh and write the output
# mesh_moving = itk.meshread(movingMeshPath, itk.D)

# orient_points_in_mesh(mesh_moving, best_orientation[0], best_orientation[1], best_orientation[2])
# mesh_moving = itk.transform_mesh_filter(mesh_moving, transform=first_transform)
# mesh_moving = itk.transform_mesh_filter(mesh_moving, transform=second_transform)

# w1 = itk.MeshFileWriter[type(mesh_moving)].New()
# w1.SetFileName("/data/Apedata/Outputs_RANSAC/" + casename + "_movingMeshRigidRegistered.vtk")
# w1.SetFileTypeAsBINARY()
# w1.SetInput(mesh_moving)
# w1.Update()

#print("Completed Rigid Refinement")

#exit(0)
# In[112]:


# # [STAR] Expectation Based PointSetToPointSetMetricv4 Registration

# import copy
# from vtk.util import numpy_support

# imageDiagonal = 100

# PixelType = itk.D
# Dimension = 3

# FixedImageType = itk.Image[PixelType, Dimension]

# # Create PointSets for registration
# movingPS = itk.PointSet[itk.D, Dimension].New()
# fixedPS = itk.PointSet[itk.D, Dimension].New()

# movingPS.SetPoints(itk.vector_container_from_array(final_mesh_points.flatten()))
# fixedPS.SetPoints(itk.vector_container_from_array(fixedMeshPoints.flatten()))


# # For getting the Bounding Box
# ElementIdentifierType = itk.UL
# CoordType = itk.F
# Dimension = 3

# VecContType = itk.VectorContainer[
#     ElementIdentifierType, itk.Point[CoordType, Dimension]
# ]
# bounding_box = itk.BoundingBox[
#     ElementIdentifierType, Dimension, CoordType, VecContType
# ].New()

# bounding_box.SetPoints(movingPS.GetPoints())
# bounding_box.ComputeBoundingBox()

# minBounds = np.array(bounding_box.GetMinimum())
# maxBounds = np.array(bounding_box.GetMaximum())

# spacing = np.sqrt(bounding_box.GetDiagonalLength2()) / imageDiagonal
# diff = maxBounds - minBounds

# # print('Spacing ', spacing)
# # print('minBounds ', minBounds)
# # print('maxBounds ', maxBounds)

# fixedImageSize = [0] * 3
# fixedImageSize[0] = math.ceil(1.25 * diff[0] / spacing)
# fixedImageSize[1] = math.ceil(1.25 * diff[1] / spacing)
# fixedImageSize[2] = math.ceil(1.25 * diff[2] / spacing)

# fixedImageOrigin = [0] * 3
# fixedImageOrigin[0] = minBounds[0] - 0.25 * diff[0]
# fixedImageOrigin[1] = minBounds[1] - 0.25 * diff[1]
# fixedImageOrigin[2] = minBounds[2] - 0.25 * diff[2]

# fixedImageSpacing = np.ones(3) * spacing
# fixedImageDirection = np.identity(3)

# fixedImage = FixedImageType.New()
# fixedImage.SetRegions(fixedImageSize)
# fixedImage.SetOrigin(fixedImageOrigin)
# fixedImage.SetDirection(fixedImageDirection)
# fixedImage.SetSpacing(fixedImageSpacing)
# fixedImage.Allocate()


# # Create BSpline Transformation object and initialize the parameters
# SplineOrder = 3
# TransformType = itk.BSplineTransform[itk.D, Dimension, SplineOrder]
# InitializerType = itk.BSplineTransformInitializer[TransformType, FixedImageType]

# transform = TransformType.New()

# numberOfGridNodesInOneDimension = 5
# transformInitializer = InitializerType.New()
# transformInitializer.SetTransform(transform)
# transformInitializer.SetImage(fixedImage)
# transformInitializer.SetTransformDomainMeshSize(
#     numberOfGridNodesInOneDimension - SplineOrder
# )
# transformInitializer.InitializeTransform()

# # Registration Loop
# numOfIterations = 10000
# maxStep = 0.1
# learningRate = 0.1

# # Good combinations
# # 10000
# # sigma: 3, Kneighbourhood 20, bspline: 4 -> 3.72
# # sigma: 3, Kneighbourhood 20, bspline: 8 -> 3.77
# # sigma: 3, Kneighbourhood 20, bspline: 6 -> 3.42
# # sigma: 3, Kneighbourhood 20, bspline: 5 -> 3.70 -> best (by qualitative comparison)
# # 
# MetricType = itk.ExpectationBasedPointSetToPointSetMetricv4[type(movingPS)]
# metric = MetricType.New()
# metric.SetFixedPointSet(movingPS)
# metric.SetMovingPointSet(fixedPS)
# metric.SetPointSetSigma(3)
# metric.SetEvaluationKNeighborhood(10)
# metric.SetMovingTransform(transform)
# metric.Initialize()

# # print('Metric Created')

# optimizer = itk.RegularStepGradientDescentOptimizerv4.D.New()
# optimizer.SetNumberOfIterations(numOfIterations)
# optimizer.SetMaximumStepSizeInPhysicalUnits(maxStep)
# optimizer.SetLearningRate(learningRate)
# optimizer.SetMinimumConvergenceValue(-100)
# optimizer.SetConvergenceWindowSize(numOfIterations)
# optimizer.SetMetric(metric)


# def iteration_update():
#     if optimizer.GetCurrentIteration() % 100 == 0:
#         print(
#             f"It: {optimizer.GetCurrentIteration()}"
#             f" metric value: {optimizer.GetCurrentMetricValue():.6f} "
#         )
#     return


# iteration_command = itk.PyCommand.New()
# iteration_command.SetCommandCallable(iteration_update)
# optimizer.AddObserver(itk.IterationEvent(), iteration_command)

# optimizer.StartOptimization()

# # Transform the point set using the final transform
# final_transform = metric.GetMovingTransform()

# e_metric = itk.EuclideanDistancePointSetToPointSetMetricv4.PSD3.New()
# e_metric.SetFixedPointSet(fixedPS)
# e_metric.SetMovingPointSet(movingPS)
# print("Euclidean Metric Before TSD Deformable Registration ", e_metric.GetValue())

# movingPSNew = itk.PointSet[itk.D, 3].New()
# numberOfPoints = movingPS.GetNumberOfPoints()

# for n in range(0, numberOfPoints):
#     movingPSNew.SetPoint(n, final_transform.TransformPoint(movingPS.GetPoint(n)))

# e_metric = itk.EuclideanDistancePointSetToPointSetMetricv4.PSD3.New()
# e_metric.SetFixedPointSet(fixedPS)
# e_metric.SetMovingPointSet(movingPSNew)
# print("Euclidean Metric After TSD Deformable Registration ", e_metric.GetValue())


# # Write the Displacement Field
# write_displacement_field = False
# if write_displacement_field:
#     convertFilter = itk.TransformToDisplacementFieldFilter.IVF33D.New()
#     convertFilter.SetTransform(final_transform)
#     convertFilter.UseReferenceImageOn()
#     convertFilter.SetReferenceImage(fixedImage)
#     convertFilter.Update()
#     field = convertFilter.GetOutput()
#     field = np.array(field)
#     np.save("displacement_field.npy", field)


# # Write the final registered mesh
# movingMeshPath = "/data/Apedata/Outputs/" + casename + "_movingMeshRigidRegistered.vtk"
# movingMesh = itk.meshread(movingMeshPath)

# movingMesh = itk.transform_mesh_filter(movingMesh, transform=final_transform)

# w1 = itk.MeshFileWriter[type(movingMesh)].New()
# w1.SetFileName("/data/Apedata/Outputs/" + casename + "_movingMeshFinalRegistered.vtk")
# w1.SetFileTypeAsBINARY()
# w1.SetInput(movingMesh)
# w1.Update()

# # Write the fixed mesh also
# fixedMesh = itk.meshread('fixedMesh.vtk')
# w1 = itk.MeshFileWriter[type(fixedMesh)].New()
# w1.SetFileName("/data/Apedata/Outputs/" + casename + "_fixedMesh.vtk")
# w1.SetFileTypeAsBINARY()
# w1.SetInput(fixedMesh)
# w1.Update()


# print("Calculating distance between landmarks")
# # ransac transform
# movingLandmarkMesh = itk.transform_mesh_filter(
#     movingLandmarkMesh, transform=first_transform
# )

# # refinement transform
# movingLandmarkMesh = itk.transform_mesh_filter(
#     movingLandmarkMesh, transform=second_transform
# )

# # deformable transform
# movingLandmarkMesh = itk.transform_mesh_filter(
#     movingLandmarkMesh, transform=final_transform
# )

# moving_landmark_points = itk.array_from_vector_container(movingLandmarkMesh.GetPoints())
# fixed_landmark_points = itk.array_from_vector_container(fixedLandmarkMesh.GetPoints())

# np.save(
#     "/data/Apedata/Outputs/" + casename + "_moving_landmark.npy", moving_landmark_points
# )
# np.save(
#     "/data/Apedata/Outputs/" + casename + "_fixed_landmark.npy", fixed_landmark_points
# )

# # Get the difference between the landmarks
# diff = np.square(moving_landmark_points - fixed_landmark_points)
# diff = np.sqrt(np.sum(diff, 1))
# np.save("/data/Apedata/Outputs/" + casename + "_diff_landmark.npy", diff)
