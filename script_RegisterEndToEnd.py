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
print("Starting moment based initialization")
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

w1 = itk.MeshFileWriter[type(fixedMesh)].New()
w1.SetFileName("fixedMesh.vtk")
w1.SetFileTypeAsBINARY()
w1.SetInput(fixedMesh)
w1.Update()

w1 = itk.MeshFileWriter[type(movingMesh)].New()
w1.SetFileName("movingMesh.vtk")
w1.SetFileTypeAsBINARY()
w1.SetInput(movingMesh)
w1.Update()

fixedLandmarkMesh = itk_transformed_landmarks[0]
movingLandmarkMesh = itk_transformed_landmarks[1]

np.save('fixedMesh_landmarks.npy', itk.array_from_vector_container(fixedLandmarkMesh.GetPoints()))
np.save('movingMesh_landmarks.npy', itk.array_from_vector_container(movingLandmarkMesh.GetPoints()))

print("Completed moment based initialization")

exit(0)
# For performing RANSAC in parallel

from vtk.util import numpy_support
from vtk.util.numpy_support import numpy_to_vtk


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
    number_of_epochs = 10000
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

    # optimizer.AddObserver(itk.IterationEvent(), print_iteration)
    optimizer.StartOptimization()

    print("Final Value ", metric.GetValue())

    # Get the correct transform and perform the final alignment
    current_transform = metric.GetMovingTransform().GetInverseTransform()
    itk_transformed_mesh = itk.transform_mesh_filter(
        mesh_moving, transform=current_transform
    )

    return (
        itk.array_from_vector_container(itk_transformed_mesh.GetPoints()),
        current_transform,
    )


def ransac_icp_parallel(
    fixedMeshPoints,
    movingMeshPoints,
    number_of_iterations,
    mesh_sub_sample_points,
    number_of_ransac_points,
    transform_type,
    convergance_value,
):
    """
        Perform Ransac by doing parallel iterations for different samples.
    """
    import numpy as np

    np.random.seed(0)

    all_points1 = fixedMeshPoints
    all_points2 = movingMeshPoints

    def process(i, mesh_sub_sample_points, number_of_ransac_points, return_result):
        """
        Args:
            i: input seed used to sample the points.
            number_of_ransac_points: Number of random points selected to perform the registration.
            mesh_sub_sample_points: Number of points used to calculate the Euclidean distance for entire mesh.
            return_result: Whether to return the transformed mesh.
        
        Returns: (Best Value, seed) or the transformed_points depending on return_result flag.
            
        """

        # Create Mesh inside the method to tackle the problem of serialization

        import itk

        mesh_fixed = itk.Mesh[itk.D, 3].New()
        mesh_moving = itk.Mesh[itk.D, 3].New()

        mesh_fixed.SetPoints(itk.vector_container_from_array(all_points1.flatten()))
        mesh_moving.SetPoints(itk.vector_container_from_array(all_points2.flatten()))

        ps1 = itk.Mesh[itk.D, 3].New()
        ps2 = itk.Mesh[itk.D, 3].New()

        # Create small point sets
        np.random.seed(i)
        random_indices = np.random.choice(
            all_points1.shape[0], size=number_of_ransac_points
        )
        p1_a = all_points1[random_indices, :]

        random_indices = np.random.choice(
            all_points2.shape[0], size=number_of_ransac_points
        )
        p2_a = all_points2[random_indices, :]

        ps1.SetPoints(itk.vector_container_from_array(p1_a.flatten()))
        ps2.SetPoints(itk.vector_container_from_array(p2_a.flatten()))

        # Similarity3DTransform works best when doing inter-species registration
        if transform_type == 0:
            TransformType = itk.Euler3DTransform[itk.D]
        elif transform_type == 1:
            TransformType = itk.ScaleVersor3DTransform[itk.D]
        elif transform_type == 2:
            TransformType = itk.Similarity3DTransform[itk.D]

        transform = TransformType.New()
        transform.SetIdentity()

        MetricType = itk.EuclideanDistancePointSetToPointSetMetricv4.PSD3
        metric = MetricType.New()
        metric.SetMovingPointSet(ps2)
        metric.SetFixedPointSet(ps1)
        metric.SetMovingTransform(transform)
        metric.Initialize()

        optimizer = itk.ConjugateGradientLineSearchOptimizerv4Template[itk.D].New()
        optimizer.SetNumberOfIterations(20)
        optimizer.SetMaximumStepSizeInPhysicalUnits(0.1)
        optimizer.SetMinimumConvergenceValue(0.0)
        optimizer.SetConvergenceWindowSize(20)
        optimizer.SetMetric(metric)
        optimizer.StartOptimization()

        #         optimizer = itk.GradientDescentOptimizerv4Template.D.New()
        #         optimizer.SetNumberOfIterations(200)
        #         optimizer.SetLearningRate(0.0001)
        #         #optimizer.SetMaximumStepSizeInPhysicalUnits( 0.1 )
        #         optimizer.SetMinimumConvergenceValue( 0.0 )
        #         optimizer.SetConvergenceWindowSize( 200 )
        #         optimizer.SetMetric(metric)
        #         optimizer.StartOptimization()

        current_transform = metric.GetTransform()
        itk_transformed_mesh = itk.transform_mesh_filter(
            mesh_moving, transform=current_transform
        )

        e_metric = itk.EuclideanDistancePointSetToPointSetMetricv4.PSD3.New()
        e_metric.SetFixedPointSet(mesh_fixed)
        e_metric.SetMovingPointSet(itk_transformed_mesh)
        e_metric.Initialize()

        current_value = e_metric.GetValue()

        if return_result:
            # Create a mesh using the moving points and transform it using the best transform
            mesh_moving = itk.Mesh[itk.D, 3].New()
            mesh_moving.SetPoints(
                itk.vector_container_from_array(movingMeshPoints.flatten())
            )

            itk_transformed_mesh = itk.transform_mesh_filter(
                mesh_moving, transform=current_transform
            )

            return itk.array_from_vector_container(itk_transformed_mesh.GetPoints())
        else:
            # To make the transform data serializable
            current_transform = itk.dict_from_transform(current_transform)
            return (current_value, i, current_transform)

    # Spawn multiple jobs to utilize all cores
    results = Parallel(n_jobs=8)(
        delayed(process)(i, mesh_sub_sample_points, number_of_ransac_points, 0)
        for i in range(number_of_iterations)
    )

    # Sort the results and get the best one i.e. the lowest one
    results = sorted(results)

    print(results[0][0], results[0][1])
    final_result = process(
        results[0][1], mesh_sub_sample_points, number_of_ransac_points, 1
    )

    return final_result, results[0]


print("Starting Ransac")
import time

number_of_iterations = 10000
number_of_ransac_points = 250
mesh_sub_sample_points = 5000
convergence_value = 3
transform_type = 0

movingMeshPath = "movingMesh.vtk"
fixedMeshPath = "fixedMesh.vtk"

movingMesh_vtk = readvtk(movingMeshPath)
fixedMesh_vtk = readvtk(fixedMeshPath)

movingMeshAllPoints = numpy_support.vtk_to_numpy(movingMesh_vtk.GetPoints().GetData())

# Sub-Sample the points for ransac
movingMeshPoints_ransac = subsample_points_poisson(movingMesh_vtk, radius=11.5)
fixedMeshPoints_ransac = subsample_points_poisson(fixedMesh_vtk, radius=11.5)

# Sub-Sample the points for rigid refinement and deformable registration
movingMeshPoints = subsample_points_poisson(movingMesh_vtk, radius=5.5)
fixedMeshPoints = subsample_points_poisson(fixedMesh_vtk, radius=5.5)

print(movingMeshPoints.shape, fixedMeshPoints.shape)
print(movingMeshPoints_ransac.shape, fixedMeshPoints_ransac.shape)


def orient_points(input_points, x, y, z):
    '''
        Orients the input points based on the x, y and z orientations given.
    '''
    output_points = copy.deepcopy(input_points)
    output_points[:, 0] = output_points[:, 0]*x
    output_points[:, 1] = output_points[:, 1]*y
    output_points[:, 2] = output_points[:, 2]*z

    return output_points

# Loop through all the combinations of orientations
# for x in [-1, 1]:
#     for y in [-1, 1]:
#         for z in [-1, 1]:
#             oriented_points = orient_points(movingMeshPoints_ransac)
# w1 = itk.MeshFileWriter[type(scaledMovingMesh)].New()
# w1.SetFileName("movingMesh_scaled.vtk")
# w1.SetFileTypeAsBINARY()
# w1.SetInput(scaledMovingMesh)
# w1.Update()


# Perform Initial alignment using Ransac parallel iterations
start_time = time.time()
transform_type = 2

itk_transformed_points, transform_matrix = ransac_icp_parallel(
    fixedMeshPoints_ransac,
    movingMeshPoints_ransac,
    number_of_iterations,
    mesh_sub_sample_points,
    number_of_ransac_points,
    transform_type,
    convergence_value,
)
end_time = time.time()

print(end_time - start_time)
# print('itk_transformed_points shape ', itk_transformed_points.shape)

print("Completed Ransac")

# For taking care of a bug in the code
first_transform = transform_matrix[2]
first_transform[0]["transformType"] = "D"
first_transform = itk.transform_from_dict(first_transform)

print("Starting Rigid Refinement")
# Perform final alignment using the Euler3DTransform
transform_type = 0
final_mesh, second_transform = final_iteration(
    fixedMeshPoints, itk_transformed_points, transform_type
)

# Write the sub-sampled moving mesh points
rigidRegisteredPoints = itk.Mesh.D3.New()
rigidRegisteredPoints.SetPoints(itk.vector_container_from_array(final_mesh.flatten()))

w1 = itk.MeshFileWriter[type(rigidRegisteredPoints)].New()
w1.SetFileName("/data/Apedata/Outputs/" + casename + "_rigidRegisteredPoints.vtk")
w1.SetFileTypeAsBINARY()
w1.SetInput(rigidRegisteredPoints)
w1.Update()

# Transform the full mesh and write the output
mesh_moving = itk.meshread(movingMeshPath, itk.D)

mesh_moving = itk.transform_mesh_filter(mesh_moving, transform=first_transform)

mesh_moving = itk.transform_mesh_filter(mesh_moving, transform=second_transform)

w1 = itk.MeshFileWriter[type(mesh_moving)].New()
w1.SetFileName("/data/Apedata/Outputs/" + casename + "_movingMeshRigidRegistered.vtk")
w1.SetFileTypeAsBINARY()
w1.SetInput(mesh_moving)
w1.Update()

print("Completed Rigid Refinement")

exit(0)
# In[112]:


# [STAR] Expectation Based PointSetToPointSetMetricv4 Registration

import copy
from vtk.util import numpy_support

imageDiagonal = 100

PixelType = itk.D
Dimension = 3

FixedImageType = itk.Image[PixelType, Dimension]

# Create PointSets for registration
movingPS = itk.PointSet[itk.D, Dimension].New()
fixedPS = itk.PointSet[itk.D, Dimension].New()

movingPS.SetPoints(itk.vector_container_from_array(final_mesh.flatten()))
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

# print('Spacing ', spacing)
# print('minBounds ', minBounds)
# print('maxBounds ', maxBounds)

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

numberOfGridNodesInOneDimension = 8
transformInitializer = InitializerType.New()
transformInitializer.SetTransform(transform)
transformInitializer.SetImage(fixedImage)
transformInitializer.SetTransformDomainMeshSize(
    numberOfGridNodesInOneDimension - SplineOrder
)
transformInitializer.InitializeTransform()

# Registration Loop
numOfIterations = 10000
maxStep = 0.1
learningRate = 0.1

MetricType = itk.ExpectationBasedPointSetToPointSetMetricv4[type(movingPS)]
metric = MetricType.New()
metric.SetFixedPointSet(movingPS)
metric.SetMovingPointSet(fixedPS)
metric.SetPointSetSigma(2.5)
metric.SetEvaluationKNeighborhood(10)
metric.SetMovingTransform(transform)
metric.Initialize()

# print('Metric Created')

optimizer = itk.RegularStepGradientDescentOptimizerv4.D.New()
optimizer.SetNumberOfIterations(numOfIterations)
optimizer.SetMaximumStepSizeInPhysicalUnits(maxStep)
optimizer.SetLearningRate(learningRate)
optimizer.SetMinimumConvergenceValue(-100)
optimizer.SetConvergenceWindowSize(numOfIterations)
optimizer.SetMetric(metric)


def iteration_update():
    print(
        f"It: {optimizer.GetCurrentIteration()}"
        f" metric value: {optimizer.GetCurrentMetricValue():.6f} "
    )


iteration_command = itk.PyCommand.New()
iteration_command.SetCommandCallable(iteration_update)
# optimizer.AddObserver(itk.IterationEvent(), iteration_command)

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
convertFilter = itk.TransformToDisplacementFieldFilter.IVF33D.New()
convertFilter.SetTransform(final_transform)
convertFilter.UseReferenceImageOn()
convertFilter.SetReferenceImage(fixedImage)
convertFilter.Update()
field = convertFilter.GetOutput()
field = np.array(field)
np.save("displacement_field.npy", field)


# Write the final registered mesh
movingMeshPath = "movingMeshRigidRegistered.vtk"
movingMesh = itk.meshread(movingMeshPath)

movingMesh = itk.transform_mesh_filter(movingMesh, transform=final_transform)

w1 = itk.MeshFileWriter[type(movingMesh)].New()
w1.SetFileName("/data/Apedata/Outputs/" + casename + "_movingMeshFinalRegistered.vtk")
w1.SetFileTypeAsBINARY()
w1.SetInput(movingMesh)
w1.Update()


print("Calculating distance between landmarks")
movingLandmarkMesh = itk.transform_mesh_filter(
    movingLandmarkMesh, transform=first_transform
)

movingLandmarkMesh = itk.transform_mesh_filter(
    movingLandmarkMesh, transform=second_transform
)

movingLandmarkMesh = itk.transform_mesh_filter(
    movingLandmarkMesh, transform=final_transform
)

moving_landmark_points = itk.array_from_vector_container(movingLandmarkMesh.GetPoints())
fixed_landmark_points = itk.array_from_vector_container(fixedLandmarkMesh.GetPoints())

np.save(
    "/data/Apedata/Outputs/" + casename + "_moving_landmark.npy", moving_landmark_points
)
np.save(
    "/data/Apedata/Outputs/" + casename + "_fixed_landmark.npy", fixed_landmark_points
)

# Get the difference between the landmarks
diff = np.square(moving_landmark_points - fixed_landmark_points)
diff = np.sqrt(np.sum(diff, 1))
np.save("/data/Apedata/Outputs/" + casename + "_diff_landmark.npy", diff)
