import itkConfig
itkConfig.LazyLoading = False

import numpy as np
import itk
from itk import ThinShellDemonsMetricv4 as tsd
from itk import ConjugateGradientLineSearchOptimizerv4Template as itkc
import math


fixedMesh = itk.meshread('A_J_skull_temp1.vtk', itk.D)
movingMesh = itk.meshread('NOD_SHILTJ_temp1.vtk', itk.D)

fixedMesh.BuildCellLinks()
movingMesh.BuildCellLinks()

PixelType = itk.D
Dimension = 3

MeshType = itk.Mesh[itk.D, Dimension]
FixedImageType = itk.Image[PixelType, Dimension]
MovingImageType = itk.Image[PixelType, Dimension]

print('Mesh Read completed')
print(fixedMesh.GetNumberOfPoints())
print(movingMesh.GetNumberOfPoints())

# For getting the Bounding Box
ElementIdentifierType = itk.UL
CoordType = itk.F
Dimension = 3

VecContType = itk.VectorContainer[
    ElementIdentifierType, itk.Point[CoordType, Dimension]
]
bounding_box = itk.BoundingBox[ElementIdentifierType, Dimension, CoordType, VecContType].New()
bounding_box.SetPoints(movingMesh.GetPoints())
bounding_box.ComputeBoundingBox()

minBounds = np.array(bounding_box.GetMinimum())
maxBounds = np.array(bounding_box.GetMaximum())

imageDiagonal = 5
spacing = np.sqrt(bounding_box.GetDiagonalLength2()) / imageDiagonal
diff = maxBounds - minBounds

fixedImageSize = [0]*3
fixedImageSize[0] = math.ceil( 1.2 * diff[0] / spacing )
fixedImageSize[1] = math.ceil( 1.2 * diff[1] / spacing )
fixedImageSize[2] = math.ceil( 1.2 * diff[2] / spacing )

fixedImageOrigin = [0]*3
fixedImageOrigin[0] = minBounds[0] - 0.1 * diff[0]
fixedImageOrigin[1] = minBounds[1] - 0.1 * diff[1]
fixedImageOrigin[2] = minBounds[2] - 0.1 * diff[2]

fixedImageSpacing = np.ones(3)*spacing
fixedImageDirection = np.identity(3)


fixedImage = FixedImageType.New()
fixedImage.SetRegions(fixedImageSize)
fixedImage.SetOrigin( fixedImageOrigin )
fixedImage.SetDirection( fixedImageDirection )
fixedImage.SetSpacing( fixedImageSpacing )
fixedImage.Allocate()


# Affine Transform Object
TransformType = itk.AffineTransform.D3
transform = TransformType.New()
transform.SetIdentity()
transform.SetCenter(minBounds + (maxBounds - minBounds)/2)

print('Transform Created')
print(transform)


MetricType = tsd.MD3
metric = MetricType.New()
metric.SetStretchWeight(1)
metric.SetBendWeight(5)
metric.SetGeometricFeatureWeight(10)
metric.UseConfidenceWeightingOn()
metric.UseMaximalDistanceConfidenceSigmaOn()
metric.UpdateFeatureMatchingAtEachIterationOff()
metric.SetMovingTransform(transform)
# Reversed due to using points instead of an image
# to keep semantics the same as in itkThinShellDemonsTest.cxx
# For the ThinShellDemonsMetricv4 the fixed mesh is regularized
metric.SetFixedPointSet(movingMesh)
metric.SetMovingPointSet(fixedMesh)
metric.SetVirtualDomainFromImage(fixedImage)
metric.Initialize()

print('TSD Metric Created')

shiftScaleEstimator = itk.RegistrationParameterScalesFromPhysicalShift[MetricType].New()
shiftScaleEstimator.SetMetric(metric)
shiftScaleEstimator.SetVirtualDomainPointSet(metric.GetVirtualTransformedPointSet())


optimizer = itkc.D.New()
optimizer.SetNumberOfIterations( 50 )
optimizer.SetScalesEstimator( shiftScaleEstimator )
optimizer.SetMaximumStepSizeInPhysicalUnits( 0.5 )
optimizer.SetMinimumConvergenceValue( 0.0 )
optimizer.SetConvergenceWindowSize( 10 )

def iteration_update():
    metric_value = optimizer.GetValue()
    current_parameters = optimizer.GetCurrentPosition()
    print(f"Metric: {metric_value:.8g}")

iteration_command = itk.PyCommand.New()
iteration_command.SetCommandCallable(iteration_update)
optimizer.AddObserver(itk.IterationEvent(), iteration_command)

print('Optimizer created')


AffineRegistrationType = itk.ImageRegistrationMethodv4.REGv4D3D3TD3D3MD3.New()
registration = AffineRegistrationType.New()
registration.SetNumberOfLevels(1)
registration.SetObjectName("registration")
registration.SetFixedPointSet(movingMesh)
registration.SetMovingPointSet(fixedMesh)
registration.SetInitialTransform(transform)
registration.SetMetric(metric)
registration.SetOptimizer(optimizer)

print('Registration Object created')
print('Initial Value of Metric ', metric.GetValue())

try:
    registration.Update()
except e:
    print('Error is ', e)

print('Final Value of Metric ', metric.GetValue())

finalTransform = registration.GetModifiableTransform()
numberOfPoints = movingMesh.GetNumberOfPoints()
for n in range(0, numberOfPoints):
    movingMesh.SetPoint(n, finalTransform.TransformPoint(movingMesh.GetPoint(n)))

itk.meshwrite(movingMesh, "affineMovingMesh.vtk")


# For doing ICP registration in ITK
# Dimension = 3

# PointSetType = itk.PointSet[itk.F, Dimension]

# fixedMesh = itk.meshread('/home/pranjal.sahu/Downloads/129X1_SVJ_.vtk')
# movingMesh = itk.meshread('/home/pranjal.sahu/Downloads/129S1_SVIMJ_.vtk')

# fixedPS = PointSetType.New()
# movingPS = PointSetType.New()

# randomMovingPS = itk.array_from_vector_container(movingMesh.GetPoints())
# randomFixedPS = itk.array_from_vector_container(fixedMesh.GetPoints())

# index = np.random.choice(movingMesh.GetNumberOfPoints(), 5000)
# randomMovingPS = np.take(randomMovingPS, index, 0)

# index = np.random.choice(fixedMesh.GetNumberOfPoints(), 5000)
# randomFixedPS = np.take(randomFixedPS, index, 0)


# p1 = itk.VectorContainer.ULPF3.New()
# p2 = itk.VectorContainer.ULPF3.New()


# for i in range(5000):
#     p1.InsertElement(i, randomMovingPS[i].astype('float'))
#     p2.InsertElement(i, randomFixedPS[i].astype('float'))

# #fixedPS.SetPoints(fixedMesh.GetPoints())
# #movingPS.SetPoints(movingMesh.GetPoints())

# #fixedPS.SetPoints(itk.vector_container_from_array(randomFixedPS))
# #movingPS.SetPoints(itk.vector_container_from_array(randomMovingPS))
# movingPS.SetPoints(p1)
# fixedPS.SetPoints(p2)


# print(fixedMesh.GetNumberOfPoints(), movingMesh.GetNumberOfPoints())
# print(fixedPS.GetNumberOfPoints(), movingPS.GetNumberOfPoints())

# OptimizerType    = itk.GradientDescentOptimizerv4
# MetricType       = itk.EuclideanDistancePointSetToPointSetMetricv4[type(fixedPS)]
# TransformType    = itk.VersorRigid3DTransform[itk.D]
# RegistrationType = itk.PointSetToPointSetRegistrationMethod[PointSetType, PointSetType]

# metric       = MetricType.New()
# transform    = TransformType.New()
# optimizer    = OptimizerType.New()
# registration = RegistrationType.New()

# print('-------------------------------------------')
# print('Transform Parameters before Optimization')
# print(np.array(transform.GetParameters()))
# print('-------------------------------------------')

# metric.SetFixedPointSet(fixedPS)
# metric.SetMovingPointSet(movingPS)
# metric.SetMovingTransform(transform)
# metric.Initialize()

# print('Init Done')

# numberOfIterations = 100
# gradientTolerance  = 1e-5
# valueTolerance     = 1e-5   
# epsilonFunction    = 1e-6
# learningRate       = 0.001

# #optimizer.SetScales(scales)
# optimizer.SetNumberOfIterations(numberOfIterations)
# optimizer.SetMetric(metric)
# optimizer.SetLearningRate(learningRate)

# def iteration_update():
#     metric_value = optimizer.GetValue()
#     current_parameters = optimizer.GetCurrentPosition()
#     print(f"Metric: {metric_value:.8g}")
#     print("Current Parameters ", np.array(current_parameters))

# iteration_command = itk.PyCommand.New()
# iteration_command.SetCommandCallable(iteration_update)
# optimizer.AddObserver(itk.IterationEvent(), iteration_command)

# print('Optimizer created')


# optimizer.StartOptimization()


# print('-------------------------------------------')
# print('Transform Parameters after Optimization')
# print(np.array(transform.GetParameters()))
# print('-------------------------------------------')



# numberOfPoints = movingMesh.GetNumberOfPoints()
# for i in range(0, numberOfPoints):
#     movingMesh.SetPoint(i, transform.TransformPoint(movingMesh.GetPoint(i)))


# itk.meshwrite(movingMesh, 'movingMesh_transformed.vtk')
