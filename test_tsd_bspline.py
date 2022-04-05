# [THINSHELLDEMONS] Read Mesh and Create TSD object

# Build the Module with Python Wrapping and import the itk package which is available under Wrapping/Generators/Python
#import sys
#sys.path.append('/home/pranjal.sahu/ITKPR7/ITK/ITK-build/Wrapping/Generators/Python')

#import itkConfig
#itkConfig.LazyLoading = False

import numpy as np
import itk
import math
import sys


print('imageDiagonal ', sys.argv[1])
print('StretchWeight ', sys.argv[2])
print('BendWeight ',      sys.argv[3])
print('GeometricWeight ', sys.argv[4])
print('MaxStep ',         sys.argv[5])

# '/home/pranjal.sahu/SlicerMorph/A_J_skull_temp2.vtk'
# '/home/pranjal.sahu/SlicerMorph/NOD_SHILTJ_temp2.vtk'

#fixedMesh = itk.meshread('/home/pranjal.sahu/SlicerMorph/A_J_skull_95.vtk', itk.D)
#movingMesh = itk.meshread('/home/pranjal.sahu/SlicerMorph/NOD_SHILTJ_95.vtk', itk.D)

movingMesh = itk.meshread('/home/pranjal.sahu/SlicerMorph/A_J_skull_temp5.vtk', itk.D)
fixedMesh = itk.meshread('/home/pranjal.sahu/SlicerMorph/NOD_SHILTJ_temp5.vtk', itk.D)

#movingMesh = itk.meshread('/home/pranjal.sahu/SlicerMorph/A_J_skull_temp3.vtk', itk.D)
#fixedMesh = itk.meshread('/home/pranjal.sahu/SlicerMorph/NOD_SHILTJ_temp3.vtk', itk.D)


#fixedMesh = itk.meshread('/media/pranjal.sahu/moredata/ITK/Modules/External/ITKThinShellDemons/test/Baseline/fixedMesh.vtk', itk.D)
#movingMesh = itk.meshread('/media/pranjal.sahu/moredata/ITK/Modules/External/ITKThinShellDemons/test/Baseline/movingMesh.vtk', itk.D)


e_metric = itk.EuclideanDistancePointSetToPointSetMetricv4.PSD3.New()
e_metric.SetFixedPointSet(fixedMesh)
e_metric.SetMovingPointSet(movingMesh)

print('Before Euclidean Metric ', e_metric.GetValue())

#fixedMesh = itk.meshread(sys.argv[1], itk.D)
#movingMesh = itk.meshread(sys.argv[2], itk.D)

fixedMesh.BuildCellLinks()
movingMesh.BuildCellLinks()

PixelType = itk.D
Dimension = 3

MeshType        = itk.Mesh[itk.D, Dimension]
FixedImageType  = itk.Image[PixelType, Dimension]
MovingImageType = itk.Image[PixelType, Dimension]


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

print('Length is ', maxBounds - minBounds)
#print('minBounds ', minBounds)
#print('minBounds ', minBounds)

imageDiagonal = int(sys.argv[1])#400
spacing = np.sqrt(bounding_box.GetDiagonalLength2()) / imageDiagonal
diff = maxBounds - minBounds

print('Spacing ', spacing)
print(bounding_box.GetDiagonalLength2())
print('minBounds ', minBounds)
print('maxBounds ', maxBounds)

fixedImageSize    = [0]*3
fixedImageSize[0] = math.ceil( 1.2 * diff[0] / spacing )
fixedImageSize[1] = math.ceil( 1.2 * diff[1] / spacing )
fixedImageSize[2] = math.ceil( 1.2 * diff[2] / spacing )

fixedImageOrigin    = [0]*3
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



SplineOrder = 3
TransformType  = itk.BSplineTransform[itk.D, Dimension, SplineOrder]
InitializerType = itk.BSplineTransformInitializer[TransformType, FixedImageType]

transform = TransformType.New()


numberOfGridNodesInOneDimension = 6
meshSize = itk.Size[3]
#meshSize.Fill(numberOfGridNodesInOneDimension - SplineOrder)

transformInitializer = InitializerType.New()

transformInitializer.SetTransform(transform)
transformInitializer.SetImage(fixedImage)
transformInitializer.SetTransformDomainMeshSize(numberOfGridNodesInOneDimension - SplineOrder)
transformInitializer.InitializeTransform()



MetricType = itk.ThinShellDemonsMetricv4.MD3
metric = MetricType.New()
metric.SetStretchWeight(float(sys.argv[2]))#(0.01)
metric.SetBendWeight(float(sys.argv[3]))# 0.01
metric.SetGeometricFeatureWeight(float(sys.argv[4]))#0.05
metric.UseConfidenceWeightingOn()
#metric.UseMaximalDistanceConfidenceSigmaOff()
metric.UseMaximalDistanceConfidenceSigmaOff()
metric.UpdateFeatureMatchingAtEachIterationOn()
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


optimizer = itk.ConjugateGradientLineSearchOptimizerv4Template.D.New()
optimizer.SetNumberOfIterations( int(sys.argv[6]) )
optimizer.SetScalesEstimator( shiftScaleEstimator )
optimizer.SetMaximumStepSizeInPhysicalUnits( float(sys.argv[5]) )
optimizer.SetMinimumConvergenceValue( 0.0 )
optimizer.SetConvergenceWindowSize( int(sys.argv[6]) )

def iteration_update():
    metric_value = optimizer.GetValue()
    current_parameters = optimizer.GetCurrentPosition()
    print(f"Metric: {metric_value:.8g}")

iteration_command = itk.PyCommand.New()
iteration_command.SetCommandCallable(iteration_update)
optimizer.AddObserver(itk.IterationEvent(), iteration_command)

print('Optimizer created')


print('Number of parameters ', transform.GetNumberOfParameters())


AffineRegistrationType = itk.ImageRegistrationMethodv4.REGv4D3D3TD3D3MD3.New()
registration = AffineRegistrationType.New()
registration.SetNumberOfLevels(1)
registration.SetObjectName("registration")
registration.SetFixedPointSet(movingMesh)
registration.SetMovingPointSet(fixedMesh)
registration.SetInitialTransform(transform)
registration.SetMetric(metric)
registration.SetOptimizer(optimizer)
registration.InPlaceOn()

# numberOfLevels = 3
# registration.SetNumberOfLevels(numberOfLevels)

# shrinkFactorsPerLevel  = itk.Array[itk.UI]()
# shrinkFactorsPerLevel.SetSize(numberOfLevels)
# shrinkFactorsPerLevel[0] = 3
# shrinkFactorsPerLevel[1] = 2
# shrinkFactorsPerLevel[2] = 1

# smoothingSigmasPerLevel = itk.Array[itk.UI]()
# smoothingSigmasPerLevel.SetSize(numberOfLevels)
# smoothingSigmasPerLevel[0] = 2
# smoothingSigmasPerLevel[1] = 1
# smoothingSigmasPerLevel[2] = 0

# registration.SetShrinkFactorsPerLevel(shrinkFactorsPerLevel)
# registration.SetSmoothingSigmasPerLevel(smoothingSigmasPerLevel)



print('Registration Object created')
print('Initial Value of Metric ', metric.GetValue())

try:
    registration.Update()
except e:
    print('Error is ', e)

print('Final Value of Metric ', metric.GetValue())



e_metric = itk.EuclideanDistancePointSetToPointSetMetricv4.PSD3.New()
e_metric.SetFixedPointSet(fixedMesh)
e_metric.SetMovingPointSet(movingMesh)
print('Final Euclidean Metric 1 ', e_metric.GetValue())

finalTransform = registration.GetModifiableTransform()

#pl1 = np.array(finalTransform.GetParameters())
#np.save('displacement_field.npy', field)

numberOfPoints = movingMesh.GetNumberOfPoints()
for n in range(0, numberOfPoints):
    movingMesh.SetPoint(n, finalTransform.TransformPoint(movingMesh.GetPoint(n)))

e_metric = itk.EuclideanDistancePointSetToPointSetMetricv4.PSD3.New()
e_metric.SetFixedPointSet(fixedMesh)
e_metric.SetMovingPointSet(movingMesh)
print('Final Euclidean Metric 2 ', e_metric.GetValue())

#itk.meshwrite(movingMesh, f'./allmeshresults/displacementMovingMesh_reverse_{sys.argv[1]}-{sys.argv[2]}-{sys.argv[3]}-{sys.argv[4]}-{sys.argv[5]}-{e_metric.GetValue()}.vtk')
itk.meshwrite(movingMesh, 'result2.vtk')
np.save(f'{sys.argv[1]}-{sys.argv[2]}-{sys.argv[3]}-{sys.argv[4]}-{sys.argv[5]}_reverse.npy', [sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5],  e_metric.GetValue()])
