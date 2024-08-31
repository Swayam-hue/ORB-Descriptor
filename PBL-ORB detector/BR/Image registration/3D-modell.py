import SimpleITK as sitk

def save_combined_central_slice(fixed, moving, transform, file_name_prefix):
    global iteration_number 
    alpha = 0.7

    central_indexes = [i/2 for i in fixed.GetSize()]

    moving_transformed = sitk.Resample(moving, fixed, transform, 
                                       sitk.sitkLinear, 0.0, 
                                       moving_image.GetPixelIDValue())
    #extract the central slice in xy, xz, yz and alpha blend them                                   
    combined = [(1.0 - alpha)*fixed[:,:,central_indexes[2]] + \
                   alpha*moving_transformed[:,:,central_indexes[2]],
                  (1.0 - alpha)*fixed[:,central_indexes[1],:] + \
                  alpha*moving_transformed[:,central_indexes[1],:],
                  (1.0 - alpha)*fixed[central_indexes[0],:,:] + \
                  alpha*moving_transformed[central_indexes[0],:,:]]
    #resample the alpha blended images to be isotropic and rescale intensity
    #values so that they are in [0,255], this satisfies the requirements 
    #of the jpg format 
    combined_isotropic = []
    for img in combined:
        original_spacing = img.GetSpacing()
        original_size = img.GetSize()
        min_spacing = min(original_spacing)
        new_spacing = [min_spacing, min_spacing]
        new_size = [int(round(original_size[0]*(original_spacing[0]/min_spacing))), 
                    int(round(original_size[1]*(original_spacing[1]/min_spacing)))]
        resampled_img = sitk.Resample(img, new_size, sitk.Transform(), 
                                      sitk.sitkLinear, img.GetOrigin(),
                                      new_spacing, img.GetDirection(), 0.0, 
                                      img.GetPixelIDValue())        
        combined_isotropic.append(sitk.Cast(sitk.RescaleIntensity(resampled_img), 
                                            sitk.sitkUInt8))
    #tile the three images into one large image and save using the given file 
    #name prefix and the iteration number
    sitk.WriteImage(sitk.Tile(combined_isotropic, (1,3)), 
                    file_name_prefix+ format(iteration_number, '03d') + '.jpg')
    iteration_number+=1    



#read the images
fixed_image =  sitk.ReadImage("training_001_ct.mha", sitk.sitkFloat32)
moving_image = sitk.ReadImage("training_001_mr_T1.mha", sitk.sitkFloat32) 

#initial alignment of the two volumes
transform = sitk.CenteredTransformInitializer(fixed_image, 
                                              moving_image, 
                                              sitk.Euler3DTransform(), 
                                              sitk.CenteredTransformInitializerFilter.GEOMETRY)

#multi-resolution rigid registration using Mutual Information
registration_method = sitk.ImageRegistrationMethod()
registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
registration_method.SetMetricSamplingPercentage(0.01)
registration_method.SetInterpolator(sitk.sitkLinear)
registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, 
                                                  numberOfIterations=100, 
                                                  convergenceMinimumValue=1e-6, 
                                                  convergenceWindowSize=10)
registration_method.SetOptimizerScalesFromPhysicalShift()
registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])
registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2,1,0])
registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
registration_method.SetInitialTransform(transform)

#add iteration callback, save central slice in xy, xz, yz planes
global iteration_number
iteration_number = 0
registration_method.AddCommand(sitk.sitkIterationEvent, 
                               lambda: save_combined_central_slice(fixed_image,
                                                                   moving_image,
                                                                   transform, 
                                                                   'output/iteration'))

registration_method.Execute(fixed_image, moving_image)
