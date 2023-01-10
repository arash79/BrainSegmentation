from multiprocessing.dummy import Pool
from skimage.filters import frangi
from spm12.regseg import seg_spm
import nibabel as nib
import numpy as np
import time
import os


def segment_brain_parts(path_to_file, output_directory):
    seg_spm(path_to_file, visual=True, store_nat_gm=True, store_nat_wm=True, store_nat_csf=True,
            store_fwd=True, store_inv=True, outpath=output_directory)


def segment_arteries(path_to_file, output_directory='result'):

    print('preprocessing data...')
    nifti_data = nib.load(path_to_file)
    MRI = nifti_data.get_fdata()
    normalized_nifti_array = (MRI - np.amin(MRI)) / (np.amax(MRI) - np.amin(MRI))
    file_shape = normalized_nifti_array.shape

    def apply_frangi_filter(array_cross_section, given_array, layer_index, axis):

        frangi_results = frangi(array_cross_section, sigmas=np.arange(0, 0.01, 0.0001), alpha=0.618, beta=0.786,
                                gamma=5, black_ridges=False, mode='reflect', cval=0)
        non_zero_counts = np.sum(frangi_results > 0)
        binary_layer = (frangi_results > 13 * np.sum(frangi_results) / non_zero_counts).astype(np.int_)

        if axis == 'first':
            given_array[layer_index, :, :] = binary_layer
        if axis == 'second':
            given_array[:, layer_index, :] = binary_layer
        if axis == 'third':
            given_array[:, :, layer_index] = binary_layer

    first_axis, second_axis, third_axis = np.zeros(file_shape), np.zeros(file_shape), np.zeros(file_shape)

    first_axis_length, second_axis_length, third_axis_length = file_shape[0], file_shape[1], file_shape[2]

    parameters_along_first_axis = [(normalized_nifti_array[i, :, :], first_axis, i, 'first') for i in range(first_axis_length)]
    parameters_along_second_axis = [(normalized_nifti_array[:, i, :], second_axis, i, 'second') for i in range(second_axis_length)]
    parameters_along_third_axis = [(normalized_nifti_array[:, :, i], third_axis, i, 'third') for i in range(third_axis_length)]

    time_of_start = time.time()
    print('applying vesselness filtering...')
    Pool(processes=os.cpu_count()).starmap(apply_frangi_filter, parameters_along_first_axis)
    print('step 1 of 3 completed.')
    Pool(processes=os.cpu_count()).starmap(apply_frangi_filter, parameters_along_second_axis)
    print('step 2 of 3 completed.')
    Pool(processes=os.cpu_count()).starmap(apply_frangi_filter, parameters_along_third_axis)
    print('process finished.')

    print(f'total elapsed time {round(time.time() - time_of_start)} seconds')

    merged_result = ((first_axis + second_axis + third_axis) >= 2).astype(np.int_)

    print('saving results...')
    merged_result_nifti = nib.Nifti1Image(merged_result, nifti_data.affine)

    nib.save(merged_result_nifti, os.path.join(output_directory, "{}_arteries.nii".format(path_to_file.split('\\')[-1][:-4])))

    return merged_result
