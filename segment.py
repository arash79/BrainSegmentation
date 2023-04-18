from STL import create_stl_file_from_array
from multiprocessing.dummy import Pool
from warnings import filterwarnings
from skimage.filters import frangi
from skimage import measure
import nibabel as nib
import pandas as pd
import numpy as np
import time
import os

filterwarnings('ignore')


class SegmentBrain:

    def __init__(self, T1w_path, INV2_path):

        T1w = nib.load(T1w_path).get_fdata()
        INV2 = nib.load(INV2_path).get_fdata()

        self.affine_transform = nib.load(T1w_path).affine

        self.T1w_file_name = T1w_path.split('\\')[-1][:-4]
        self.INV2_file_name = INV2_path.split('\\')[-1][:-4]
        self.T1w_file_path = T1w_path
        self.INV2_file_path = INV2_path

        self.T1w = (T1w - np.amin(T1w)) / (np.amax(T1w) - np.amin(T1w))
        self.INV2 = (INV2 - np.amin(INV2)) / (np.amax(INV2) - np.amin(INV2))

        self.shape = self.T1w.shape
        self.ResultsPath = 'results'
        self.FieldTripPath = os.path.join(self.ResultsPath, 'FieldTrip')
        self.traversed = []

    @staticmethod
    def filter_connected_components(array, length_threshold):

        """
        This method filters out connected components of any array which have size smaller than a given length threshold
        :param array: The given N-D array
        :param length_threshold: The desired minimum length
        :return: Filtered array
        """

        labeled_array = measure.label(array, connectivity=2)
        properties = measure.regionprops(labeled_array)
        keep_mask = np.zeros_like(labeled_array, dtype=bool)

        for property_ in properties:
            if property_.major_axis_length < length_threshold:
                keep_mask[labeled_array == property_.label] = True

        filtered_array = array.copy()
        filtered_array[keep_mask] = 0

        return filtered_array

    def _check_dim(self, coordinate: tuple) -> bool:

        """
        This method checks whether the given coordinate lies in the boundary of the current array or it exceeds its
        boundaries. This method is used in the region growing algorithm.

        :param coordinate: given coordinate system
        :return: True if the given coordinate is inside the plane otherwise it returns False
        """

        excess_x = coordinate[0] < 0 or coordinate[0] >= self.shape[0] - 1
        excess_y = coordinate[1] < 0 or coordinate[1] >= self.shape[1] - 1

        return not (excess_x or excess_y)

    def _region_growing(self, given_seed: tuple, vessel_mask: np.array, masked_cross_section: np.array,
                        threshold: float, voxel_value: float) -> None:

        """
        This method expands the given seed region by checking that whether its neighbors have an intensity value within
        the tolerance range of the given seed intensity. If True it announces them as vessels otherwise it let them be.

        :param given_seed: The seed which was provided using the Frangi vesselness filtering
        :param vessel_mask: The whole cross-section consisting of different seeds obtained by the Frangi method
        :param masked_cross_section: The weighted average mask of T1w and INV2 cross-sections
        :param threshold: The tolerance range of voxels intensities similarities
        :param voxel_value: The minimum intensity value to detect a voxel as a vessel, It is used to reduce noise
        :return: None, It effects directly on to the given vessel mask
        """

        stack = [given_seed]

        while len(stack):

            current_seed = stack.pop()

            if current_seed in self.traversed:
                continue

            if masked_cross_section[current_seed] < voxel_value:
                vessel_mask[current_seed] = np.False_
                continue

            current_x, current_y = current_seed[0], current_seed[1]

            adjacent_voxels = []
            possible_moves = [-1, 0, 1]

            for i in possible_moves:
                for j in possible_moves:
                    if i != 0 or j != 0:
                        neighbor = (current_x + i, current_y + j)
                        if not vessel_mask[neighbor]:
                            adjacent_voxels.append(neighbor)

            self.traversed.append(current_seed)

            for neighbor in adjacent_voxels:
                if np.isclose(masked_cross_section[current_seed], masked_cross_section[neighbor], rtol=threshold):
                    vessel_mask[neighbor] = np.True_
                    if neighbor not in stack and self._check_dim(neighbor):
                        stack.append(neighbor)

    def _segment_arteries(self, rg_threshold: float) -> np.array:

        """
        This method generates the vessel mask using the Frangi vesselness filtering. Since it generates noisy results
        it is protected and will be called from another method which is coded below. This method calls uses
        multiprocessing to increase the calculations speed.

        :param rg_threshold: desired threshold for region growing algorithm, normally between 0.1 and 0.4 is fine
        :return: vessel mask (which by the way can be noisy)
        """

        first_axis, second_axis, third_axis = np.zeros(self.shape), np.zeros(self.shape), np.zeros(self.shape)
        first_axis_length, second_axis_length, third_axis_length = self.shape[0], self.shape[1], self.shape[2]

        def frangi_filter(T1w_cross_section: np.array, INV2_cross_section: np.array, axis: str,
                          layer_index: int) -> None:

            T1w_result = frangi(T1w_cross_section, sigmas=np.arange(.5, 1.5, .25), alpha=0.9, beta=0.9,
                                gamma=40, black_ridges=False)
            INV2_result = frangi(INV2_cross_section, sigmas=np.arange(.5, 1.5, .25), alpha=0.9, beta=0.9,
                                 gamma=40, black_ridges=False)

            try:
                T1w_threshold = np.percentile(T1w_result[np.nonzero(T1w_result)], 99)
            except IndexError:
                T1w_threshold = 0

            try:
                INV2_threshold = np.percentile(INV2_result[np.nonzero(INV2_result)], 97)
            except IndexError:
                INV2_threshold = 0

            T1w_result = T1w_result > T1w_threshold
            INV2_result = INV2_result > INV2_threshold

            layer_dataframe = pd.DataFrame({'T1w': T1w_cross_section.ravel(), 'INV2': INV2_cross_section.ravel()})
            correlation_matrix = layer_dataframe.corr()
            try:
                eigen_values = np.linalg.eig(correlation_matrix)[0]
                T1w_weight, INV2_weight = eigen_values[0], eigen_values[1]
            except np.linalg.LinAlgError:
                T1w_weight, INV2_weight = 1, 1

            T1w_weight, INV2_weight = INV2_weight, T1w_weight

            numerator = (T1w_weight * T1w_cross_section) + (INV2_weight * INV2_cross_section)
            denominator = (T1w_weight + INV2_weight)
            masked_cross_section = numerator / denominator

            try:
                voxel_threshold = np.percentile(masked_cross_section[np.nonzero(masked_cross_section)], 90)
            except IndexError:
                voxel_threshold = 0

            masked_frangi_results = T1w_result | INV2_result

            vessel_mask = (masked_cross_section > voxel_threshold)
            filtered_vessels = masked_frangi_results & vessel_mask

            nonzero_elements = np.nonzero(filtered_vessels)
            vessel_seeds = list(zip(nonzero_elements[0], nonzero_elements[1]))

            for seed in vessel_seeds:
                self._region_growing(given_seed=seed, vessel_mask=filtered_vessels,
                                     masked_cross_section=masked_cross_section, threshold=rg_threshold,
                                     voxel_value=voxel_threshold)

            if axis == 'first':
                first_axis[layer_index, :, :] = filtered_vessels
            if axis == 'second':
                second_axis[:, layer_index, :] = filtered_vessels
            if axis == 'third':
                third_axis[:, :, layer_index] = filtered_vessels

        parameters_along_first_axis = [(self.T1w[i, :, :], self.INV2[i, :, :], 'first', i) for i in range(first_axis_length)]
        parameters_along_second_axis = [(self.T1w[:, i, :], self.INV2[:, i, :], 'second', i) for i in range(second_axis_length)]
        parameters_along_third_axis = [(self.T1w[:, :, i], self.INV2[:, :, i], 'third', i) for i in range(third_axis_length)]

        time_of_start = time.time()
        print('applying vesselness filtering...')
        Pool(processes=4 * os.cpu_count()).starmap(frangi_filter, parameters_along_first_axis)
        print('step 1 of 3 completed.')
        Pool(processes=4 * os.cpu_count()).starmap(frangi_filter, parameters_along_second_axis)
        print('step 2 of 3 completed.')
        Pool(processes=4 * os.cpu_count()).starmap(frangi_filter, parameters_along_third_axis)
        print('process finished.')

        print(f'total elapsed time {round(time.time() - time_of_start)} seconds')

        merged_result = ((first_axis.astype(int) + second_axis.astype(int) + third_axis.astype(int)) >= 2)

        return merged_result

    def segment_arteries(self, region_growing_threshold: float = 0.4, drop_intensively: bool = True,
                         save_as_nifti: bool = True) -> np.array:

        """
        This method calls the private method of above. It requires the user to perform the FieldTrip segmentation on
        both T1w and INV2 modalities in advance, because it uses some of its results to mask out unwanted noises in the
        raw vessel mask obtained by the above method.

        :param region_growing_threshold: The region growing threshold which was discussed previously
        :param drop_intensively: if set to True it will drop components of size smaller than a length threshold using
        the static method of above.
        :param save_as_nifti: If set to True it will save the numpy array as nifti file
        :return: vessel mask
        """

        T1w_soft_tissue_path = os.path.join(self.FieldTripPath, 'ST_T1w.nii')
        T1w_skull_path = os.path.join(self.FieldTripPath, 'SKULL_T1w.nii')
        INV2_soft_tissue_path = os.path.join(self.FieldTripPath, 'ST_INV2.nii')
        INV2_skull_path = os.path.join(self.FieldTripPath, 'SKULL_INV2.nii')

        try:
            T1w_soft_tissue = nib.load(T1w_soft_tissue_path).get_fdata()
            T1w_skull = nib.load(T1w_skull_path).get_fdata()
            INV2_soft_tissue = nib.load(INV2_soft_tissue_path).get_fdata()
            INV2_skull = nib.load(INV2_skull_path).get_fdata()
        except FileNotFoundError:
            print('You should first perform FieldTrip brain segmentation to obtain the cleaning masks.')
            return

        segmented_vessels = self._segment_arteries(rg_threshold=region_growing_threshold)
        extra_voxels = T1w_soft_tissue + INV2_soft_tissue + T1w_skull + INV2_skull
        mask = np.zeros_like(extra_voxels, dtype=bool)
        mask[extra_voxels == 0] = True
        spared_area_boundaries = (self.shape[2] // 6, self.shape[2] // 2)
        mask[:, :, spared_area_boundaries[0]: spared_area_boundaries[1]] = True
        cleaned_vessels = np.where(mask, segmented_vessels, 0)

        if drop_intensively is True:
            connected_components_minimum_length = 100
            cleaned_vessels = SegmentBrain.filter_connected_components(array=cleaned_vessels,
                                                                       length_threshold=connected_components_minimum_length)

        if save_as_nifti is True:
            vessel_nifti = nib.Nifti1Image(cleaned_vessels, self.affine_transform)
            nib.save(vessel_nifti, os.path.join(self.FieldTripPath, "vessel_mask.nii"))

        return cleaned_vessels

    def ft_brain_segmentation(self, nifti_file_path, FieldTrip_PATH):

        """
        This method performs FieldTrip brain segmentation on the given nifti file.

        :param nifti_file_path: The path to nifti file
        :param FieldTrip_PATH: The local path to FieldTrip MatLab package
        :return: A dictionary consisting of numpy arrays of segmented parts
        """

        config = {'brainsmooth': 100, 'scalpsmooth': 100, 'skullsmooth': 100, 'brainthreshold': 0.5,
                  'scalpthreshold': 100, 'skullthreshold': 1000, 'downsample': 1, 'output': {'tpm'},
                  'spmmethod': 'mars'}

        print('starting MATLAB engine...')
        engine = matlab.engine.start_matlab()

        FieldTrip_PATH = engine.genpath(FieldTrip_PATH)

        print('adding FieldTrip to path...')
        engine.addpath(FieldTrip_PATH)

        print('setting environment...')
        setting_result = engine.ft_defaults

        print('reading MRI data...')
        file_name = nifti_file_path.split('\\')[-1][:-4]
        MRI = engine.ft_read_mri(nifti_file_path)
        MRI['coordsys'] = 'ras'

        print('segmenting tissues...')
        segmented = engine.ft_volumesegment(config, MRI)

        print('dividing segmented parts...')
        gray_matter = np.array(segmented['gray'])
        white_matter = np.array(segmented['white'])
        CSF = np.array(segmented['csf'])
        skull = np.array(segmented['bone'])
        soft_tissue = np.array(segmented['softtissue'])

        engine.quit()

        print('saving results...')

        GM_nifti = nib.Nifti1Image(gray_matter, self.affine_transform)
        nib.save(GM_nifti, os.path.join(self.FieldTripPath, "GM_{}.nii".format(file_name)))

        WM_nifti = nib.Nifti1Image(white_matter, self.affine_transform)
        nib.save(WM_nifti, os.path.join(self.FieldTripPath, "WM_{}.nii".format(file_name)))

        CSF_nifti = nib.Nifti1Image(CSF, self.affine_transform)
        nib.save(CSF_nifti, os.path.join(self.FieldTripPath, "CSF_{}.nii".format(file_name)))

        SKULL_nifti = nib.Nifti1Image(skull, self.affine_transform)
        nib.save(SKULL_nifti, os.path.join(self.FieldTripPath, "SKULL_{}.nii".format(file_name)))

        ST_nifti = nib.Nifti1Image(soft_tissue, self.affine_transform)
        nib.save(ST_nifti, os.path.join(self.FieldTripPath, "ST_{}.nii".format(file_name)))

        return {'GM': gray_matter, 'WM': white_matter, 'csf': CSF, 'skull': skull, 'st': soft_tissue}


INV2_FILE, T1W_FILE = 'eINV2.nii', 'eT1w.nii'
class_object = SegmentBrain(T1w_path=T1W_FILE, INV2_path=INV2_FILE)
a = class_object.segment_arteries(drop_intensively=True)
# a = filter_connected_components(a, 50)
# one = getLargestCC(a, 1)
# print(len(np.nonzero(one)[0]))
# two = getLargestCC(a, 2)
# a = clean_results(a)
create_stl_file_from_array(a, 'the_best')
# create_stl_file_from_array(b, 'b')
# create_stl_file_from_array(c, 'c')
