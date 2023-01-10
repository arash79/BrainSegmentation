from STL import create_stl_file_from_array, create_stl_file_from_nifti
from segment import segment_brain_parts, segment_arteries
from warnings import filterwarnings
import os

filterwarnings('ignore')

T1w_filename = 'T1w.nii'
eINV2_filename = 'eINV2.nii'
directory = 'C:\\Users\\User\\PycharmProjects\\BrainSegmentation\\result\\'
T1W_PATH = f'data\\{T1w_filename}'
INV2_PATH = f'data\\{eINV2_filename}'

if len(os.listdir(directory)) > 0:
    for filename in os.listdir(directory):
        os.remove(os.path.join(directory, filename))

segment_brain_parts(T1W_PATH, directory)
segmented_arteries = segment_arteries(INV2_PATH, directory)

create_stl_file_from_nifti(os.path.join(directory, f'c1{T1w_filename.split(".")[0]}.nii'), 'GM')
create_stl_file_from_nifti(os.path.join(directory, f'c2{T1w_filename.split(".")[0]}.nii'), 'WM')
create_stl_file_from_nifti(os.path.join(directory, f'c3{T1w_filename.split(".")[0]}.nii'), 'CSF')
create_stl_file_from_array(segmented_arteries, 'ARTERIES')
