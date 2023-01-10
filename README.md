# BrainSegmentation

### Note: It is recommended to use this project in a virtual environment.

A full guide on how to set up a virtual environment in python is available at:

https://www.freecodecamp.org/news/how-to-setup-virtual-environments-in-python/

This project relies heavily on some third party libraries. In order to use these files you have to install those libraries first. You can install the mentioned libraries by entering the following command in your terminal (after activating the virtual environment):

```python
pip install -r requirements.txt
```

to have a fully functional installation of the modules, you have to make sure to have Matlab on your system, because spm12 python wrapper currently requires matlab installation to work.

All data were fetched from openneuro dataset which is linked below:

https://openneuro.org/datasets/ds003642/versions/1.1.0/download

I have included the original INV1 and INV2 images of code 045 in the ```data``` directory but due to upload size limits imposed by Github, I couldn't include the T1w image which is by the way crucial for brain segmentation function.

