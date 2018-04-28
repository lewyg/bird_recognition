# Bird recognition

University project for Neutral Networks course.

## Image preprocessing

Raw images with folders structure put into _resources/in_ - folders are image classes.

Run _preprocessing/run_proprocess.py_

Preprocessed images are in _resources/out_ with same folders structure. 

Image features and class labels are in _resources/data_ and _resources/labels_

## Image recognition

Two versions for multilayer Perceptron:

scikit - _recognition/run_scikit.py_

tensorflow + keras - _recognition/run_keras.py_

## Configuration

All configurations are in _config.py_