Image Captioning Using Neural Networks

Download Flickr 30K dataset and annotations for this project (https://illinois.edu/fb/sec/7805261).

Download VGG19 tensorflow model

Execution:
Give permission to requirements.sh and execute it.
$ chmod +x requirements.sh
$ ./requirements.sh

Install caffe

Install requirements using pip
$ pip install -r requirements.txt

Initialize jupyter notebook and run it
$ jupyter notebook

open and run create_flickr_dataset.ipynb

Train the model
$ python train.py

Test
Open test.py and provide path to trained model
$ python test.py "path/to/image"
