# Introduction
A custom implementation of the Darknet platform tailored for the 3G-AMP research project under the Applied Physics Laboratory at the Univeristy of Washington. This project can be used to perform inferencing on a pretrained YOLOv3 network.

# Requirements
- CUDA Version: 10.1
- CUDA compilation tools, release 10.1, V10.1.105
- Tensorflow-GPU: 1.13.1
- Torch: 1.0.1.post2
- PyTorch: 1.0.1 (py3.6_cuda10.0.130_cudnn7.4.2_2)
- Cuda Toolkit: 10.0.130
- OpenCV-Python: 3.4.4.19
- Anaconda

For a detailed list of all dependencies, please refer to the [cvenv.yml](cvenv.yml) present in the root directory of this repository.

# Generating dataset
To generate an annotated dataset, use the [labelImg](https://github.com/tzutalin/labelImg) tool. Instructions on using the tool can be found at the linked repository page. The labels must be generated using the YOLO configuration provided by the tool to be compatible with the training process. Each ```<image>.jpg``` would now have a corresponding ```<image>.txt``` file associated with it which would contain the list of bounding boxes. An additional file ```classes.txt``` would be generated which would contain the names of the different classes of objects present in the dataset which have been labelled.

Once these files are ready, we move on to create our train, test and validation splits. For each of the splits, we would create a txt file where each line of the file would containing a path to an image file which belongs to the subset. 

**Note**: For this generation process, we usually collect all the images into a single folder and use that folder with the labelImg tool. Check the ```data/images_subset``` and the ```data/labels_subset``` (present on the APL machine) for an example.

# Training the model
For the training of the model, we could use either the [Darknet](https://github.com/pjreddie/darknet) or any similar implementations such as the  [YOLOv3 by Ultralytics](https://github.com/ultralytics/yolov3). For the model and weights present on this repository, we have used the original Darknet implementation. 

### Instructions for training with Darknet
1. The network architecture and the training configuration must be specified in the [cfg/yolov3-amp.cfg](cfg/yolov3-amp.cfg) file. This file would then be placed in the ```cfg``` folder under the Darknet project.
2. Next, we create a file which would specify the path to the ```train.txt``` and the ```val.txt``` file which we prepared in the previous section. This file would also specify the number of classes and path to the ```classes.txt``` which we created in the previous section. A demo of this file can be seen in [cfg/amp.data](cfg/amp.data).
3. Download pretrained convolution weights and place them in the root directory of the Darknet project. 
```sh
wget https://pjreddie.com/media/files/darknet53.conv.74
```
4. Train the model using the command:
```sh
./darknet detector train cfg/amp.data cfg/yolov3-amp.cfg darknet53.conv.74
```

# Detection using a pre-trained model
To perform detection on images using a pre-trained model, first the environment needs to be set and the dependencies need to be installed. The following command can be used to install all the dependencies present in [cvenv.yml](cvenv.yml) into a new conda environment:
```sh
conda env create -f cvenv.yml
conda activate cvenv
```

Next, create the ```test_images``` folder inside the ```data``` folder and place the images on which inferencing needs to be performed here.

Next, place the model configuration file in the ```cfg``` folder. The last best known configuration is [cfg/yolov3-amp.cfg](cfg/yolov3-amp.cfg). Place the pre-trained model weights in the ```weights``` folder. Weights corresponding to the aforementioned configuration file is [weights/yolov3.weights](weights/yolov3.weights) (present on the APL machine).

Finally, run the detector.
```sh
python detector.py
```
The output files can be found in the ```data/output``` directory. If the files have been named and placed in the same way as the instructions mentioned above, then the above command must be enough to perform the detection. If anything needs to be changed, command line arguments can be passed to the [detector.py](detector.py) script for the same. Help on the command line arguments which can be passed to the script can be obtained using the following command:
```sh
python detector.py --help
```

# Future steps

### Establishing metrics
Instead of writing all the metrics from scratch, it would be preferred to use an established framework. This would enable fair comparison and easy benchmarking of performance. One such library is [Object-Detection-Metrics](https://github.com/rafaelpadilla/Object-Detection-Metrics). The have compiled together the metrics used by various object detection challenges such as the Pascal VOC Challenge, COCO Detection Challenge, etc.

Instructions on how to compute the metrics can be found on the repository page. The library requires the ground truth predictions and the detections files to be present in a certain format. 

**Note**: The ground truth has labels in the format ```<class_name> <left> <top> <width> <height>``` whereas the predictions are of in the format ```<class_name> <left> <top> <right> <bottom>``` after processing them, so choose the appropriate flags while running the performance analysis script of the aforementioned repository. For more details on the flags, read [this section](https://github.com/rafaelpadilla/Object-Detection-Metrics#optional-arguments).


### Creating test script
The code for creating a test script is very similar to that of the [detector.py](detector.py) script. The only change we need to make is that instead of keeping a track of the images, we keep track of the file names and instead of drawing the bounding boxes on the images, we write the class names and the bounding box coordinates to a file.

All the processing done till [line 177](detector.py#L177) is required and everything afterwards can be replaced to write the predictions to files instead. Also, make sure to keep a track of the filenames instead of the image data.


### Building training platform
Due to lack of time, the [Darknet](https://github.com/pjreddie/darknet) project was used to train the system. While make the inference framework, I made sure to be completely compatible with the cfg files from the Darknet project. So any network trained on the Darknet can be used directly on this platform to perform inferencing. Hence, a training platform compatible with the training process specified in the Darknet [paper](https://arxiv.org/pdf/1506.02640v3.pdf) can be used to train the network.

Most of the heavy-lifting has been done in the inference platform and only some wrappers need to be written to parse the training specification and to train the network accordingly. A good sample of this training process can be seen in the [YOLOv3 by Ultralytics](https://github.com/ultralytics/yolov3) project.
