# ALEX: Detecting Environmental Constraints towards Fluid Exoskeleton Gait (M-12-FJUS-051)

## Section 1: Feature Extraction

## Feature Extraction

This repository contains the scripts to detect staircases real-time using Intel RealSense D435i depth camera and calculate its tread and riser. <br><br>
Libraries Required:
1. Open3D 0.13.0
2. Tensorflow >=2.0
3. Numpy 

## RGB-D Images Dataset

This repository contains RGB-D images of staircases for static feature extraction tests.

## Section 2: Training a Machine Learning Model on [Spartan](https://dashboard.hpc.unimelb.edu.au/)
This repository details some stairs image data set and basic scripts in training a machine learning model using TensorFlow 2(TF2) and the TensorFlow2 Object Detection (TFOD) API on Spartan (Linux). There are also some helpful simple linux commands and spartan-specific commands as a refresher.

## Stair Images Dataset

This repository contains stair images along with their xml files (train and test sets) for machine learning object detection using TensorFlow Object Detection API.
- Train set: 231 images
- Test set: 25 images


## Data Preparation Scripts Created to Run On Spartan (Linux)

List of scripts:

<br/>

| Script name | Description | Remarks |
| ---        | ---    | ---    |
| dl_model | Downloads and extracts the pretrained model from the TF2 Model Zoo, creates the appropriate directories for the model, copies and amends the pipeline.config file. Also creates two .txt files that are command line instructions to train and evaluate the model | Change the PRETRAINED_MODEL_NAME and PRETRAINED_MODEL_URL accordingly to your model of choice from the [TF2  Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md). Remember to load web_proxy if running interactive job using ```module load web_proxy``` before running this script |
| create_directories_tflabel.py | Creates the appropriate directories for the TF2 workspace |  |
| create_tfrecords | Runs ```generate_tfrecord.py``` to create train.record and test.record |  |
| update_config.py | Updates the pipeline.config file located in Tensorflow/workspace/models/[model name] | Change the labels in this according to your labels. In this project we have used 'upstairs' as our label. If using non-SSD pretrained model, might have to check and make ammendments to the ```pipeline.config``` file manually |
<br/>



### Instructions

[NOTE: It is required for the user to change all of the path directories in the scripts appropriately to your own path directories.]

The following instructions build upon pre-requisite knowledge on installing the TFOD API along with its necessary dependency libraries (TF2, COCO API) and directory setup following this [YouTube Video](https://www.youtube.com/watch?v=yqkISICHH-U&t=14199s) [Credits to Nicholas Renotte]. Furthermore, it assumes access to Unimelb's High Performance Computing (HPC) System - Spartan.

The first step is to navigate to a selected directory of choice (this will be the root directory) and copy ```create_directories_tflabel.py```, ```update_config.py```, ```dl_model``` from the ```scripts``` directory in this repository.

Then, ```nano``` the ```dl_model``` file and change the ```PRETRAINED_MODEL_NAME``` and ```PRETRAINED_MODEL_URL``` corresponding to the pre-trained model of choice that you want to download from the [TensorFlow 2 Detection Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md). The ```PRETRAINED_MODEL_NAME``` will be your directory name for the pre-trained model in ```Tensorflow/workspace/pre-trained-models/``` (the folder hierarchy is shown below).

Next, run the script with
```
sh dl_model
```
This sets up the directories nicely and the structure should look like this:    
```
Tensorflow/
    └─ models/
    └─ protoc/
    └─ scripts/    
    └─ workplace/
       ├─ annotations/
       ├─ images/
       ├─ models/
       └─ pre-trained-models/
```
We will only work with the ```scripts``` and ```workplace``` directories, as ```models``` and ```protoc``` would contain the TFOD files after installation, which will not be covered. [Here](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html#tensorflow-object-detection-api-installation) is a tutorial for that.

Next, copy the train and test directory (containing train and test images & XML files respectively) from this repository into ```Tensorflow/workspace/images/```

Then, copy ```create_tfrecords``` and ```generate_tfrecord.py``` from the ```scripts``` directory in this repository into ```Tensorflow/scripts/```

Cd into ```scripts``` directory, and make the TFRecord (.record) files by running
```
sh create_tfrecords
```
This creates the train.record and test.record files.

The next step is to check the ```pipeline.config``` file in ```Tensorflow/workspace/pre-trained-models/[pre trained model of choice]``` to ensure if it is correct. Note that the ```update_config.py``` files work would for the ssd models, and likely not work for some other models. Hence, it is encouraged to change the config file manually.

Finally, the training can commence. A [pre trained model name].txt file was generated, and echoing its contents would give the command line instructions to train the model.


## Helpful Linux Commands

### Moving multiple files into directory
```
mv -t [dest] [file1] [file2] [file3]
```


## Helpful Spartan Server Commands

### Example running interactive jobs by accessing a computer node directly using ```sinteractive``` [ref](https://dashboard.hpc.unimelb.edu.au/job_submission/#:~:text=local/common/depend.-,INTERACTIVE%20JOBS,-An%20alternative%20to)

```
sinteractive --time=0:30:0 --partition=gpgpu --gres=gpu:1 --qos=gpgpumse
```

To show the available partitions
```
sinfo -s
```

To show the full queue in a particular partition (say gpgpu), the queue of current user, and particular queue status of a submitted job
```
showq -p [partition name]
showq -u
squeue --job [jobid]
```

To submit slurm script on Spartan
```
sbatch [script name].slurm
```
