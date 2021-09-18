# Alex_ML - Stair Images Dataset, Basic Scripts for [Spartan](https://dashboard.hpc.unimelb.edu.au/), Simple Linux Commands and Spartan Commands
This repository details some image dataset and basic scripts in training a machine learning model using TensorFlow 2(TF2) and the TensorFlow2 Object Detection API.

## ToDo
[Todo: Instructions]

[Todo: navigating in spartan]




## Stair Images Dataset

This repository contains stair images along with their xml files (train and test sets) for machine learning object detection using TensorFlow Object Detection API.
Train set: 231 images
Test set: 25 images


## Spartan Scripts

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

The following instructions build upon pre-requisite knowledge and directory setup following this [YouTube Video](https://www.youtube.com/watch?v=yqkISICHH-U&t=14199s) [Credits to Nicholas Renotte]. Furthermore, it assumes access to Unimelb's High Performance Computing (HPC) System - Spartan.

Copy train and test directory into ```Tensorflow/workspace/images/```

Copy contents of ```scripts``` directory in this repository into ```Tensorflow/scripts/```

Cd into ```scripts``` directory Make TF records by running
```
sh create_tfrecords
```

Copy the scripts ```dl_model``` ```create_directories_tflabel.py``` ```update_config.py``` into the same directory that contains the TensorFlow directory.



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
