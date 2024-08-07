# HOTR Project User Manual

## Introduction

### HOI

- Definition: Human Object Interaction detection.  

 ### HOTR

- HOTR: End-to-End Human-Object Interaction Detection with Transformers
- Reference: [Arxiv](https://arxiv.org/abs/2104.13682 'HOTR: End-to-End Human-Object Interaction Detection with Transformers') & [Github](https://github.com/kakaobrain/HOTR 'https://github.com/kakaobrain/HOTR') - CVPR 2021
- Features:
  * One stage method. 
  * Stable and accurate output. 
  * Easy deployment. 

## How to use the official weight model to process video input and camera input

### Download project files

First, you need to download the project file from [HOTR_demo](https://github.com/Zhou1993napolun/HOTR_demo)

### Environmental Installation

1. Make sure conda is installed in your Linux system

2. Create a new environment for running the HOTR program

   ``` Linux指令
   conda create -n hotr python=3.8
   conda activate hotr 
   ```

3. Install the packages needed to run the program

   ``` 
   conda install pytorch torchvision
   conda install cython scipy
   pip install pycocotools
   pip install opencv-python
   pip install wandb
   ```

** Note that the versions of pytorch, torchevision and cuda need to correspond to each other in order to be successfully installed. Detailed version information can be viewed from the [PyTorch official website](https://pytorch.org/)*

​		The version used in this project is

​		NUC: torch = `2.0.1`, torchvision = `0.15.2a0`  
​		AI server: torch = `1.10.0`, torchvision = `0.11.0` ，cuda 10.2



### starting program

1. Some preparation is required before starting the program

   * Put the video files to be processed in the `input/` directory

   * Clear the contents of the `output/temp/` folder
   * If you need to use a camera as input, you need to set the camera's IP address and port number in the predict_video.py file in advance

   * Put the trained model in the corresponding folder (currently only the model trained by hico can be used to process videos)
     * `checkpoints/hico_det/hico_q16.pth`
     * `data/hico_20160224_det/list_action.txt`
     * `data/hico_20160224_det/corre_hico.npy`

​					**The weight file can be downloaded from [kakaobrain](https://arena.kakaocdn.net/brainrepo/hotr/hico_q16.pth 'COCO detector for HICO-DET dataset')*

2. Start video processing

   If the video is used as the input of the model:

   ```sh
   python predict_video.py --camera 2 --outputip yourip:port 
   ```

   If the camera is used as the input of the model:

   ``` sh
   python predict_video.py --outputip yourip:port --inputip yourcamip:port
   ```

3. Some additional parameters

   * yourip:port

     > Which IP address and port does flask push the processed video stream to.
     >
     > exemple : 192.168.28.171:5000

   * yourcamip:port
   
     > Web camera's IP address and port

## How to train the model yourself

### Dataset Download

The article provides 2 data sets to train the model

* V-COCO            First download the [COCO2014](https://cocodataset.org/#download) dataset, and then use the method on [VCOCO](https://github.com/s-gupta/v-coco) to separate the      V-COCO dataset
* HICO-DET       Since the University of Michigan has closed the download link, you can find the relevant download link from [GEN-VLKT](https://github.com/YueLiao/gen-vlkt)

After downloading is complete, place these files in the specified folder

``` sh
# dataset setup
HOTR
 │─ v-coco
 │   │─ data
 │   │   │─ instances_vcoco_all_2014.json
 │   │   :
 │   └─ coco
 │       │─ images
 │       │   │─ train2014
 │       │   │   │─ COCO_train2014_000000000009.jpg
 │       │   │   :
 │       │   └─ val2014
 │       │       │─ COCO_val2014_000000000042.jpg
 :       :       :
 │─ hico_20160224_det
 │       │─ list_action.txt
 │       │─ annotations
 │       │   │─ trainval_hico.json
 │       │   │─ test_hico.json
 │       │   └─ corre_hico.npy
 │       │- images
 │       │    │- train2015
 │       │    └─ test2015
```

If you want to put the dataset in another folder, you need to add a command `--data_path [:your_own_directory]/[v-coco/hico_20160224_det]` during training

### Start the training program

For training and testing, you can run on a single GPU or multiple GPUs. (Based on current testing, multiple GPUs may cause problems, so a single GPU is recommended)

``` sh
# single-gpu training / testing
$ make [vcoco/hico]_single_[train/test]

# multi-gpu training / testing (8 GPUs)
$ make [vcoco/hico]_multi_[train/test]
```

The trained weights will be saved in the KakaoBrain_HOTR_[hicodet/vcoco] folder in the checkpoint file.

Even with a small amount of vcoco data, it takes about 1.5 days to train (with a 1080 graphics card), but the training threshold can be reached in about 50 epochs. So you can change the value in the Makefile in the project to make appropriate adjustments.



## How to make your own HICO-DET dataset to train the model

### Dataset Introduction

HICO-DET is a dataset for detecting human-object interactions (HOIs) in images. It contains 47,776 images (38,118 in training set and 9,658 in test set), 600 HOI categories constructed from 80 object categories and 117 verb categories. HICO-DET provides over 150,000 annotated human-object pairs. V-COCO provides 10,346 images (2,533 for training, 2,867 for validation, and 4,946 for testing) and 16,199 person instances. Each person is annotated with 29 action categories and there are no interaction labels including objects.

For the annotations of 117 action categories and 600 HOI categories, you can refer to [HICO](https://blog.csdn.net/irving512/article/details/115122416), but 80 of the object categories are wrong. The labels of the object categories refer to the annotations of COCO, and relevant information can be found in the [COCO dataset](https://blog.csdn.net/weixin_41466947/article/details/98783700).

### Format of the HICO DET dataset

```sh
    {
        "file_name": "HICO_train2015_00000001.jpg",
        "img_id": 1,
        "annotations": [
            {
                "bbox": [
                    503.46153846153845,
                    377.53846153846155,
                    761.1538461538461,
                    1635.2307692307693
                ],
                "category_id": 1
            },
            {
                "bbox": [
                    459.23076923076917,
                    1012.1538461538461,
                    568.8461538461539,
                    1379.4615384615383
                ],
                "category_id": 25
            }
        ],
        "hoi_annotation": [
            {
                "subject_id": 0,
                "object_id": 1,
                "category_id": 37,
                "hoi_category_id": 210
            }
        ]
    },
```

* file_name: Refers to the image name
* img_id: An id that uniquely identifies the image.
* bbox: The bounding box of the quadrilateral, the corresponding variables are [xmin, ymin, xmax, ymax]
* category_id : The id of the corresponding object category, here 1 is person and 25 is backpack
* subject_id : object of the subject. This 0 represents the number of bboxes that are the object of our subject.
* object_id : The object of the object. This 1 represents the number of bboxes that are the objects of our object.
* category_id : Refers to the action between two objects. Here, the action corresponding to 37 is hold
* hoi_category_id : Refers to the HOI action label, which consists of nouns and verbs. Here, 210 corresponds to backpack hold

**Note that this is just an example. There are often many action interaction IDs or many objects in one picture. We just need to mark them one by one according to the standard.*

### How to create the HICO-DET dataset

In order to create a data set, some corresponding tools are needed. Here, LabelMe is used as an example, and the operating system used is the Windows system.

1. First install[Anaconda | The Operating System for AI](https://www.anaconda.com/)

2. Start cmd in anaconda and enter the command `pip install labelme` to download LabelMe

3. You can start it by typing `labelme` in anaconda's cmd.

4. After selecting the image directory to be annotated in the upper left corner, select the rectangle tool to annotate. Each image will have a separate json file as the annotation file for the image.

5. After the labeling is completed, we also need to process the json, because it has no labels for interactive actions and the format is incorrect. By running convert_labelme2hicode.py under my project, we can achieve batch renaming of pictures and format conversion and data labeling of json files.

6. The labeled data set is thrown into the program for training in the way of hicodet to obtain a new model

## Model comparison

### Official HICO-DET model vs. self-created HICO-DET model

In our own dataset, we annotated 10 datasets of people holding backpacks. The annotated action label is hold. The hoi label is backpack hold

After training the data 100 times, even though we have a small amount of data, compared to the official model, it detects the action labels between people and backpacks more accurately, and the reliability has increased from 0.6 to 0.85. So if we feed the model more human-computer interaction data about a specific scene, we can achieve more accurate HOI detection in this scene.

## How to deploy models on Jetson

### Version selection

The various library versions we used in this project are as follows

```
python 3.10.12
CUDA 12.2
pytorch 2.3.0
torchaudio 2.3
torchvision 0.18
```

### Jetson system flashing

1. Burning the Jetson system can be achieved by using [SDK Manager](https://developer.nvidia.com/sdk-manager) (Note: Because the orin series is the latest series, you need to use the latest version of sdkmanager to burn jetpack6.0, otherwise there will be problems (sdkmanager2.1.0))

2. When burning the system, you can only burn Linux without installing CUDA and other components. The Jetson models used in this experiment are [Jetson AGX Orin](https://developer.nvidia.com/embedded/learn/get-started-jetson-agx-orin-devkit), [Jetson Orin Nano](https://developer.nvidia.com/embedded/learn/get-started-jetson-orin-nano-devkit)

3. After the burning is complete, use the following command to install CUDA and other components

   ```sh
   sudo apt update 
   sudo apt dist-upgrade 
   sudo reboot 
   sudo apt install nvidia-jetpack
   ```

4. Configure environment variables, use the ``nano ~/.bashrc`` command to open the configuration file, and add the following two lines to the end of the file

   ```
   export PATH=/usr/local/cuda-12.2/bin${PATH:+:${PATH}}
   export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
   #其中cuda的版本根据你实际的版本更改
   ```

5. Finally, use the ``source ~/.bashrc`` command to make the changes take effect. Use the ``nvcc -V`` command. If the CUDA version information appears, the configuration is successful.

### Code environment configuration

1. In order to facilitate later management, this project uses Miniconda for environment configuration. Use the following command to install Miniconda

   ```sh
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh
   
   # 下载完成后，通过如下执行进行安装
   bash Miniconda3-latest-Linux-aarch64.sh
   
   # 安装完成后使用 
   source ~/.bashrc 
   ```

2. The libraries required in this project can be installed using the following code

   ```sh
   conda install -c conda-forge opencv
   conda install -c conda-forge pycocotools
   conda install -c conda-forge wandb
   conda install -c conda-forge scipy
   conda install -c conda-forge flask
   ```

3. Use the official link to download [PyTorch for Jetson](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048). The version used in this project is 2.3.0. After downloading, use pip install to install it

### starting program

1. Use the following command to start the program with video as input

   ```sh
   python predict_video.py --camera 2 --outputip yourip:port 
   ```

2. Use the following command to start the program with the webcam as input

   ```sh
   python predict_video.py --outputip yourip:port --inputip yourcamip:port
   ```



### Performance comparison of different Jetson models

|           | Average processing time per frame | Total processing time | RAM           | GPU temperature | CPU Temperature | CPU usage  | GPU usage             |
| :-------: | --------------------------------- | --------------------- | ------------- | --------------- | --------------- | ---------- | --------------------- |
| AGX(15W)  | 0.73 s                            | 416 s                 | 6330/62841 MB | 46.2°           | 45.2°           | 20% (4核)  | 99%, 407mhz           |
| AGX(30W)  | 0.395 s                           | 229 s                 | 6327/62841 MB | 46.2°           | 45.1°           | 10% (8核)  | 99% ; 614mhz          |
| AGX(50W)  | 0.22 s                            | 128,92 s              | 6370/62841 MB | 44.5°           | 44.782°         | 12% (12核) | 99%, 815mhz，811mhz   |
| AGX(MAXN) | 0.162 s                           | 100.90 s              | 6324/62841 MB | 50.312°         | 47.812°         | 12% (12核) | 99%, 1291mhz，1292mhz |
|  NX(10W)  | 0.72 s                            | 412.43 s              | 5801/15656 MB | 52.512°         | 51.875°         | 19%(4 核)  | 99% 610mhz            |
|  NX(15W)  | 0.69 s                            | 390.33 s              | 5864/15656 MB | 53,93°          | 54,84°          | 19%(4 核)  | 99% 611mhz            |
|  NX(25W)  | 0.62 s                            | 355.32 s              | 5833/15656 MB | 54,062°         | 54,062°         | 14%(8 核)  | 99% 407mhz            |
| NX(MAXN)  | 0.32 s                            | 187.30 s              | 5758/15656 MB | 61.281°         | 62.218°         | 13%(8核)   | 99%917mhz             |



## Deployment with Docker

* Use the following command to pull the image (make sure you have downloaded Docker Engine and Docker Nvidia toolkit)

  ​	Jetson Version

  ~~~ bash
  sudo docker pull orangelabschina/hotr:jetson
  ~~~

  ​        x86 version

  ~~~ bash
  sudo docker pull orangelabschina/hotr:x86
  ~~~

* Use the following command to start the program (the camera is used as input by default)

  ​        Jetson Version

  ~~~ bash
  sudo docker run --init --runtime=nvidia --gpus all -it orangelabschina/hotr:jetson
  ~~~

  ​        x86 version

  ~~~ bash
  sudo docker run --init --runtime=nvidia --gpus all -it orangelabschina/hotr:x86
  ~~~

* The current program also supports custom camera input or video input, and has customized the IP address of the network camera and the push IP address of the processed video.

  ​       Take the x86 version as an example

  ​       Use the camera as the input of the model (specify that the processed video is pushed to 0.0.0.0:5000 and the input is the 192.168.12.150:8090 camera)

  ~~~bash
  sudo docker run --init --runtime=nvidia --gpus all -it orangelabschina/hotr:x86 python3 predict_video.py --outputip 0.0.0.0:5000 --inputip 192.168.12.150:8090 
  ~~~

    Use the video as the input of the model (specify that the processed video is pushed to 0.0.0.0:5000)

  ~~~ bash
  sudo docker run --init --runtime=nvidia --gpus all -it orangelabschina/hotr:x86 python3 predict_video.py --outputip 0.0.0.0:5000 --camera 2
  ~~~

  















