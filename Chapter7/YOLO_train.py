# Framework: https://github.com/AlexeyAB/darknet

# -----------------Importing Necessary Framework-----------------
!git clone https://github.com/AlexeyAB/darknet


# -------------------Creating Location Shortcut------------------
from google.colab import drive
drive.mount('/content/gdrive/')
!ln -s /content/gdrive/My\ Drive/ /mydrive


# -----------------------Compiling Darknet-----------------------
%cd darknet
!sed -i 's/OPENCV=0/OPENCV=1/' Makefile
!sed -i 's/GPU=0/GPU=1/' Makefile
!sed -i 's/CUDNN=0/CUDNN=1/' Makefile
!make


# --------------------Creating Configuration---------------------
!cp cfg/yolov3.cfg cfg/yolov3_training.cfg
!sed -i 's/batch=1/batch=64/' cfg/yolov3_training.cfg
!sed -i 's/subdivisions=1/subdivisions=16/' cfg/yolov3_training.cfg
!sed -i 's/max_batches = 500200/max_batches = 10000/' cfg/yolov3_training.cfg
!sed -i 's/steps=400000,450000/steps=8000,9000/' cfg/yolov3_training.cfg
!sed -i '610 s@classes=80@classes=5@' cfg/yolov3_training.cfg
!sed -i '696 s@classes=80@classes=5@' cfg/yolov3_training.cfg
!sed -i '783 s@classes=80@classes=5@' cfg/yolov3_training.cfg
!sed -i '603 s@filters=255@filters=30@' cfg/yolov3_training.cfg
!sed -i '689 s@filters=255@filters=30@' cfg/yolov3_training.cfg
!sed -i '776 s@filters=255@filters=30@' cfg/yolov3_training.cfg


# -----------Create Folder on Google Drive to Save Data----------
!mkdir "/content/gdrive/MyDrive/yolov3"
!echo -e "Person\nLaptop\nMug\nBicycle\nHandbag" > data/obj.names
!echo -e 'classes= 5\ntrain  = data/train.txt\nvalid  = data/test.txt\nnames = data/obj.names\nbackup = /mydrive/yolov3' > data/obj.data


# ---------------------Copy and Unzip Dataset--------------------
!cp /mydrive/yolov3/obj.zip ../
!unzip ../obj.zip -d data/


# -----------------Generate Image and Label Paths----------------
!cp /mydrive/yolov3/generate_train.py ./
!python generate_train.py


# -------------------------Training Phase------------------------
!wget http://pjreddie.com/media/files/darknet53.conv.74
!./darknet detector train data/obj.data cfg/yolov3_custom.cfg darknet53.conv.74 -dont_show
# !./darknet detector train data/obj.data cfg/yolov3_training.cfg /mydrive/yolov3/yolov3_custom_last.weights -dont_show