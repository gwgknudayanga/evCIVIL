#!/bin/sh
source $HOME/miniconda3/bin/activate
conda activate udaya
### General options
### –- specify queue --
#BSUB -q gpuv100

### -- set the job Name --
#BSUB -J "udaya1"

### -- ask for number of cores (default: 1) --
#BSUB -n 4
#BSUB -M 32GB
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"

### -- set walltime limit: 20:00 -- maximum 24 hours for GPU-queues right now
#BSUB -W 20:00
# request 5GB of system-memory
#BSUB -R "rusage[mem=35GB]"

### -- Specify the output and error file. %J is the job-id --

### -- -o and -e mean append, -oo and -eo mean overwrite --
TStep=10
LRate=0.0005
Arc=vgg_13 


#BSUB -o logs/gpu_10_0.0005_vgg_13_%J.out
#BSUB -e logs/gpu_10_0.0005_vgg_13_%J.er

#conda activate udaya_snn

mkdir -p logs
##pip3 install --upgrade pip3
#pip3 install pillow --user
#pip3 install torch --user
#pip3 install h5py --user
#pip3 install numpy --user
#pip3 install comet-ml --user
#pip3 install matplotlib --user
#pip3 install pytorch-lightning --user
#git clone https://github.com/fangwei123456/spikingjelly.git
#cd spikingjelly
#python setup.py install
#cd -


#python  /work3/kniud/PUB_1/CUSTOM_DSET_ANALYSIS/Eff_2/snn_event_code/damage_classifier_new.py -test -network_output 1 -epochs 50 -architecture vgg_13 -trainCsvFile train_ideal_dark.csv -testCsvFile test_dark.csv -numTSteps 5 -numTBins 2 -initMethod k_normal #-voxelANNInitPath /work3/kniud/PUB_1/CUSTOM_DSET_ANALYSIS/ann_event_cam/4_channel/ckpt-damage-classifier-vgg/damage-classifier-vgg-epoch=39-train_acc=0.8751.ckpt  #I-no_train -pretrained /work3/kniud/Voxel_grid/final_model_voxel_cube.pth #/work3/kniud/Voxel_grid/rep_0_best_models/final_model.pth

#ifconfig > ./run_logs/abc

python ./tools/train.py --conf-file ./configs/yolov6m_finetune.py --save_dir /dtu/eumcaerotrain/data/YOLOv6_Statistics/yolov6n_image_based_all --input_img_type 1 --num_input_channels 3 --data-path ./data/defects_image_based_all.yaml > run_logs/log_image_based_all

#python  tools/train.py --resume /dtu/eumcaerotrain/data/Image_Data/yolo_nas_img_dataset/checkpoint_9th_Feb_field_evimg/597_ckpt.pt  > run_logs/log   
#./darknet detector train cfg/whill.data cfg/whill-frozen.cfg ./darknet53.conv.74 -json_port 8070 -mjpeg_port 8080 -dont_show -map  > ./run_logs/log
#python  tools/train.py --epochs 400  > ./run_logs/def  #--resume /work3/kniud/object_detection/YOLOv6/runs/train/exp1/weights/790_ckpt.pt

#./darknet detector train cfg/defects1.data cfg/defects1-frozen.cfg ./darknet53.conv.74 -json_port 8070 -mjpeg_port 8060 -dont_show -map > ./run_logs/log
