"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import os
import shutil
from argparse import ArgumentParser

import torch
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.model import SSD, ResNet, MobileNetV2,SSDLite     #SSDLite
from src.utils import generate_dboxes, Encoder, coco_classes
from src.transform import SSDTransformer
from src.loss import Loss
from src.process import train, evaluate
from src.dataset import collate_fn, CocoDataset, DefectDataset


# The confidence thres value for evaluate is mentioned in the decode_single(self, bboxes_in, scores_in, nms_threshold, max_output, max_num=200) 
#function which is in utils.py+

def load_model_for_available_keys(model,checkpoint,device):
    
    #We added this because we need to initialize the event-histogram detector 
    #with the weights of image-based detector. Here the histogram detector is 2 channel and 
    # the image-based detector is 3 channel input.
    
    for key in model.state_dict().keys():
        if key in checkpoint.keys():
            #print("key is ",key)
            if checkpoint[key].shape != model.state_dict()[key].shape:
                print("continuing as tensor shape difference ")
                continue
            model.state_dict()[key] -= model.state_dict()[key]
            model.state_dict()[key] += checkpoint[key].to(device)

def get_args():

    dataset_type = "defect"
    if dataset_type == "coco":
        data_path = "/dtu/eumcaerotrain/data/Image_Data/coco_2017_dataset/"
        ckpt_save_folder = "/dtu/eumcaerotrain/data/SSD_Statistics/ckpt/ckpt_save_path_coco/"
        log_path = "/dtu/eumcaerotrain/data/SSD_Statistics/log/log_coco"
        resume_ckpt_path = "/media/udayanga/data_2/SSD_new_github/ckpt/ckpt_event_based/SSD.pth"  #"/dtu/eumcaerotrain/data/SSD_Statistics/ckpt/ckpt_save_path_coco/best_map_0.4389120665728695.pth"   #"/media/udayanga/data_2/SSD_new_github/resume_ckpt_path"
        finetune_ckpt_path = "/dtu/eumcaerotrain/data/SSD_Statistics/ckpt/ckpt_save_path_coco/best_map_0.4389120665728695.pth" #"/media/udayanga/data_2/SSD_new_github/fine_tune/fine_tune_coco/"
        pretrained_backbone_weights_path = ""
        val_ann_gt_json_save_path = "" #"/media/udayanga/data_2/SSD_new_github/val_ann_gt_json_save_path/for_coco"
        num_classes = 81

    elif dataset_type == "defect":
        
        data_path = "/dtu/eumcaerotrain/data/latest_dataset_encode_benchmark"
        ckpt_save_folder = "/dtu/eumcaerotrain/data/SSD_Statistics/ckpt/ckpt_event_based_temp/"
        log_path = "/dtu/eumcaerotrain/data/SSD_Statistics/log/log_event_based_temp2/"
        resume_ckpt_path = "" #"/media/udayanga/data_2/SSD_new_github/resume_ckpt_path"
        finetune_ckpt_path = "/work3/kniud/object_detection/SSD_github/SSD-pytorch_2-array_based/weights/SSD.pth"
        pretrained_backbone_weights_path = "" # ""
        val_ann_gt_json_save_path = "/dtu/eumcaerotrain/data/SSD_Statistics/val_ann_gt_json_save_path/val_gt_event_based/val_gt_anns.json"
        num_classes = 3
        input_image_type = 2
        num_input_channels = 2

    parser = ArgumentParser(description="Implementation of SSD")
    parser.add_argument("--data-path", type=str, default=data_path,
                        help="the root folder of dataset")
    parser.add_argument("--save-folder", type=str, default=ckpt_save_folder,
                        help="path to folder containing model checkpoint file")
    parser.add_argument("--log-path", type=str, default=log_path)

    parser.add_argument("--model", type=str, default="ssd", choices=["ssd", "ssdlite"],
                        help="ssd-resnet50 or ssdlite-mobilenetv2")
    parser.add_argument("--epochs", type=int, default=600, help="number of total epochs to run")
    parser.add_argument("--batch-size", type=int, default=16, help="number of samples for each iteration")
    parser.add_argument("--multistep", nargs="*", type=int, default=[60, 83],
                        help="epochs at which to decay learning rate")
    parser.add_argument("--amp", action='store_true', help="Enable mixed precision training")

    parser.add_argument("--lr", type=float, default=5.2e-3, help="initial learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum argument for SGD optimizer")
    parser.add_argument("--weight-decay", type=float, default=0.0005, help="momentum argument for SGD optimizer")
    parser.add_argument("--nms-threshold", type=float, default=0.5)
    parser.add_argument("--conf_thres",type=float,default=0.1,help="")
    parser.add_argument("--num-workers", type=int, default=4)

    """parser.add_argument('--local-rank', default=0, type=int,
                        help='Used for multi-process training. Can either be manually set ' +
                             'or automatically set by using \'python -m multiproc\'.')"""
    
    parser.add_argument("--resume-ckpt-path",type=str,default =resume_ckpt_path, help="checkpoint from which we can resume the training. a complete checkpoint ")
    parser.add_argument("--fine_tune_path",type=str,default=finetune_ckpt_path,help = "previously trained model based on coco weights. ")
    parser.add_argument("--pretrained_backbone_weights_path", default=pretrained_backbone_weights_path, type=str,help="pretrained weights for the backbone(resnet/vgg) ")
    parser.add_argument("--val_ann_gt_json_save_path",default =val_ann_gt_json_save_path,type = str, help = "for coco evaluations ... ")
    parser.add_argument("--dataset_type",default=dataset_type, type=str, help="")
    parser.add_argument("--num_classes",type=int,default=num_classes,help="number of classes including background class")
    parser.add_argument("--input_image_type",type = int, default = input_image_type, help = "decide whether this is RGB image from public dataset, gray scale image from event camera or event histogram based on events.")
    parser.add_argument("--num_input_channels",type=int,default=num_input_channels,help="number of input channels to the backbone ... ")
    parser.add_argument("--train_img_file",type=str,default="night_outdoor_and_daytime_train_files_event_based.txt",help="text file which contain train image npz paths")  
    parser.add_argument("--test_img_file",type=str,default="test_files_event_based.txt",help="text file which contain test image npz paths ")
    parser.add_argument("--trained_model_to_evaluate",type=str,default="",help="path to pretrained model to evaluate ")
    
    args = parser.parse_args()
    return args


def main(opt):
    if torch.cuda.is_available():
        #torch.distributed.init_process_group(backend='nccl', init_method='env://')
        num_gpus = 1      #torch.distributed.get_world_size()
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
        num_gpus = 1
    
    train_params = {"batch_size": opt.batch_size * num_gpus,
                    "shuffle": True,
                    "drop_last": True,
                    "num_workers": opt.num_workers,
                    "collate_fn": collate_fn}

    test_params = {"batch_size": opt.batch_size * num_gpus,
                   "shuffle": False,
                   "drop_last": False,
                   "num_workers": opt.num_workers,
                   "collate_fn": collate_fn}

    if opt.model == "ssd":
        dboxes = generate_dboxes(model="ssd")
        model = SSD(backbone=ResNet(num_input_channels=opt.num_input_channels), num_classes=opt.num_classes)
        
    else:
        dboxes = generate_dboxes(model="ssdlite")
        model = SSDLite(backbone=MobileNetV2(), num_classes=opt.num_classes)

    if opt.dataset_type == "coco":
    
        train_set = CocoDataset(opt.data_path, 2017, "train", SSDTransformer(dboxes, (300, 300), val=False))
        train_loader = DataLoader(train_set, **train_params)
        test_set = CocoDataset(opt.data_path, 2017, "val", SSDTransformer(dboxes, (300, 300), val=True))
        test_loader = DataLoader(test_set, **test_params)


    elif opt.dataset_type == "defect":

        """
        train_img_file = "train_imgs.txt" 
        train_ann_file = "train:anns.txt"
        val_img_file = "val_imgs.txt"
        val_ann_file = "val_anns.txt"
        """

        train_img_file = opt.train_img_file #"night_outdoor_test_image_event_based.txt"  #"night_outdoor_and_daytime_train_files_event_based.txt" #"train_files_event_based.txt"
        val_img_file = opt.test_img_file #"night_outdoor_test_image_event_based.txt" #"test_files_event_based.txt"
        train_ann_file = ""
        val_ann_file = ""

        train_set = DefectDataset(opt.data_path,image_csv = os.path.join(opt.data_path,train_img_file),ann_csv = os.path.join(opt.data_path,train_ann_file),mode = "train",transform = SSDTransformer(dboxes, (300, 300), val=False,input_image_type = opt.input_image_type),ground_truth_val_ann_json_path = opt.val_ann_gt_json_save_path,input_image_type = opt.input_image_type)
        
        train_loader = DataLoader(train_set, **train_params)
        
        test_set = DefectDataset(opt.data_path,image_csv = os.path.join(opt.data_path,val_img_file),ann_csv = os.path.join(opt.data_path,val_ann_file),mode = "val",transform = SSDTransformer(dboxes, (300, 300), val=True,input_image_type = opt.input_image_type),ground_truth_val_ann_json_path = opt.val_ann_gt_json_save_path,input_image_type = opt.input_image_type)
        
        test_loader = DataLoader(test_set, **test_params)

    encoder = Encoder(dboxes)

    opt.lr = opt.lr * num_gpus * (opt.batch_size / 32)
    criterion = Loss(dboxes)

    optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum,
                                weight_decay=opt.weight_decay,
                                nesterov=True)
    #scheduler = MultiStepLR(optimizer=optimizer, milestones=opt.multistep, gamma=0.1)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, threshold=0.001, verbose=True)
    #scheduler = lr_scheduler.CosineAnnealingLR(optimizer, 600, eta_min=1e-5, last_epoch=-1, verbose='deprecated')


    if torch.cuda.is_available():
        model.cuda()
        criterion.cuda()

        """if opt.amp:
            from apex import amp
            from apex.parallel import DistributedDataParallel as DDP
            model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
        else:
            from torch.nn.parallel import DistributedDataParallel as DDP
        # It is recommended to use DistributedDataParallel, instead of DataParallel
        # to do multi-GPU training, even if there is only a single node.
        model = DDP(model)"""



    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)
    os.makedirs(opt.log_path)

    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)
    checkpoint_path = os.path.join(opt.save_folder, "SSD.pth")

    writer = SummaryWriter(opt.log_path)

    if opt.resume_ckpt_path != "":
        checkpoint = torch.load(opt.resume_ckpt_path)
        first_epoch = checkpoint["epoch"] + 1
        model.load_state_dict(checkpoint["model_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        optimizer.load_state_dict(checkpoint["optimizer"])
    else:
        first_epoch = 0
        if opt.trained_model_to_evaluate:
            ckpt = torch.load(opt.trained_model_to_evaluate)
            model.load_state_dict(ckpt["model_state_dict"])
        elif opt.fine_tune_path != "":
            fine_tune_ckpt = torch.load(opt.fine_tune_path)
            load_model_for_available_keys(model,fine_tune_ckpt["model_state_dict"],"cuda:0")
        elif opt.pretrained_backbone_weights_path != "":
            model.backbone.load_state_dict(checkpoint["model_state_dict"])

    current_best_map_val = 0
    current_map_val = 0
    epoch = 0

    if opt.trained_model_to_evaluate:
        current_map_val = evaluate(model, test_loader, epoch, writer, encoder, opt.nms_threshold,conf_thres = opt.conf_thres,dataset_type = opt.dataset_type,val_gt_ann_path = opt.val_ann_gt_json_save_path)
        return
    
    for epoch in range(first_epoch, opt.epochs):
        train(model, train_loader, epoch, writer, criterion, optimizer, scheduler, opt.amp,current_map_val)
        current_map_val = evaluate(model, test_loader, epoch, writer, encoder, opt.nms_threshold,conf_thres = opt.conf_thres,dataset_type = opt.dataset_type,val_gt_ann_path = opt.val_ann_gt_json_save_path)
        
        checkpoint = {"epoch": epoch,
                      "model_state_dict": model.state_dict(),
                      "optimizer": optimizer.state_dict(),
                      "scheduler": scheduler.state_dict()}

        if current_map_val > current_best_map_val:
            checkpoint_base_path = checkpoint_path.rsplit("/",1)[0]
            best_checkpoint_full_path = os.path.join(checkpoint_base_path,"best_map_" + str(current_map_val) + ".pth")
            torch.save(checkpoint,best_checkpoint_full_path)
            current_best_map_val = current_map_val

        else:        
            checkpoint = {"epoch": epoch,
                      "model_state_dict": model.state_dict(),
                      "optimizer": optimizer.state_dict(),
                      "scheduler": scheduler.state_dict()}
            torch.save(checkpoint, checkpoint_path)


if __name__ == "__main__":
    opt = get_args()
    opt.train_img_file = "night_outdoor_and_daytime_train_files_event_based_2.txt"
    opt.test_img_file = "night_outdoor_test_files_event_based.txt"
    opt.nms_thres = 0.4
    opt.conf_thres = 0.1
    opt.input_image_type = 2
    opt.num_input_channels = 2
    opt.model = "ssd"
    opt.resume_ckpt_path = ""
    opt.trained_model_to_evaluate = "/dtu/eumcaerotrain/data/SSD_Statistics/ckpt/ckpt_event_based_temp/best_map_0.866865775979742.pth" #"/dtu/eumcaerotrain/data/SSD_Statistics/ckpt/ckpt_event_based_temp/best_map_0.8674154655550038.pth"   #"/dtu/eumcaerotrain/data/SSD_Statistics/ckpt/ckpt_event_based/best_map_0.44046476490241476.pth"  #"/media/udayanga/data_2/SSD_new_github/SSD-pytorch_2-array_based/weights/mobilenet_no_lab_0.10489204807482531.pth"  #"/media/udayanga/data_2/SSD_new_github/ckpt/ckpt_event_based/best_map_0.38206401981151644.pth"  #"/media/udayanga/data_2/SSD_new_github/SSD-pytorch_2-array_based/weights/best_map_0.246_nolab_event.pth"
    main(opt)
