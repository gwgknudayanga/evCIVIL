"""
@author: Viet Nguyen (nhviet1009@gmail.com)
"""
import torch
from torch.utils.data import Dataset
from torchvision.datasets import CocoDetection
from torch.utils.data.dataloader import default_collate
import os
from PIL import Image
import numpy as np
import warnings
import json
import os.path as osp
import tqdm
from pathlib import Path
import cv2

def make_dvs_frame(events, height=None, width=None, color=True, clip=3,forDisplay = False):
    """Create a single frame.

    Mainly for visualization purposes

    # Arguments
    events : np.ndarray
        (t, x, y, p)
    x_pos : np.ndarray
        x positions
    """
    if height is None or width is None:
        height = events[:, 2].max()+1
        width = events[:, 1].max()+1

    histrange = [(0, v) for v in (height, width)]

    pol_on = (events[:, 3] == 1)
    pol_off = np.logical_not(pol_on)
    img_on, _, _ = np.histogram2d(
            events[pol_on, 2], events[pol_on, 1],
            bins=(height, width), range=histrange)
    img_off, _, _ = np.histogram2d(
            events[pol_off, 2], events[pol_off, 1],
            bins=(height, width), range=histrange)

    on_non_zero_img = img_on.flatten()[img_on.flatten() > 0]
    on_mean_activation = np.mean(on_non_zero_img)
    off_non_zero_img = img_off.flatten()[img_off.flatten() > 0]
    off_mean_activation = np.mean(off_non_zero_img)

    # on clip
    if clip is None:
        on_std_activation = np.std(on_non_zero_img)
        img_on = np.clip(
            img_on, on_mean_activation-3*on_std_activation,
            on_mean_activation+3*on_std_activation)
    else:
        img_on = np.clip(
            img_on, -clip, clip)

    # off clip
    
    if clip is None:
        off_std_activation = np.std(off_non_zero_img)
        img_off = np.clip(
            img_off, off_mean_activation-3*off_std_activation,
            off_mean_activation+3*off_std_activation)
    else:
        img_off = np.clip(
            img_off, -clip, clip)

    if color:

        frame = np.zeros((height, width, 2))
        #img_on /= img_on.max()
        frame[..., 0] = img_on
        """img_on -= img_on.min()
        img_on /= img_on.max()"""

        #img_off /= img_off.max()
        frame[..., 1] = img_off
        """img_off -= img_off.min()
        img_off /= img_off.max()"""

        #print("absolute max and min = ",np.abs(frame).max())
        frame /= np.abs(frame).max()
        if forDisplay:
            third_channel = np.zeros((height,width,1))
            frame = np.concatenate((frame,third_channel),axis=2)

    else:
        frame = img_on - img_off
        #frame -= frame.min()
        #frame /= frame.max()
        frame /= np.abs(frame).max()

    return frame


def COCO2YOLO(anns,img_width,img_height):
    anns[:,1] = (anns[:,1] + anns[:,3]/2)/img_width
    anns[:,2] = (anns[:,2] + anns[:,4]/2)/img_height
    anns[:,3] /= img_width  
    anns[:,4] /= img_height
    return anns

def YOLO2COCO(anns,img_width,img_height):

    anns[:,1] -= anns[:,3]/2
    anns[:,2] -= anns[:,4]/2
    anns[:,1] *= img_width
    anns[:,3] *= img_width
    anns[:,2] *= img_height
    anns[:,4] *= img_height

def generate_coco_format_labels_custom(save_path,img_paths_list,anns_for_images_list,class_names,input_image_type = 0):

        # for evaluation with pycocotools
        dataset = {"categories": [], "annotations": [], "images": []}
        dataset_size = len(img_paths_list)

        for i, class_name in enumerate(class_names):
            dataset["categories"].append(
                {"id": i, "name": class_name, "supercategory": ""}
            )

        ann_id = 0
        print(f"Convert to COCO format")
        print(f"Dataset size: {dataset_size}")

        for i in range(len(img_paths_list)):  # (img_path, info)
            print(i)

            #_,labels, name,width,height = self.get_element(i,is_event_frame)
            path = Path(img_paths_list[i])
            if input_image_type == 0:
                im = cv2.imread(img_paths_list[i])
                width = im.shape[1]
                height = im.shape[0]
            elif input_image_type == 1:
                im = np.load(img_paths_list[i])["frame_img"]
                width = im.shape[1]
                height = im.shape[0]
            elif input_image_type == 2:
                im = np.load(img_paths_list[i])["time_base_evframe"] #["ev_color_img"]
                width = im.shape[1]
                height = im.shape[0]
                
            name = path.stem
            labels = anns_for_images_list[i]

            dataset["images"].append(
                {
                    "file_name": name,
                    "id": name,
                    "width": width,
                    "height": height,
                }
            )
            if list(labels):
                for label in labels:
                    c, x, y, w, h = label[:5]
                    # convert x,y,w,h to x1,y1,x2,y2
                    x1 = (x - w / 2) * width
                    y1 = (y - h / 2) * height
                    x2 = (x + w / 2) * width
                    y2 = (y + h / 2) * height
                    # cls_id starts from 0
                    cls_id = int(c)
                    w = max(0, x2 - x1)
                    h = max(0, y2 - y1)
                    dataset["annotations"].append(
                        {
                            "area": h * w,
                            "bbox": [x1, y1, w, h],
                            "category_id": cls_id,
                            "id": ann_id,
                            "image_id": name,
                            "iscrowd": 0,
                            # mask
                            "segmentation": [],
                        }
                    )
                    ann_id += 1

        with open(save_path, "w") as f:
            json.dump(dataset, f)
            print(f"Convert to COCO format finished. Results saved in {save_path}")

def collate_fn(batch):

    items = list(zip(*batch))
    items[0] = default_collate([i for i in items[0] if torch.is_tensor(i)])
    items[1] = list([i for i in items[1] if i])
    items[2] = list([i for i in items[2] if i])
    items[3] = default_collate([i for i in items[3] if torch.is_tensor(i)])
    items[4] = default_collate([i for i in items[4] if torch.is_tensor(i)])
    return items


class CocoDataset(CocoDetection):

    def __init__(self, root, year, mode, transform=None):
        annFile = os.path.join(root, "annotations", "instances_{}{}.json".format(mode, year))
        root = os.path.join(root, "{}{}".format(mode, year))
        super(CocoDataset, self).__init__(root, annFile)
        self._load_categories()
        self.transform = transform

    def _load_categories(self):

        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x["id"])

        self.label_map = {}
        self.label_info = {}
        counter = 1
        self.label_info[0] = "background"
        for c in categories:
            self.label_map[c["id"]] = counter
            self.label_info[counter] = c["name"]
            counter += 1

    def __getitem__(self, item):
        image, target = super(CocoDataset, self).__getitem__(item)
        width, height = image.size
        boxes = []
        labels = []
        if len(target) == 0:
            return None, None, None, None, None
        for annotation in target:
            bbox = annotation.get("bbox")
            boxes.append([bbox[0] / width, bbox[1] / height, (bbox[0] + bbox[2]) / width, (bbox[1] + bbox[3]) / height])

            labels.append(self.label_map[annotation.get("category_id")])
        boxes = torch.tensor(boxes)
        labels = torch.tensor(labels)
        if self.transform is not None:
            image, (height, width), boxes, labels = self.transform(image, (height, width), boxes, labels)
        return image, target[0]["image_id"], (height, width), boxes, labels


class DefectDataset(Dataset):

    def __init__(self,root,image_csv = "",ann_csv = "",mode = "", transform = None,ground_truth_val_ann_json_path = "",input_image_type = 0):
        self.root = root
        self.transform = transform
        self.mode = mode
        self.input_image_type = input_image_type
        class_names = ["crack","spalling"]
        self.category_ids = [0,1]

        self.img_paths, self.anns = self.get_imgs_labels(root,image_csv,ann_csv,class_names,input_image_type,ground_truth_val_ann_json_path)

        """if image_csv != "":
            with open(image_csv,"r") as f:
                self.images = f.readlines()
                img_file_names = [os.path.join(self.root,file.rstrip()) for file in self.images]

        if ann_csv != "":
            with open(ann_csv,"r") as f:
                self.anns = f.readlines()
        
        #Expect these annotations in coco format
                
        for ann_file in self.anns:
                ann_file = os.path.join(self.root,ann_file.rstrip())
                ann_list_per_image = np.loadtxt(ann_file,dtype=np.float32)
                ann_list_per_image_cpy = ann_list_per_image.copy()
                ann_list_per_image_cpy = ann_list_per_image_cpy.reshape(-1,5)
                anns_list_for_images.append(ann_list_per_image_cpy)
        
        if mode == "val":
            generate_coco_format_labels_custom(ground_truth_val_ann_json_path,img_file_names,anns_list_for_images,class_names,input_image_type = 0)"""
    
    def get_imgs_labels(self,root,image_csv,ann_csv,class_names,input_image_type,ground_truth_val_ann_json_path):

        img_file_names = []
        anns_list_for_images = []

        with open(image_csv,"r") as f:
            img_file_names = f.readlines()
            img_file_names = [os.path.join(root,file.rstrip()) for file in img_file_names]
            if input_image_type > 0:
                for file_full_name in img_file_names:
                    #file_full_name = os.path.join(args.dataset_parent_folder,file.strip())
                    npz_file = np.load(file_full_name)
                    if "ann_array" not in npz_file:
                        ann_list_per_image = np.zeros((1,5))

                    else:  
                        ann_list_per_image = npz_file["ann_array"]
                    if input_image_type == 1:
                        img_width,img_height = npz_file["frame_img"].shape[1],npz_file["frame_img"].shape[0]
                    else:
                        img_width,img_height = npz_file["time_base_evframe"].shape[1],npz_file["time_base_evframe"].shape[0]
                    ann_list_per_image = COCO2YOLO(ann_list_per_image,img_width,img_height)
                    anns_list_for_images.append(ann_list_per_image)

            if input_image_type == 0: 
                with open(ann_csv,"r") as f2:
                    ann_file_names = f2.readlines()
                    for ann_file in ann_file_names:
                        ann_file = os.path.join(root,ann_file.rstrip())
                        ann_list_per_image = np.loadtxt(ann_file)
                        ann_list_per_image = ann_list_per_image.reshape(-1,5)
                        anns_list_for_images.append(ann_list_per_image)

        if self.mode == "val":
            generate_coco_format_labels_custom(ground_truth_val_ann_json_path,img_file_names,anns_list_for_images,class_names,input_image_type)
        
        return img_file_names,anns_list_for_images


    def getCatIds(self):
        return self.category_ids
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, index):
        
        boxes = []
        labels = []
        #Read images
        

        img_path = self.img_paths[index]

        """img_obj = Image.open(img_path)
        img_width = img_obj.width
        img_height = img_obj.height
        img = img_obj.convert('RGB')"""
        if self.input_image_type == 0:
            img = cv2.imread(img_path)

        elif self.input_image_type == 1:

            img = np.load(img_path)["frame_img"] #cv2.imread(path)
            img = np.array(Image.fromarray(img).convert('RGB'), dtype=np.uint8)

        elif self.input_image_type == 2:

            img = np.load(img_path)["time_base_evframe"] #cv2.imread(path)
            #print("loading ev image")
            #events = np.load(img_path)["events"]
            
            #img = make_dvs_frame(events, height=260, width=346, color=True, clip=None,forDisplay = False)
            
            
        img_height,img_width = img.shape[:2]
        
        #Read anns
        anns = self.anns[index].copy() # All are in yolo format
        YOLO2COCO(anns,img_width,img_height)
        anns = np.round(anns,3)
        for ann in anns:
            #coco to bbox and send 
            boxes.append([ann[1]/img_width,ann[2]/img_height, (ann[1] + ann[3])/img_width, (ann[2] + ann[4])/img_height])
            #print("file path with annotation is ",img_path," ",ann[0])
            modified_ann = ann[0] + 1
            labels.append(modified_ann)

        #img = torch.tensor(img)
        #img = img.permute(2,0,1)
                
        boxes = torch.tensor(boxes,dtype=torch.float)
        labels = torch.tensor(labels,dtype=torch.float)

        #print("Box shape is  111111111 ",boxes.shape)

        if self.transform is not None:

            img, (img_height, img_width), boxes, labels = self.transform(img, (img_height, img_width), boxes, labels)
        

        file_name_without_ext = Path(img_path).stem

        #print("Box shape is  22222222 ",boxes.shape)
        
        return img, file_name_without_ext , (img_height, img_width), boxes, labels











