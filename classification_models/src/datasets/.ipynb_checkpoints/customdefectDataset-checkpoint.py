from torch.utils.data import Dataset
import torch.nn.functional as F
import torchvision.transforms as Tr
import torch
import glob
import random
import os
import warnings
import numpy as np
from PIL import Image
from PIL import ImageFile
# from event_data_augmentors import *
# from data_encoder import make_dvs_frame
# from data_visualizer import save_images_for_matchscore_calculation,draw_labels_on_image

#This method is for image-based stuff
def resize(image, size):
    #image = torch.moveaxis(image,2,0)
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image

#To conversion in-between different annotation formats for object detection
def Box2COCO(anns):
    anns[:,3] -= anns[:,1]
    anns[:,4] -= anns[:,2]
    return anns

def COCO2YOLO(anns,img_width,img_height):
    anns[:,1] = (anns[:,1] + anns[:,3]/2)/img_width
    anns[:,2] = (anns[:,2] + anns[:,4]/2)/img_height
    anns[:,3] /= img_width  
    anns[:,4] /= img_height
    return anns
    
def YOLO2BOX(anns,img_width,img_height):
    anns[:,1] = (anns[:,1] - anns[:,3]/2)*img_width
    anns[:,2] = (anns[:,2] - anns[:,4]/2)*img_height
    anns[:,3] = anns[:,1] + anns[:,3] * img_width
    anns[:,4] = anns[:,2] + anns[:,4]*img_height
    return anns

#Label array : has the coco annotation format [class_id, x_tf, y_tf, box_width, box_height]
#Imgs : the image array

visualize_data_save_folder_path_before_crop = "/media/udayanga/data_2/Event_based_classification_evaluation/Code/"
visualize_data_save_folder_path_after_crop = "/media/udayanga/data_2/Event_based_classification_evaluation/Code/visualize_crop/"
debug = False

class CustomDefectDataSet(Dataset):
    
    def __init__(self,dataset_parent_folder="",data_csv_path="",img_size=128,transform=False,mode="train",dataset_type="npz_event_based",event_encoding_method = 0):
        
        with open(data_csv_path, "r") as file:
            self.data_files = file.readlines()
        
        self.dataset_parent = dataset_parent_folder
        self.img_size = img_size
        self.transform = transform
        self.mode = mode
        self.dataset_type = dataset_type
        self.batch_count = 0
        self.batch_first = True
        self.event_encoding_method = event_encoding_method
        self.n_classes = 2
   
            
    def train_transform(self,event_array,anns,local_path):
            
        event_array, (h0, w0),(h1,w1),anns = resize_image(event_array,target_img_size=416,label_arr = anns,force_load_size=None)
        event_array,pad,anns = padToSquare(event_array,anns)
        
        """if random.random() > 0.6:
            anns[:,3] += anns[:,1]
            anns[:,4] += anns[:,2]
            event_array, anns = random_affine(event_array, labels=anns, degrees=10, translate=0.1, scale=0.1, shear=10, new_shape=416)
            #General Flip augs : For this input should be yolo format
            anns = Box2COCO(anns)

            #Let's use only cropping here as augment """
        
        #Initially try with only two types of augmentations lr and ud flipping

        anns = COCO2YOLO(anns,img_width=event_array.shape[1],img_height=event_array.shape[0])
        event_array,anns = general_augment(event_array, anns)
        x_max = event_array[:,1].max()
        y_max = event_array[:,2].max()

        if debug == True:

            color_frame = make_dvs_frame(event_array, height=y_max, width=x_max, color=True, clip=3,forDisplay = True)
            to_save_png_name = os.path.join(visualize_data_save_folder_path_before_crop,local_path + ".png")
            saved_image_path = save_images_for_matchscore_calculation(to_save_png_name,color_frame,isImgFrame = False,isvoxelgrid = False)
            draw_and_save_output_image(saved_image_path,anns)

        return event_array,anns
        
    def test_transform(self,event_array,ann_array):

        event_array, (h0, w0),(h1,w1),ann_array = resize_image(event_array,target_img_size=416,label_arr = ann_array ,force_load_size=None)
        event_array,pad,ann_array = padToSquare(event_array,ann_array)
        ann_array = COCO2YOLO(ann_array,img_width=event_array.shape[1],img_height=event_array.shape[0])

        return event_array,ann_array
    
    def get_desired_tlx_tfy_brx_bry_for_for_classification_area_from_ann_array(self,ann_array):
        
        first_ann_row = ann_array[0,:]

        class_id = first_ann_row[0]
        tl_x_coord = first_ann_row[1]
        tl_y_cord = first_ann_row[2]
        widths = first_ann_row[3]
        heights = first_ann_row[4]

        br_x_coord = tl_x_coord + widths
        br_y_coord = tl_y_cord + heights

        #We consider only first annotation/bounding box
        #others we will ignore
        desired_x_min = tl_x_coord
        desired_y_min = tl_y_cord
        desired_x_max = br_x_coord
        desired_y_max = br_y_coord

        return [class_id,desired_x_min,desired_y_min,desired_x_max,desired_y_max]

    def resizeToTargetSize(self,img):
        
        #Using nearest neighbour interpolation
        img = F.interpolate(img.unsqueeze(0), size=self.img_size, mode='nearest') #This requires since cropping nin
        img = img.squeeze(0)
        return img
    
    def general_augment(self,img):
        #img should be in hwc format

        if random.random() > 0.5:
            img = torch.flip(img,dims=(1,))
        elif random.random() > 0.5:
            img = torch.flip(img,dims=(0,))

        return img
    
    def __getitem__(self, index):

        npz_path = self.data_files[index % len(self.data_files)].rstrip()
        npz_full_path = os.path.join(self.dataset_parent,npz_path)
        data_sample = np.load(npz_full_path)
        #print("current data sample is ",npz_path)
        if self.dataset_type == "npz_event_based":

            if self.event_encoding_method == 0:

                events_numpy_array = data_sample["events"]
                #first_time_stamp = events_numpy_array[0,0]
                #events_numpy_array[:,0] -= first_time_stamp #becuase of this uint74 can be torch.float (torch.float32)
                ann_array = data_sample["ann_array"]

                if self.transform:
                    local_name = npz_path.rsplit(".",1)[0]
                    if self.mode == "train":
                        #Apply train transforms
                        events_numpy_array,ann_array = self.train_transform(events_numpy_array,ann_array,local_name)
                    
                    if self.mode == "val" or self.mode == "test":
                        events_numpy_array,ann_array = self.test_transform(events_numpy_array,ann_array)
                #We consider only first annotation/bounding box
                #others we will ignore
                class_id,desired_x_min,desired_y_min,desired_x_max,desired_y_max = self.get_desired_tlx_tfy_brx_bry_for_for_classification_area_from_ann_array(ann_array)
                #Extract events for the desired volume
                temp_1 = events_numpy_array[events_numpy_array[:,1] >= desired_x_min,:]
                temp_2 = temp_1[temp_1[:,1]<= desired_x_max,:]
                temp_3 = temp_2[temp_2[:,2] >= desired_y_min,:]
                final_selected_array = temp_3[temp_3[:,2] <= desired_y_max,:]
                final_selected_array = final_selected_array[:,[0,3,1,2]]
            

            elif self.event_encoding_method == 1:
                
                #Here we have coco anns

                img = data_sample["ev_color_img"]
                width = img.shape[1]
                height = img.shape[0]

                #annotations are in coco format

                ann_array = data_sample["ann_array"]
                ann_array = ann_array.reshape(-1,5)

                labels = ann_array[:,0]

                spall_indices = np.argwhere(labels == 1)
                if spall_indices.any(): #give priority to spallings

                    desired_ann = ann_array[spall_indices[0],:].reshape(-1,5)
                else:
                    desired_ann = ann_array[0,:].reshape(-1,5)
                
                left = int(desired_ann[:,1])
                top = int(desired_ann[:,2])
                right = int(desired_ann[:,3] + desired_ann[:,1])
                bottom = int(desired_ann[:,4] + desired_ann[:,2])

                #cropping the desired area

                img = img[top:bottom,left:right,:] # Here we assume im shape is (h,w,c)
                img = np.transpose(img,[2,0,1])
                img = torch.tensor(img,dtype=torch.float32)
                img = self.resizeToTargetSize(img) # Now we have an image resize to 128x128, No need to keep aspect ratios for classification

                if self.transform: #Apply transforms 
                    self.general_augment(img) #
                
                """to_save_png_name = os.path.join(visualize_data_save_folder_path_after_crop,npz_path + ".png")
                print("to_save_png_name ",to_save_png_name)
                temp = torch.concatenate((img,torch.zeros(1,128,128)),axis = 0).permute([1,2,0])
                saved_image_path = save_images_for_matchscore_calculation(to_save_png_name,temp.numpy(),isImgFrame = False,isvoxelgrid = False)
                print("saved_image_path ",saved_image_path)"""

                # sample = torch.tensor(img ,dtype=torch.float32)
                sample = img.clone()
                class_id = torch.tensor(desired_ann[:,0],dtype=torch.float32)
    
                #print("shape of final selected array ",padded_array.shape)
                #padded_array = padded_array.astype(np.float32)
                #padded_array = torch.tensor(padded_array,dtype=torch.float) #.permute(0,3,1,2)
                """
                #if debug:
                current_width = final_selected_array[:,1].max()
                current_height = final_selected_array[:,2].max()
                color_frame = make_dvs_frame(final_selected_array, height=current_height, width=current_width, color=True, clip=3,forDisplay = True)
    
                final_selected_array = np.transpose(final_selected_array, [2,0,1])
                padded_array,pad,label_array = padToSquare(final_selected_array,ann_array)
    
                to_save_png_name = os.path.join(visualize_data_save_folder_path_after_crop,npz_path + ".png")
                print("to_save_png_name ",to_save_png_name)
                saved_image_path = save_images_for_matchscore_calculation(to_save_png_name,color_frame,isImgFrame = False,isvoxelgrid = False)
                #draw_and_save_output_image(saved_image_path,ann_array[0,:])
                
                color_frame_tensor = torch.tensor(color_frame,dtype=torch.float).permute(2,0,1)
    
                sample = Tr.Resize((128,128), Tr.InterpolationMode.NEAREST)(color_frame_tensor)"""
    
                #sample = torch.tensor(final_selected_array.astype(np.float64),dtype=torch.float).permute(0,3,1,2)
                #sample = torch.tensor(final_selected_array.astype(np.float64))
    
                #Pad to square filter to obtain the fisheye view
    
            return npz_path,sample,class_id
 
        elif self.dataset_type == "npz_image_based":
            
            frame_array = data_sample["frame_img"]
            
            img = Image.fromarray(frame_array).convert('RGB')
            #gray_img = img.convert('L')
            gray_img_array = np.array(img)

            ann_array = data_sample["ann_array"]
            ann_array = ann_array.reshape(-1,5)

            labels = ann_array[:,0]

            spall_indices = np.argwhere(labels == 1)
            if spall_indices.any(): #give priority to spallings

                desired_ann = ann_array[spall_indices[0],:].reshape(-1,5)
            else:
                desired_ann = ann_array[0,:].reshape(-1,5)
                
            left = int(desired_ann[:,1])
            top = int(desired_ann[:,2])
            right = int(desired_ann[:,3] + desired_ann[:,1])
            bottom = int(desired_ann[:,4] + desired_ann[:,2])

            gray_img_array = gray_img_array[top:bottom,left:right] # Here we assume im shape is (h,w,c)
            gray_img_array = np.transpose(gray_img_array,[2,0,1])
            gray_img_array = torch.tensor(gray_img_array,dtype=torch.float32)
            gray_img_array = self.resizeToTargetSize(gray_img_array) # N


            if self.transform:

                self.general_augment(gray_img_array)

            class_ids = desired_ann[:,0]

            class_id = torch.tensor(desired_ann[:,0],dtype=torch.float32)

            return npz_path,gray_img_array ,class_id #torch.tensor(anns.copy(),dtype=torch.float)
        
        else:
            #TODO : If we want to image files which exist as png/jpg instead of npz files
            return

    """def collate_fn(self, batch):

        self.batch_count += 1

        # Drop invalid images
        batch = [data for data in batch if data is not None]

        paths, imgs, targets = list(zip(*batch))

        #imgs = torch.stack([resize(img, self.img_size) for img in imgs])
            
        imgs = torch.stack([img for img in imgs])

        print("IMAGES SHAPE ",imgs.shape)

        target_tensor = torch.stack(targets)

        print("TARGET_TENSOR ",target_tensor.shape)

        
        return paths, imgs, target_tensor"""

    def __len__(self):
        return len(self.data_files)
    
    def collate_fn_for_ev_images(self,batch):
        print(batch)

    def collate_fn_for_raw_events(self,batch):

        #collate_fn, when sending raw events
        samples_output = []
        targets_output = []
        
        max_length = max([sample.shape[0] for _,sample, target in batch])
        paths = []
        for path,sample, target in batch:
            if not isinstance(sample, torch.Tensor):
                sample = torch.tensor(sample)
            if not isinstance(target, torch.Tensor):
                target = torch.tensor(target)
            if sample.is_sparse:
                sample.sparse_resize_(
                    (max_length, *sample.shape[1:]),
                    sample.sparse_dim(),
                    sample.dense_dim(),
                )
            else:
                #print("shape of the sample ",sample.shape)
                sample = torch.cat(
                    (
                        sample,
                        torch.zeros(
                            max_length - sample.shape[0],
                            *sample.shape[1:],
                            device=sample.device
                        ),
                    )
                )
            samples_output.append(sample)
            targets_output.append(target)
            paths.append(path)
        #paths_tensor = torch.stack(paths)
        samples_output = torch.stack(samples_output, 0 if self.batch_first else 1)
        if len(targets_output[0].shape) > 1:
            targets_output = torch.stack(targets_output, 0 if self.batch_first else -1) 
        else:
            targets_output = torch.tensor(targets_output, device=target.device)
        return (paths,samples_output, targets_output)