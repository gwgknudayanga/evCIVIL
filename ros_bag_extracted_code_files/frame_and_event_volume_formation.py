import h5py
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import shutil
import random
import math
#from skimage import data, color
import argparse
import json

#Here we assume we have all the h5 files together with annotation npy files

#For each video there will be set of npz files for event-based as well as image-based
#For event-based the content of npz file is like this.
#[ev_color_img,ev_voxel_grid,events,ann_array]

#For image-based the content of npz file is like this.
#[frame_img,ann_array]

## The ann_array for the image-based is obtained based on event-based, Hence build event-based first.

crack_time_window_thres = 20000
spalling_time_window_thres = 20000

def get_annotations_as_numpy_arr_from_json_file(json_file_path,annotation_strategy = 0):
    image_index = -1
    temp_list = []
    desired_width = 346
    desired_height = 260
    #print("json file path ",json_file_path)
    with open(json_file_path,"r") as f:
        data = json.load(f)
        image_local_name = data["imagePath"]
        
        img_width = data["imageWidth"]
        img_height = data["imageHeight"]
        
        if annotation_strategy == 0:
            image_index = image_local_name.rsplit(".",1)[0]
        else:
            temp = image_local_name.split(".",1)[0]
            image_index = temp.split("_",1)[0]
        
        num_of_labels = len(data['shapes'])
        
        for label_counter in range(num_of_labels):
            #print("label counter ",label_counter, " ",data['shapes'][label_counter])
            class_id = 0 #crack
            if data['shapes'][label_counter]['label'] == "spalling":
                class_id = 1
            bbox = np.array(data['shapes'][label_counter]['points']).reshape(-1,4)
            #print("pre bbox x1 y1 ",bbox[:,0]," ",bbox[:,1])
            #print("pre bbox x2 y2 ",bbox[:,2]," ",bbox[:,3])
            """min_x = min(bbox[:,0].copy(),bbox[:,2].copy())
            max_x = max(bbox[:,0].copy(),bbox[:,2].copy())
            min_y = min(bbox[:,1].copy(),bbox[:,3].copy())
            max_y = max(bbox[:,1].copy(),bbox[:,3].copy())"""

            min_x = np.min(bbox[:,0::2].copy(),axis = 1)
            max_x = np.max(bbox[:,0::2].copy(),axis = 1)
            min_y = np.min(bbox[:,1::2].copy(), axis = 1)
            max_y = np.max(bbox[:,1::2].copy(),axis = 1)

            #print("min_x max_x min_y max_y ",min_x," ",max_x," ",min_y," ",max_y)
            
            bbox[:,0] = min_x * ((desired_width/img_width))
            bbox[:,2] = max_x * ((desired_width/img_width))
            bbox[:,1] = min_y * ((desired_height/img_height))
            bbox[:,3] = max_y * ((desired_height/img_height))
            
            #print("post bbox x1 y1 ",bbox[:,0]," ",bbox[:,1])
            #print("post bbox x2 y2 ",bbox[:,2]," ",bbox[:,3])
            current_box_row = np.concatenate((np.array([class_id]).reshape(-1,1),bbox),axis=1)
            temp_list.append(current_box_row)

    if len(temp_list) == 0:
        return None,-1
    return np.vstack(temp_list),int(image_index)

def draw_labels_on_image(image_path,ann_array):
    
    #annotations are expected to be in coco format
    
    num_of_anns = len(ann_array)
    image = cv2.imread(image_path)
    print("image shape ",image.shape)
    
    for idx in range(num_of_anns):
        annotation = ann_array[idx]
        class_label = int(annotation[0])
        x_min = int(annotation[1])
        y_min = int(annotation[2])
        x_max = int(annotation[3]) + x_min
        y_max = int(annotation[4]) + y_min

        print(class_label," ",x_min," ",y_min," ",x_max," ",y_max)
        
        # Load the image
        top_left =  (x_min,y_min)
        bottom_right = (x_max,y_max)
        color = (0, 255, 0)  # Green color
        thickness = 2
        cv2.rectangle(image, top_left, bottom_right, color, thickness)

        label_text = ""
        
        if class_label == 0:
            label_text = "crack"
        elif class_label == 1:
            label_text = "spalling"
            
        # Define the position and font settings for the text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        font_color = (255, 0, 0)  # White color
        text_position = (top_left[0], top_left[1] - 10)

        cv2.putText(image, label_text, text_position, font, font_scale, font_color, font_thickness)
        
        # Display the image with the rectangle
    cv2.imwrite(image_path, image)

def draw_labels_for_npz_file(npz_file_path,path_to_save,isImg = False):
    data = np.load(npz_file_path)
    ev_color_image = data["ev_color_img"]
    temp1 = npz_file_path.rsplit("/",1)[1]  #.rsplit("_",1)[1]
    event_idx = temp1.rsplit(".",1)[0]
    third_plane = np.zeros((ev_color_image.shape[0],ev_color_image.shape[1],1))
    to_save_img_array = np.concatenate((ev_color_image,third_plane), axis = 2)
    #print(to_save_img_array)
    save_images_for_matchscore_calculation(path_to_save,event_idx,to_save_img_array,isImgFrame=False,isvoxelgrid = False)
    ann_array = data["ann_array"]
    #print("path to save ",path_to_save,event_idx)
    #image_path = os.path.join(path_to_save,str(event_idx) + "_evframe.png")
    draw_labels_on_image(image_path,ann_array)

def cv2_illum_correction(src_img_path,isImg = True):

    if isImg:
        img = src_img_path
    else:
        img = cv2.imread(src_img_path)

    num_of_dims = len(img.shape)

    if num_of_dims > 2:
        if img.shape[2] > 3: #this is in CHW format
            #bring to HWC format
            img = img.transpose([1,2,0])
        if img.shape[2] < 3: #this is gray scale
            img = img[:,:,0]
        else:
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        
    # Load the image
 
    # Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    print("shape of the image is ",img.shape)
    clahe_result = clahe.apply(img)

    #clahe_result = cv2.fastNlMeansDenoising(clahe_result, None, h=10, templateWindowSize=7, searchWindowSize=21)
    # Display the original and processed images side by side
    """plt.subplot(121), plt.imshow(image, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
 
    plt.subplot(122), plt.imshow(clahe_result, cmap='gray')
    plt.title('CLAHE Result'), plt.xticks([]), plt.yticks([])"""
    #cv2.imshow(clahe_result)
    return clahe_result

def events_to_voxel_grid(events, num_bins, width, height,normalize=True):
    """
    Build a voxel grid with bilinear interpolation in the time domain from a set of events.

    :param events: a [N x 4] NumPy array containing one event per row in the form: [timestamp, x, y, polarity]
    :param num_bins: number of bins in the temporal axis of the voxel grid
    :param width, height: dimensions of the voxel gridaedat_data/            h5_files/              match
    """
    
    if events.shape[0] <= 0:
        print("No events for the voxel grid ")
        return
    
    assert(events.shape[1] == 4)
    assert(num_bins > 0)
    assert(width > 0)
    assert(height > 0)
    
    events = events.copy()
    voxel_grid = np.zeros((num_bins, height, width), np.float32).ravel()

    # normalize the event timestamps so that they lie between 0 and num_bins
    last_stamp = events[-1, 0]
    first_stamp = events[0, 0]
    deltaT = last_stamp - first_stamp

    if deltaT == 0:
        deltaT = 1.0

    events[:, 0] = (num_bins - 1) * (events[:, 0] - first_stamp) / deltaT
    ts = events[:, 0]
    xs = events[:, 1].astype(np.int32)
    ys = events[:, 2].astype(np.int32)
    pols = events[:, 3].astype(np.int32)
    pols[pols == 0] = -1  # polarity should be +1 / -1

    tis = ts.astype(np.int32)
    dts = ts - tis
    vals_left = pols * (1.0 - dts)
    vals_right = pols * dts

    valid_indices = tis < num_bins
    np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * width
              + tis[valid_indices] * width * height, vals_left[valid_indices])

    valid_indices = (tis + 1) < num_bins
    np.add.at(
        voxel_grid,
        xs[valid_indices] + ys[valid_indices] * width +
        (tis[valid_indices] + 1) * width * height,
        vals_right[valid_indices])

    voxel_grid = np.reshape(voxel_grid, (num_bins, height, width))
    
    if normalize:
        nonzero_ev = (voxel_grid != 0)
        num_nonzeros = nonzero_ev.sum()
        if num_nonzeros > 0:
                # compute mean and stddev of the **nonzero** elements of the event tensor
                # we do not use PyTorch's default mean() and std() functions since it's faster
                # to compute it by hand than applying those funcs to a masked array
                mean = voxel_grid.sum() / num_nonzeros
                stddev = np.sqrt((voxel_grid ** 2).sum() / num_nonzeros - mean ** 2)
                mask = nonzero_ev.astype(np.float32)
                voxel_grid = mask * (voxel_grid - mean) / stddev

    return voxel_grid

def make_dvs_frame(events, height=260, width=346, color=True, clip=3,forDisplay = False):
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
    off_non_zero_img = img_off.flatten()[img_off.flatten() > 0]
    off_mean_activation = np.mean(off_non_zero_img)
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
        frame[..., 0] = img_on
        """img_on -= img_on.min()
        img_on /= img_on.max()"""

        frame[..., 1] = img_off
        """img_off -= img_off.min()
        img_off /= img_off.max()"""

        #print("absolute max and min = ",np.abs(frame).max())
        if forDisplay:
            third_channel = np.zeros((height,width,1))
            frame = np.concatenate((frame,third_channel),axis=2)

    else:
   
        frame = img_on - img_off
        #frame -= frame.min()
        #frame /= frame.max()
        frame /= np.abs(frame).max()

    return frame

def save_images_for_matchscore_calculation(modified_images_to_matchtest_path,file_id,img_array,isImgFrame,isvoxelgrid = False,target_size = (346,260)):

    print("path_to_save ",modified_images_to_matchtest_path)
    modifier = "_evframe"
    if isImgFrame:
        modifier = "_frame"

    relative_out_fname = str(file_id) + modifier + ".png"
    output_full_fname = os.path.join(modified_images_to_matchtest_path,relative_out_fname)

    print("relative_out_fname ",relative_out_fname)

    fig,ax = plt.subplots(frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    
    fig.add_axes(ax)
    if isvoxelgrid:
        ax.imshow(img_array[0],cmap="gray",aspect="auto")
    else:
        ax.imshow(img_array,cmap="gray",aspect="auto")

    #plt.close()
    #fig.savefig(output_full_fname)

    fig.canvas.draw()             
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    data = cv2.cvtColor(data, cv2.COLOR_RGB2GRAY)

    plt.close()
    #print("data.shape",data)
    im = Image.fromarray(data)
    im = im.resize(target_size)
    print("output full path ",output_full_fname)
    im.save(output_full_fname)
    return output_full_fname

def check_area_fully_filled(selected_events,img_width = 346,img_height = 260,satis_events_per_area_1 = 200):
    is_area_fully_filled = False

    #devisible = 32 #2^
    satis_events_per_area = satis_events_per_area_1

    width_bins = img_width >> 5
    height_bins = img_height >> 5

    #print(selected_events[:,1].astype(np.int8).shape)
    bin_x_coord = selected_events[:,1].astype(int) >> 5
    bin_y_coord = selected_events[:,2].astype(int) >> 5

    histrange = [(0,v) for v in (width_bins,height_bins)]

    area_stat_arr,_,_ = np.histogram2d(bin_x_coord,bin_y_coord,bins=(width_bins,height_bins),range = histrange)
    #print("maximum number of events ",np.max(area_stat_arr))
    
    mean_val = np.mean(area_stat_arr)
    
    diff_val = area_stat_arr - mean_val
    
    diff_exceeded = False
    diff_exceeded = np.any(diff_val > 350)
                     
    if np.max(area_stat_arr) >= satis_events_per_area:
        #print("area is fully filled")
        is_area_fully_filled = True
        
    tot_num_of_events = np.sum(area_stat_arr)
    is_tot_event_sufficient = False
    if tot_num_of_events > 10000:
        is_tot_event_sufficient = True
    return diff_exceeded,is_tot_event_sufficient

def run_area_count_method(selected_events,frame_match_idx, satis_area_event_count = 250,event_packet_step_size = 250, defect_type = "crack",img_width=346,img_height=260):

    count = 1
    scaled_packet_step_size = 0
    t_window_thres = -1
    if defect_type == "crack":
        t_window_thres = crack_time_window_thres
    elif defect_type == "spalling":
        t_window_thres = spalling_time_window_thres
    while(True):

        scaled_packet_step_size = event_packet_step_size * count
        vol_start_idx = frame_match_idx - scaled_packet_step_size
        vol_end_idx = frame_match_idx + scaled_packet_step_size
        is_an_area_fully_filled,is_tot_event_sufficient = check_area_fully_filled(selected_events[vol_start_idx:vol_end_idx,:],img_width,img_height,satis_area_event_count)

        if vol_start_idx < 0:
            vol_start_idx = 0
        
        if vol_end_idx >= len(selected_events):
            vol_end_idx = len(selected_events) - 1
            event_time_window = selected_events[vol_end_idx,0] - selected_events[vol_start_idx,0]
            break

        event_time_window = selected_events[vol_end_idx,0] - selected_events[vol_start_idx,0]
        """if (event_time_window > 30000):
            if not is_an_area_fully_filled:
                #print("¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤")"""
        
        if event_time_window >= t_window_thres:
            if is_an_area_fully_filled:
                break
                #if is_tot_event_sufficient:
                #print("breakinggggggggggggggggggg ---------------")
                #break    
        #if (event_time_window >= 20000):
            #if is_an_area_fully _filled:
            #    if is_tot_event_sufficient:
        #    break
        #if len(selected_events[vol_start_idx:vol_end_idx,:]) >= 10000:
        #    break
        
        count += 1

    return scaled_packet_step_size

def get_desired_anns(ev_vol_start_idx,ev_vol_end_idx,full_annotation_array):
    
    temp = full_annotation_array[full_annotation_array[:,0] > np.float32(ev_vol_start_idx),:]
    desired_ann_arr = temp[temp[:,0] < np.float32(ev_vol_end_idx),:]
    
    return desired_ann_arr  #ignore first colum (0th column) as it contain event indices
    
def select_event_volume_and_create_event_frame(desired_event_index,full_annotation_array,path_to_event_h5_file,destination_npz_base_path,
                                               destination_images_base_path,defect_type):

    
    #the names of image frame file, event frame file as well as the npy label file will be the same
    event_db = h5py.File(path_to_event_h5_file,"r")
    event_times = event_db["event_timestamp"]
    event_x = event_db["x"]
    event_y = event_db["y"]
    event_polarity = event_db["polarity"]
    events = np.vstack((event_times,event_x,event_y,event_polarity))
    events = events.T
    
    event_db.close()

    current_match_event_index = int(desired_event_index)
    initial_volume_half_event_count = 15000

    ev_vol_start_idx = current_match_event_index - initial_volume_half_event_count
    ev_vol_end_idx =  current_match_event_index + initial_volume_half_event_count

    if ev_vol_start_idx < 0 or ev_vol_end_idx > event_times.shape[0]:
        print("cannot frame event volume 1111 ") 
        return

    #print("current match index ",current_match_event_index)
    #print("ev_vol_start_idx ",ev_vol_start_idx)
    #print("ev_vol_end_idx ",ev_vol_end_idx)
    
    selected_ev_vol_length = run_area_count_method(events,current_match_event_index,satis_area_event_count = 300,event_packet_step_size = 10,defect_type = defect_type)

    ev_vol_start_idx = current_match_event_index - selected_ev_vol_length
    ev_vol_end_idx = current_match_event_index + selected_ev_vol_length

    if ev_vol_start_idx < 0 or ev_vol_end_idx > event_times.shape[0]: 
        print("continue as not a better candidate event volume")
        return

    desired_event_volume = events[ev_vol_start_idx:ev_vol_end_idx,:]
    
    desired_ev_gray_img = make_dvs_frame(desired_event_volume,height=260,width=346,color=False,clip=None)
    desired_ev_color_img = make_dvs_frame(desired_event_volume,height=260,width=346,color=True,clip=None,forDisplay=False)
    dvs_voxel_grid = events_to_voxel_grid(desired_event_volume.copy(), num_bins=3,width=346, height=260)

    temp1 = path_to_event_h5_file.rsplit("/",1)[0]
    video_sample_local_name = temp1.rsplit("/",1)[1]
    full_subfolder_name_for_npz_files_for_video = os.path.join(destination_npz_base_path,video_sample_local_name)
    if not os.path.exists(full_subfolder_name_for_npz_files_for_video):
        os.makedirs(full_subfolder_name_for_npz_files_for_video)

    #Get all the annotations related this this event volume

    desired_ann_array_org = get_desired_anns(ev_vol_start_idx,ev_vol_end_idx,full_annotation_array)
    desired_ann_array = desired_ann_array_org[:,1:] #ignore first column as they are just event indices
    
    #Dump the npz file

    npz_full_path_around_current_frame_of_the_video = os.path.join(full_subfolder_name_for_npz_files_for_video,str(desired_event_index) + ".npz")
    np.savez(npz_full_path_around_current_frame_of_the_video,ev_color_img = desired_ev_color_img,ev_voxel_grid = dvs_voxel_grid,events=desired_event_volume,ann_array = desired_ann_array)

    #Dump the ev_image
    
    full_sub_folder_name_for_evimages_files_for_video = os.path.join(destination_images_base_path,video_sample_local_name)
    if not os.path.exists(full_sub_folder_name_for_evimages_files_for_video):
        os.makedirs(full_sub_folder_name_for_evimages_files_for_video)
    output_file_name = save_images_for_matchscore_calculation(full_sub_folder_name_for_evimages_files_for_video,file_id = desired_event_index,img_array = desired_ev_gray_img,isImgFrame = False)

    ann_array = np.load(npz_full_path_around_current_frame_of_the_video)["ann_array"]
    print("desired ev vol start index ",ev_vol_start_idx)
    print("desired ev vol end index ",ev_vol_end_idx)
    print("desired ann array ",desired_ann_array_org)
    draw_labels_on_image(output_file_name,ann_array)


def get_intensity_frame_related_to_event_index(desired_event_index,path_to_frame_h5_file,path_to_event_h5_file,destination_images_base_path,
                                            destination_npz_base_path_frame_based,destination_npz_base_path_for_event_based):
    #convension is we find the event time stamp which is closest to the frame time stamp but the value of closest event time stamp > frame time stamp
    #Hence here we should get the frame time stamp which is closest but less than the time stamp at desired_event_index location in event h5 file.

    event_db = h5py.File(path_to_event_h5_file,"r")
    event_times = event_db["event_timestamp"]

        
    frame_db = h5py.File(path_to_frame_h5_file,"r")
    frame_times = frame_db["frame_timestamp"]
    frames = frame_db["frames"]

    desired_frame_idx = np.max(np.argwhere(frame_times < event_times[desired_event_index]))

    #desired_frame_idx = np.max(np.argwhere(frame_times < event_times[desired_event_index]))
    desired_frame = frames[desired_frame_idx]
    desired_img_illum_corrected =  cv2_illum_correction(desired_frame)

    temp1 = path_to_event_h5_file.rsplit("/",1)[0]
    video_sample_local_name = temp1.rsplit("/",1)[1]

    #Get the corresponding event npz file for this image
    corresponding_event_npz_full_name = os.path.join(destination_npz_base_path_for_event_based,video_sample_local_name,str(desired_event_index) + ".npz")

    print("corresponding_event_npz_full_name ",corresponding_event_npz_full_name)
    if not os.path.exists(corresponding_event_npz_full_name):
        return

    event_npz_data = np.load(corresponding_event_npz_full_name)
    desired_ann_array = event_npz_data["ann_array"]
    
    #Dump the npz file
    
    full_subfolder_name_for_npz_files_for_video = os.path.join(destination_npz_base_path_frame_based,video_sample_local_name)
    if not os.path.exists(full_subfolder_name_for_npz_files_for_video):
        os.makedirs(full_subfolder_name_for_npz_files_for_video)
        
    npz_full_path_around_current_frame_of_the_video = os.path.join(full_subfolder_name_for_npz_files_for_video,str(desired_event_index) + ".npz")
    
    np.savez(npz_full_path_around_current_frame_of_the_video,frame_img = desired_img_illum_corrected,ann_array = desired_ann_array)

    #Save the npy file together with npz
    
    #Dump the image
    
    full_sub_folder_name_for_images_files_for_video = os.path.join(destination_images_base_path,video_sample_local_name)
    output_file_name = save_images_for_matchscore_calculation(full_sub_folder_name_for_images_files_for_video,file_id = desired_event_index,img_array = desired_img_illum_corrected,isImgFrame = True)
    
    ann_array = np.load(npz_full_path_around_current_frame_of_the_video)["ann_array"]
    draw_labels_on_image(output_file_name,ann_array)
    
"""def get_intensity_frame_related_to_frame_index_when_event_and_image_frames_not_match(desired_frame_index,path_to_frame_h5_file,destination_images_base_path,destination_npz_base_path_frame_based,desired_ann_array_json_path):
   
    frame_db = h5py.File(path_to_frame_h5_file,"r")
    frame_times = frame_db["frame_timestamp"]
    frames = frame_db["frames"]

    desired_frame = frames[desired_frame_index]
    desired_img_illum_corrected =  cv2_illum_correction(desired_frame)

    print("path_to_frame_h5_file ",path_to_frame_h5_file)

    video_sample_local_name = path_to_frame_h5_file.rsplit("/",2)[1]
    #Dump the npz file
    
    full_subfolder_name_for_npz_files_for_video = os.path.join(destination_npz_base_path_frame_based,video_sample_local_name)
    
    print("full_subfolder_name_for_npz_files_for_video ",full_subfolder_name_for_npz_files_for_video)
    
    if not os.path.exists(full_subfolder_name_for_npz_files_for_video):
        print(full_subfolder_name_for_npz_files_for_video)
        os.makedirs(full_subfolder_name_for_npz_files_for_video)

    desired_ann_array,image_idx = get_annotations_as_numpy_arr_from_json_file(desired_ann_array_json_path,annotation_strategy = 1)

    desired_ann_array[:,3] -= desired_ann_array[:,1]
    desired_ann_array[:,4] -= desired_ann_array[:,2]
    

    print("ann array ",desired_ann_array)
    
    npz_full_path_around_current_frame_of_the_video = os.path.join(full_subfolder_name_for_npz_files_for_video,str(desired_frame_index) + "_frame.npz")

    print("npz_full_path_around_current_frame_of_the_video ",npz_full_path_around_current_frame_of_the_video)
    print("frame_img ",desired_img_illum_corrected.shape)
    print("desired_ann_array ",desired_ann_array.shape)
    
    np.savez(npz_full_path_around_current_frame_of_the_video,frame_img = desired_img_illum_corrected,ann_array = desired_ann_array)

    #Save the npy file together with npz
    
    #Dump the image
    
    full_sub_folder_name_for_images_files_for_video = os.path.join(destination_images_base_path,video_sample_local_name)
    
    if not os.path.exists(full_sub_folder_name_for_images_files_for_video):
        os.makedirs(full_sub_folder_name_for_images_files_for_video)
        
    output_file_name = save_images_for_matchscore_calculation(full_sub_folder_name_for_images_files_for_video,file_id = desired_frame_index,img_array = desired_img_illum_corrected,isImgFrame = True)

    draw_labels_on_image(output_file_name,desired_ann_array)"""

def dump_frame_npz_files_separately_for_auto_exposure(path_to_frame_h5_file,desired_frame_tstamp,full_ann_array,destination_npz_base_path_frame_based,destination_images_base_path):
    #Get the frame corresponding to the frame idx
    #Get the annotation in bbox format
    #Convert to coco format
    #Save the image together with annotation in npz file whose name is the image id.
    #save image for mathscore calcualtion 
    #Draw the labels on image 

    frame_db = h5py.File(path_to_frame_h5_file,"r")
    frame_times = frame_db["frame_timestamp"]
    frames = frame_db["frames"]

    desired_frame_index = np.argwhere( frame_times == desired_frame_tstamp)
    assert(desired_frame_index.size != 0)
    desired_frame_index = desired_frame_index.item()
    desired_frame = frames[desired_frame_index]
    desired_img_illum_corrected =  cv2_illum_correction(desired_frame)

    print("path_to_frame_h5_file ",path_to_frame_h5_file)

    video_sample_local_name = path_to_frame_h5_file.rsplit("/",2)[1]
    #Dump the npz file
    
    full_subfolder_name_for_npz_files_for_video = os.path.join(destination_npz_base_path_frame_based,video_sample_local_name)
    
    print("full_subfolder_name_for_npz_files_for_video ",full_subfolder_name_for_npz_files_for_video)
    
    if not os.path.exists(full_subfolder_name_for_npz_files_for_video):
        print(full_subfolder_name_for_npz_files_for_video)
        os.makedirs(full_subfolder_name_for_npz_files_for_video)
    
        print("ann array ",full_ann_array)
    
    npz_full_path_around_current_frame_of_the_video = os.path.join(full_subfolder_name_for_npz_files_for_video,str(desired_frame_index) + "_frame.npz")

    print("npz_full_path_around_current_frame_of_the_video ",npz_full_path_around_current_frame_of_the_video)
    print("frame_img ",desired_img_illum_corrected.shape)
    print("desired_ann_array ",full_ann_array.shape)

    desired_ann_array = full_ann_array[full_ann_array[:,0] == desired_frame_tstamp,1:]

    np.savez(npz_full_path_around_current_frame_of_the_video,frame_img = desired_img_illum_corrected,ann_array = desired_ann_array)

    #Save the npy file together with npz
    
    #Dump the image
    
    full_sub_folder_name_for_images_files_for_video = os.path.join(destination_images_base_path,video_sample_local_name)
    
    if not os.path.exists(full_sub_folder_name_for_images_files_for_video):
        os.makedirs(full_sub_folder_name_for_images_files_for_video)
        
    output_file_name = save_images_for_matchscore_calculation(full_sub_folder_name_for_images_files_for_video,file_id = desired_frame_index,img_array = desired_img_illum_corrected,isImgFrame = True)

    draw_labels_on_image(output_file_name,desired_ann_array)

    return
    


    
   





    
    
    
    
    
    
    
    
    
    

    
    
    
    
    
    
    
    

