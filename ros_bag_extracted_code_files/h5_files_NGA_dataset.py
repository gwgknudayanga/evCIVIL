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
from skimage import data, color
import argparse

event_time_window_thres = 15000
#event_time_window >= 20000

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

def run_area_count_method(selected_events,frame_match_idx, satis_area_event_count = 250,event_packet_step_size = 250, img_width=346,img_height=260):

    count = 1
    scaled_packet_step_size = 0
    while(True):

        scaled_packet_step_size = event_packet_step_size * count
        vol_start_idx = frame_match_idx - scaled_packet_step_size
        vol_end_idx = frame_match_idx + scaled_packet_step_size
        is_an_area_fully_filled,is_tot_event_sufficient = check_area_fully_filled(selected_events[vol_start_idx:vol_end_idx,:],img_width,img_height,satis_area_event_count)
        
        if vol_end_idx >= len(selected_events):
            vol_end_idx = len(selected_events) - 1
            event_time_window = selected_events[vol_end_idx,0] - selected_events[vol_start_idx,0]
            break

        event_time_window = selected_events[vol_end_idx,0] - selected_events[vol_start_idx,0]
        """if (event_time_window > 30000):
            if not is_an_area_fully_filled:
                #print("¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤")"""

        if event_time_window >= event_time_window_thres:
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

def gammaCorrection(src, gamma):
    invGamma = 1 / gamma

    table = [((i / 255) ** invGamma) * 255 for i in range(256)]
    table = np.array(table, np.uint8)

    return cv2.LUT(src, table)


def cv2_illum_correction(src_img_path,isImg = True):

    if isImg:
        img = src_img_path
    
    else:
        img = cv2.imread(src_img_path)

    num_of_dims = len(img.shape)

    if num_of_dims > 2:
        #img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
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

def illum_correction_for_frames(src_img_path,hist_equalize = False,min_max_normalize = False,isImg = True):
    #method = 1, histogram equalization
    #method = 1, min max normalization
    #method = 2, gamma correction
    required_brightness = 125.0
    if isImg:
        img = src_img_path
    
    else:
        img = cv2.imread(src_img_path)

    num_of_dims = len(img.shape)
    if num_of_dims < 3:
        img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    else:
        if img.shape[2] == 1:
            img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

    img_hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    avg_brightness = np.sum(img_hsv[:,:,2])/(img.shape[0] * img.shape[1])
    gamma = (np.log(125)/np.log(avg_brightness)) 
    #print("gamma val ",gamma)
    img_for_gamma_correct = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    adjusted = gammaCorrection(img_for_gamma_correct, gamma=gamma)
    adjusted_hsv = cv2.cvtColor(adjusted,cv2.COLOR_BGR2HSV)
    avg_brightness_adjusted = np.sum(adjusted_hsv[:,:,2])/(adjusted_hsv.shape[0] * adjusted_hsv.shape[1])
    adjusted = cv2.cvtColor(adjusted,cv2.COLOR_BGR2GRAY)
    return adjusted


def exp_averaged_dvs_frame(events,height=260,width=346,color = True, clip=3,forDisplay =False):

    num_of_bins_t_average = 8

    events_per_bin =int(len(events) / num_of_bins_t_average)

    current_start_idx = 0

    mid = int(num_of_bins_t_average/2)
    
    mid_frame = make_dvs_frame(events[mid *events_per_bin : (mid + 1)*events_per_bin],height,width,color,clip,forDisplay)

    prev_average_ascend_frame = mid_frame
    prev_average_descend_frame = mid_frame

    for bin_num in range(mid + 1,num_of_bins_t_average,1):
        current_frame = make_dvs_frame(events[bin_num *events_per_bin : (bin_num + 1)*events_per_bin],height,width,color,clip,forDisplay)
        prev_average_ascend_frame = 0.6 * prev_average_ascend_frame + 0.4 * current_frame 

    for bin_num in range(mid,0,-1):
        current_frame = make_dvs_frame(events[(bin_num - 1) * events_per_bin : bin_num * events_per_bin],height,width,color,clip,forDisplay)
        prev_average_descend_frame = 0.6 * prev_average_descend_frame + 0.4 * current_frame
    
    frame = (prev_average_ascend_frame + prev_average_descend_frame) / 2
    
    return frame


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

def save_images_for_matchscore_calculation(modified_images_to_matchtest_path,file_id,img_array,isImgFrame,isvoxelgrid = False):
    
    modifier = "_evframe"
    if isImgFrame:
        modifier = "_frame"

    relative_out_fname = str(file_id) + modifier + ".png"
    output_full_fname = os.path.join(modified_images_to_matchtest_path,relative_out_fname)

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
    im.save(output_full_fname)

def get_img_eventimg_match_score(file_path,file_id):
    relative_img_fname = str(file_id) + "_frame.png"
    relative_evimg_fname = str(file_id) + "_evframe.png"

    get_image_full_fname = os.path.join(file_path,relative_img_fname)
    get_evimg_full_fname = os.path.join(file_path,relative_evimg_fname)

    #print("image comparison statistics ...")
    #print(get_image_full_fname)
    #print(get_evimg_full_fname)

    frame_img = cv2.imread(get_image_full_fname,0)
    event_img = cv2.imread(get_evimg_full_fname,0)

    assert (frame_img.shape == event_img.shape)

    orb = cv2.ORB_create(nfeatures=500)
    kp1, des1 = orb.detectAndCompute(frame_img, None)
    kp2, des2 = orb.detectAndCompute(event_img, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    selected = matches[:50]
    #average_match_score = (sum(selected) + 0.0) / len(selected)
    distance_list = [element.distance for element in selected]
    average_distance = sum(distance_list)/len(distance_list)
    return average_distance

def remove_low_score_img_and_evimgs(img_evimg_path,high_score_file_id_list):
    #print(high_score_file_id_list)
    for file in os.listdir(img_evimg_path):
        img_idx = file.split('_',1)[0]
        #print("img_idx ", img_idx)
        if int(img_idx) in high_score_file_id_list:
            #print("continue")
            continue
        #print("removing ",img_idx)
        os.remove(os.path.join(img_evimg_path,file))

def select_best_n_matching_pairs(img_evimg_matchscore_dict,isDebug = False,n=10):
    #based on the matching distance
    sorted_dict = sorted(img_evimg_matchscore_dict.items(), key=lambda x:x[1][0])
    selected_file_ids = list(dict(sorted_dict).keys())[:n]
    return selected_file_ids

def remove_low_score_npz_files(npz_folder_path,parent_img_name,high_score_file_id_list):
    to_keep_npz_files_list = [parent_img_name + "_" + str(id) + ".npz" for id in high_score_file_id_list]
    #print("to keep f list ",to_keep_npz_files_list)
    for npz_f_name in os.listdir(npz_folder_path):
        #print("npz list ",npz_f_name)
        if not npz_f_name in to_keep_npz_files_list:
            #print("deleting ",npz_f_name)
            full_name_to_delete_file = os.path.join(npz_folder_path,npz_f_name)
            os.remove(full_name_to_delete_file)


def create_event_frames_for_respective_frame_bins(image_h5py_db,event_h5py_db,npz_path,images_to_matchtest_path,event_vol_method=2,encoding_method=0):
    
    event_db = h5py.File(event_h5py_db,"r")
    event_times = event_db["event_timestamp"]
    event_x = event_db["x"]
    event_y = event_db["y"]
    event_polarity = event_db["polarity"]
    events = np.vstack((event_times,event_x,event_y,event_polarity))
    events = events.T
    t_stamp_min = np.min(events[:,0])
    t_stamp_max = np.max(events[:,0])       
    
    search_start_idx = 0
    
    base_name = event_h5py_db.rsplit('/',1)[1]
    #base_name = base_name.rsplit('\\',1)[1]
    base_name = base_name.rsplit("_",1)[0]
    
    npz_save_path = os.path.join(npz_path,base_name)
    if not os.path.exists(npz_save_path):
        os.makedirs(npz_save_path)
    else:
        return
    

    image_db = h5py.File(image_h5py_db,"r")
    frame_times = image_db["frame_timestamp"]
    frames = image_db["frames"]


    related_event_idx_list = []

    if len(frame_times) < 1:
        print("No any recorderd frames ")
        return
    required_event_frames_per_bin = max(90//len(frame_times),30)

    for frame_time in frame_times:
        
        #print(event_times[0:100])
        #print("frame time is ",frame_time)
        if not len(np.argwhere(event_times <= frame_time)):
            continue
        related_event_time_idx = np.max(np.argwhere(event_times <= frame_time))
        related_event_idx_list.append(related_event_time_idx)
        #print("frame vs event time ",frame_time, " ",event_times[related_event_time_idx]," ",related_event_time_idx)
        
    
    prev_event_idx = 0
    event_bin_dict = {}
    bin_start_idx = 0
    for idx,event_idx in enumerate(related_event_idx_list):
  
        if idx > 0:
            event_bin_dict[idx - 1] = (bin_start_idx, prev_event_idx + (event_idx - prev_event_idx)//2)
            bin_start_idx = prev_event_idx + (event_idx - prev_event_idx)//2
        prev_event_idx = event_idx

    #for last bin
    event_bin_dict[len(related_event_idx_list) - 1] =  (bin_start_idx,len(event_times) -1)

    for bin_key,value in event_bin_dict.items():

        current_bin_min_max = value
        bin_slice = (current_bin_min_max[1] - current_bin_min_max[0])//6
        for sample_idx in range(required_event_frames_per_bin):
            current_match_event_index = current_bin_min_max[0] + ((sample_idx + 1) * bin_slice)
        
            ev_vol_start_idx = current_match_event_index - 10000
            ev_vol_end_idx =  current_match_event_index + 10000
                
            if ev_vol_start_idx < 0 or ev_vol_end_idx > event_times.shape[0]: 
                print("continue as not a better candidate event volume")
                continue
                
            if event_vol_method == 2:
                #print("current match event idx is ",current_match_event_index)
                selected_ev_vol_length = run_area_count_method(events,current_match_event_index,satis_area_event_count = 300,event_packet_step_size = 10)
                #satis_area_event_count = 300
                #print("333333333333333")
                ev_vol_start_idx = current_match_event_index - selected_ev_vol_length
                ev_vol_end_idx = current_match_event_index + selected_ev_vol_length

            if ev_vol_start_idx < 0 or ev_vol_end_idx > event_times.shape[0]: 
                print("continue as not a better candidate event volume")
                continue
                
            desired_event_volume = events[ev_vol_start_idx:ev_vol_end_idx,:]
                
            if encoding_method == 0:
                desired_ev_gray_img = make_dvs_frame(desired_event_volume,height=260,width=346,color=False,clip=None)
                desired_ev_color_img = make_dvs_frame(desired_event_volume,height=260,width=346,color=True,clip=None,forDisplay=False)      
                
            modified_images_to_matchtest_path = os.path.join(images_to_matchtest_path,base_name)
            #print("base name is ",base_name)
            #print("modified_images_to_matchtest_path ",modified_images_to_matchtest_path)
            if not os.path.exists(modified_images_to_matchtest_path):
                os.makedirs(os.path.join(modified_images_to_matchtest_path))
            
            current_name = str(bin_key) + "_" + str(sample_idx)

            relative_npz_name = base_name + "_" + str(current_name) + ".npz"
            npz_full_name = os.path.join(npz_save_path,relative_npz_name)
            np.savez(npz_full_name,ev_gray_img = desired_ev_gray_img, ev_color_img = desired_ev_color_img,
                            events=desired_event_volume)
                
            save_images_for_matchscore_calculation(modified_images_to_matchtest_path,file_id = current_name,img_array = desired_ev_gray_img,isImgFrame = False)
        
        desired_img = frames[bin_key]
        desired_img =  cv2_illum_correction(desired_img) #illum_correction_for_frames(desired_img)
        desired_img = (desired_img - desired_img.min())/desired_img.max()
        save_images_for_matchscore_calculation(modified_images_to_matchtest_path,file_id = bin_key,img_array = desired_img,isImgFrame = True)

 
def create_event_frames_from_event_h5_file(event_h5py_db,npz_path,images_to_matchtest_path,event_vol_method=2,encoding_method=0):
    
    event_db = h5py.File(event_h5py_db,"r")
    event_times = event_db["event_timestamp"]
    event_x = event_db["x"]
    event_y = event_db["y"]
    event_polarity = event_db["polarity"]
    events = np.vstack((event_times,event_x,event_y,event_polarity))
    events = events.T
    t_stamp_min = np.min(events[:,0])
    t_stamp_max = np.max(events[:,0])       
    
    required_event_frames = 100
    search_start_idx = 0
    
    base_name = event_h5py_db.rsplit('/',1)[1]
    #base_name = base_name.rsplit('\\',1)[1]
    base_name = base_name.rsplit("_",1)[0]
    
    npz_save_path = os.path.join(npz_path,base_name)
    if not os.path.exists(npz_save_path):
        os.makedirs(npz_save_path)
    else:
        return
    
    current_idx = 0
    print("ssssssssssssssssss ")
    for f_idx in range(1,required_event_frames):
        print("11111111111111111 ")
     
        current_idx += 1
        
        current_t_candidate = np.uint32(t_stamp_min + ((t_stamp_max - t_stamp_min)/required_event_frames)*f_idx)
        
        candidates = np.argwhere(event_times[search_start_idx:] < current_t_candidate)
        max_index = np.max(candidates)
        current_match_event_index = search_start_idx + max_index
        search_start_idx = current_match_event_index
        
        ev_vol_start_idx = current_match_event_index - 10000
        ev_vol_end_idx =  current_match_event_index + 10000
        
        if ev_vol_start_idx < 0 or ev_vol_end_idx > event_times.shape[0]: 
            print("continue as not a better candidate event volume")
            continue
        
        if event_vol_method == 1: #events that falls within 10ms around the frame_time
            prefix_tstamp = current_t_candidate - 5000
            suffix_tstamp = current_t_candidate  + 5000
            ev_vol_start_idx = np.min(np.argwhere(event_times > prefix_tstamp))
            ev_vol_end_idx = np.max(np.argwhere(event_times < suffix_tstamp))
       
        if event_vol_method == 2:
            #print("current match event idx is ",current_match_event_index)
            selected_ev_vol_length = run_area_count_method(events,current_match_event_index,satis_area_event_count = 300,event_packet_step_size = 10)
            #satis_area_event_count = 300
            #print("333333333333333")
            ev_vol_start_idx = current_match_event_index - selected_ev_vol_length
            ev_vol_end_idx = current_match_event_index + selected_ev_vol_length

        if ev_vol_start_idx < 0 or ev_vol_end_idx > event_times.shape[0]: 
            print("continue as not a better candidate event volume")
            continue
        
        desired_event_volume = events[ev_vol_start_idx:ev_vol_end_idx,:]
        
        if encoding_method == 0:
            desired_ev_gray_img = make_dvs_frame(desired_event_volume,height=260,width=346,color=False,clip=None)
            desired_ev_color_img = make_dvs_frame(desired_event_volume,height=260,width=346,color=True,clip=None,forDisplay=False)      
        
        modified_images_to_matchtest_path = os.path.join(images_to_matchtest_path,base_name)
        #print("base name is ",base_name)
        #print("modified_images_to_matchtest_path ",modified_images_to_matchtest_path)
        if not os.path.exists(modified_images_to_matchtest_path):
            os.makedirs(os.path.join(modified_images_to_matchtest_path))
        
        relative_npz_name = base_name + "_" + str(current_idx) + ".npz"
        npz_full_name = os.path.join(npz_save_path,relative_npz_name)
        np.savez(npz_full_name,ev_gray_img = desired_ev_gray_img, ev_color_img = desired_ev_color_img,
                     events=desired_event_volume)
        
        save_images_for_matchscore_calculation(modified_images_to_matchtest_path,file_id = current_idx,img_array = desired_ev_gray_img,isImgFrame = False)
            
def create_image_event_pairs_from_h5py_data(image_h5py_db,event_h5py_db,npz_path,images_to_matchtest_path,event_vol_method = 2,illum_correction = True,encoding_method = 0):
    #Normally these paths are for a particular image type only: For example "npz_path = <base_path>/npz_files/crack", images_to_matchtest_path = "<base_path/match_img_evimg_files/crack"

    if not os.path.exists(image_h5py_db):
        return
    image_db = h5py.File(image_h5py_db,"r") 
    event_db = h5py.File(event_h5py_db,"r")
    for key in image_db.keys():
        print(key)
    frame_times = image_db["frame_timestamp"]
    frames = image_db["frames"]
    event_times = event_db["event_timestamp"]
    event_x = event_db["x"]
    event_y = event_db["y"]
    event_polarity = event_db["polarity"]
    #print("Maximum value of the event_x ",np.array(event_polarity).max())
    #print("Minimum value of the event_y ",np.array(event_polarity).min())
    events = np.vstack((event_times,event_x,event_y,event_polarity))
    events = events.T
    #print("After maximum x value = ",events[:,1].max()," ",events[:,2].max())
    #print("frame times ",frame_times)
    
    #print()
    
    img_evimg_matchscore_dict = {}

    search_start_idx = 0
    frame_count = 0
    
    base_name = image_h5py_db.rsplit('/',1)[1]
    #base_name = base_name.rsplit('\\',1)[1]
    base_name = base_name.rsplit("_",1)[0]

    best_npz_file_idx_list = []

    required_samples_per_recording = 50

    """if len(frame_times) > 400:
        required_samples_per_recording = 15"""

    #print("Number of frames ",len(frame_times))

    if len(frame_times) < required_samples_per_recording:
        divisor = 1

    else:
        divisor = math.ceil(len(frame_times) / required_samples_per_recording)
    divisor = 1

    #print("divisor is ",divisor)
    #print("Number of frames ", len(frame_times))

    npz_save_path = os.path.join(npz_path,base_name)
    if not os.path.exists(npz_save_path):
        os.makedirs(npz_save_path)
    else:
        return
    
    annotation_file = image_h5py_db.rsplit("_",1)[0] + "_label.npy"
    
    dumped_frames = 0
    
    #ann_array = np.load(annotation_file)
    
    #if ann_array.shape[0] > required_samples_per_recording:
    #    divisor = math.ceil(np.unique(ann_array[:,-1]).shape[0] / required_samples_per_recording)
        
    #else:
    #1divisor = 1
    
    desired_frame_count = 0

    num_of_ignored_frames = 0

    #previous_gray_img = np.zeros((260,346))

    for frame_idx,frame_time in enumerate(frame_times):
        
        frame_count += 1
        #check for whether there is any annotation, otherwise continue
        #if frame_time not in ann_array[:,6]:
        #    print("frame time NOTTTTT in the ann_array ")
        #    continue
            
        desired_frame_count += 1
        
        #print("frame time in the ann_array ")
         

        if (desired_frame_count % divisor) != 0:
            #print("continueee ")
            continue
        
        
        print("frame time is ",frame_time)
              
        print("event_times ",event_times)
        
        candidates = np.argwhere(event_times[search_start_idx:] < frame_time)
        print("number of candidates ",len(candidates))
      

        if len(candidates) == 0:
            num_of_ignored_frames += 1
            continue

        max_index = np.max(candidates)
        current_match_event_index = search_start_idx + max_index
        search_start_idx = current_match_event_index
        
        #10000 events around the matching event number
        ev_vol_start_idx = current_match_event_index - 5000
        ev_vol_end_idx =  current_match_event_index + 5000
        

        if ev_vol_start_idx < 0 or ev_vol_end_idx > event_times.shape[0]: 
            print("continue as not a better candidate event volume",event_times.shape[0]," ",current_match_event_index)
            continue

        if event_vol_method == 1: #events that falls within 10ms around the frame_time
            prefix_tstamp = frame_time - 5000
            suffix_tstamp = frame_time  + 5000
            ev_vol_start_idx = np.min(np.argwhere(event_times > prefix_tstamp))
            ev_vol_end_idx = np.max(np.argwhere(event_times < suffix_tstamp))
        
        #print("Number of events ",ev_vol_end_idx - ev_vol_start_idx)
        #print(event_x)

        if event_vol_method == 2:
            #print("current match event idx is ",current_match_event_index)
            selected_ev_vol_length = run_area_count_method(events,current_match_event_index,satis_area_event_count = 350,event_packet_step_size = 10)
            #print("333333333333333")
            ev_vol_start_idx = current_match_event_index - selected_ev_vol_length
            ev_vol_end_idx = current_match_event_index + selected_ev_vol_length

        if ev_vol_start_idx < 0 or ev_vol_end_idx > event_times.shape[0]: 
            print("continue as not a better candidate event volume")
            continue
        
        #print("Number of events ",ev_vol_end_idx - ev_vol_start_idx)
        
        desired_img = frames[frame_idx]
        if illum_correction:
            desired_img = cv2_illum_correction(desired_img) #illum_correction_for_frames(desired_img)

        desired_event_volume = events[ev_vol_start_idx:ev_vol_end_idx,:]
        #print("shape of the stacked event data 2 ",desired_event_volume.shape)

        if encoding_method == 0:

            #desired_ev_gray_img = exp_averaged_dvs_frame(desired_event_volume,height=260,width=346,color=False,clip=3)
            desired_ev_gray_img = make_dvs_frame(desired_event_volume,height=260,width=346,color=False,clip=3)
            desired_ev_color_img = make_dvs_frame(desired_event_volume,height=260,width=346,color=True,clip=3,forDisplay=False)
            dvs_dens_map = events_to_voxel_grid(desired_event_volume.copy(), num_bins=3,width=346, height=260)
        
        
        
        """fig = plt.figure()
        fig.add_subplot(1,3,1)
        plt.imshow(desired_ev_gray_img,cmap="gray")
        fig.add_subplot(1,3,2)
        plt.imshow(desired_img,cmap="gray")
        fig.add_subplot(1,3,3)
        plt.imshow(desired_ev_color_img,cmap="gray")
        fig.canvas.draw()        
        data = np.frombuffer(fig.canvas.tostring_rgb(),dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        im = Image.fromarray(data)
        im.show()"""
        #cv2.imwrite("C:/Users/gwgkn/Research_work/New_Event_data_2/h5_files/crack/sss.jpg",data)

        
        # Check for features for best matching pairs here.
        current_idx = frame_idx # + 2
        
        modified_images_to_matchtest_path = os.path.join(images_to_matchtest_path,base_name)
        #print("base name is ",base_name)
        #print("modified_images_to_matchtest_path ",modified_images_to_matchtest_path)
        if not os.path.exists(modified_images_to_matchtest_path):
            os.makedirs(os.path.join(modified_images_to_matchtest_path))

        #save_images_for_matchscore_calculation(modified_images_to_matchtest_path,file_id = current_idx,img_array = desired_img,isImgFrame = True)
        #save_images_for_matchscore_calculation(modified_images_to_matchtest_path,file_id = current_idx,img_array = desired_ev_gray_img,isImgFrame = False)

        #Here we save the matching event idx (match_event_idx) for the frame so that it will helpful for SNNs when dealing with event_volumes
            
        #avg_match_score = get_img_eventimg_match_score(file_path = modified_images_to_matchtest_path,file_id = current_idx)
        #img_evimg_matchscore_dict[current_idx] = (avg_match_score,desired_ev_gray_img,desired_ev_color_img,desired_img,desired_event_volume,current_match_event_index)

        
        #within_divisior_batch_selected_flist = select_best_n_matching_pairs(img_evimg_matchscore_dict,isDebug = False,n=1)
        #random_picked_list = random.choices(list(dict(img_evimg_matchscore_dict).keys()),k=1)
            
        #best_idx = within_divisior_batch_selected_flist[0]
        relative_npz_name = base_name + "_" + str(current_idx) + ".npz"
        npz_full_name = os.path.join(npz_save_path,relative_npz_name)


        #np.savez(npz_full_name,ev_gray_img = desired_ev_gray_img, ev_color_img = desired_ev_color_img,img_frame=desired_img,
        #            events=desired_event_volume)
        

        #desired_img = (desired_img - desired_img.min())/desired_img.max()
        #Image.fromarray(np.uint8(ee*255))

        #if previous_gray_img.any() > 0:
        #    desired_ev_gray_img = 0.8 * previous_gray_img + 0.2 * desired_ev_gray_img
        
        #previous_gray_img = desired_ev_gray_img

        save_images_for_matchscore_calculation(modified_images_to_matchtest_path,file_id = current_idx,img_array = desired_img,isImgFrame = True)
        save_images_for_matchscore_calculation(modified_images_to_matchtest_path,file_id = current_idx,img_array = desired_ev_gray_img,isImgFrame = False,isvoxelgrid=False)

        #best_npz_file_idx_list = best_npz_file_idx_list + within_divisior_batch_selected_flist
        #if not random_picked_list[0] in within_divisior_batch_selected_flist:
        #    best_npz_file_idx_list = best_npz_file_idx_list + random_picked_list
            
        #print("elements in npz best id list ",len(best_npz_file_idx_list))
        #remove_low_score_npz_files(npz_folder_path = npz_save_path,parent_img_name = base_name,high_score_file_id_list = best_npz_file_idx_list)
        #img_evimg_matchscore_dict.clear()


    print("NUmber of ignored frames ", num_of_ignored_frames)
    #remove_low_score_img_and_evimgs(img_evimg_path = modified_images_to_matchtest_path,high_score_file_id_list = best_npz_file_idx_list)   
    #print("Finishinggggggggggggggggggggggggggggggggggg")
    """best_npz_file_idx_list = select_best_n_matching_pairs(img_evimg_matchscore_dict,isDebug = False,n=10) # keep only best 'n' npz files that have higher matching score .
    #if isDebug is true write down the matching pair indexs to a text file whose name is 'base_name'.
    #returns a list of file indices to keep
    remove_low_score_npz_files(npz_folder_path = npz_save_path,parent_img_name = base_name,high_score_file_id_list = best_npz_file_idx_list) #remove npz files which are not in the 'best_npz_file_idx_list'"""

def create_npz_for_NGA_from_h5_files(h5_file_base_path,npz_dest_path,image_ev_img_dest_path,is_img_event_pair = 2):

    
    for h5_data_sample in sorted(os.listdir(h5_file_base_path)):
        print("h5 data sample ",h5_data_sample)
        h5_file_path = os.path.join(h5_file_base_path,h5_data_sample)
        #print("h5 file path is ",h5_file_path)
        if is_img_event_pair != 1:
            image_db_local_name = h5_data_sample + "_frames.h5"
        event_db_local_name = h5_data_sample + "_events.h5"
        #print("image db path is ",os.path.join(h5_file_path,image_db_local_name))
        if is_img_event_pair != 1:
            image_h5py_db = os.path.join(h5_file_path,image_db_local_name)
        event_h5py_db = os.path.join(h5_file_path,event_db_local_name)
        if is_img_event_pair == 0:
            create_image_event_pairs_from_h5py_data(image_h5py_db,event_h5py_db,npz_dest_path,image_ev_img_dest_path,
                                                event_vol_method = 2,illum_correction = True,encoding_method = 0)
        elif is_img_event_pair == 1:
            create_event_frames_from_event_h5_file(event_h5py_db,npz_dest_path,image_ev_img_dest_path,
                                                   event_vol_method=2,encoding_method=0)
        
        else:
            create_event_frames_for_respective_frame_bins(image_h5py_db,event_h5py_db,npz_dest_path,image_ev_img_dest_path,event_vol_method=2,encoding_method=0)
       
      
def main():
    #h5_file_base_path = "/dtu/eumcaerotrain/data/Event_Dataset_2/h5_files/texture"
    #npz_path = "/dtu/eumcaerotrain/data/Event_Dataset_2/npz_files/texture"
    #images_evimages_path = "/dtu/eumcaerotrain/data/Event_Dataset_2/match_img_evimg_files/tex1ure/"
    #create_npz_for_NGA_from_h5_files(h5_file_base_path,npz_path,images_evimages_path)/media/udayanga/OS/Users/gwgkn/Research_work/Ros_recording/8th_december_indoor/
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5_file_base_path", type=str,default="")
    parser.add_argument("--npz_path", type=str,default="")
    parser.add_argument("--images_evimages_path", type=str,default="")


    args = parser.parse_args() 

    base_path = "/media/udayanga/data_2/Final_Data_set/To_Labelled/Event_data_6_sep_30/"
    defect_type = "spalling"
    args.h5_file_base_path = base_path + "/h5_sub_files/" + defect_type #+ "/h5_sub_files/" + defect_type
    args.npz_path = base_path + "/npz_files/" + defect_type
    args.images_evimages_path = base_path + "/images_evimages_files/" + defect_type

    #create_npz_for_NGA_from_h5_files(args.h5_file_base_path,args.npz_path,args.images_evimages_path,is_img_event_pair = True)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
    #  is_img_event_pair = 0 means both image and evimages
    #  is_img_event_pair = 1 means evframes only with random spacing+++++++++++++++++
    #   is_img_event_pair = 2 means evframes only with bins around frames

    print("h5 file path is ",base_path)
    create_npz_for_NGA_from_h5_files(args.h5_file_base_path,args.npz_path,args.images_evimages_path,is_img_event_pair = 0)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   


if __name__ == "__main__":
    main()