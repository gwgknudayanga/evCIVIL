import h5py
import numpy as np
import os
import random

event_time_window_thres = 40000
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import glob

required_samples = 6
init_volume_max = 15000
event_time_window_thres = 15000

##### Utility functions ########

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
    #print("shape of the image is ",img.shape)
    clahe_result = clahe.apply(img)

    return clahe_result

def draw_labels_on_image(image_path,ann_array):
    
    #annotations are expected to be in coco format
    
    num_of_anns = len(ann_array)
    image = cv2.imread(image_path)
    #print("image shape ",image.shape)
    
    for idx in range(num_of_anns):
        annotation = ann_array[idx]
        class_label = int(annotation[0])
        x_min = int(annotation[1])
        y_min = int(annotation[2])
        x_max = int(annotation[3]) + x_min
        y_max = int(annotation[4]) + y_min

        #print(class_label," ",x_min," ",y_min," ",x_max," ",y_max)
        
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
    im_resized = im.resize((346,260))
    im_resized.save(output_full_fname)
    return output_full_fname

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



#### Major functions ####

def create_event_frames_and_visualize_labels(event_h5py_db,npy_label_file,labelled_output_path):
    
    event_db = h5py.File(event_h5py_db,"r")
    event_times = event_db["event_timestamp"]
    event_x = event_db["x"]
    event_y = event_db["y"]
    event_polarity = event_db["polarity"]
    events = np.vstack((event_times,event_x,event_y,event_polarity))
    events = events.T

    num_of_events = events.shape[0]


    labels = np.load(npy_label_file)
    num_of_labels = labels.shape[0]
    #print("number of labels ",num_of_labels)

    if num_of_labels > required_samples:
        selected_rows = random.sample(list(range(num_of_labels)),required_samples)
    else:
        selected_rows = list(range(num_of_labels))

    for idx in selected_rows:

        current_desired = labels[idx]
        tstamp = current_desired[0]
      
        desired_event_idxs = np.argwhere(event_times == tstamp)
        desired_event_idx = desired_event_idxs[len(desired_event_idxs) // 2].item()

        init_vol_len = init_volume_max

        min = desired_event_idx - (init_vol_len // 2)
        max = desired_event_idx + (init_vol_len // 2)

        if (min < 0) or (max > (num_of_events - 1)):
            print("not a proper event volume")
            continue

        initial_event_vol = events[min:max,:]
        desired_event_local_idx = (init_vol_len // 2)
        selected_ev_vol_length = run_area_count_method(initial_event_vol,desired_event_local_idx,satis_area_event_count = 300,event_packet_step_size = 10)

        ev_vol_start_idx = desired_event_idx - selected_ev_vol_length
        ev_vol_end_idx = desired_event_idx + selected_ev_vol_length

        desired_event_volume = events[ev_vol_start_idx:ev_vol_end_idx,:]

        desired_ev_gray_img = make_dvs_frame(desired_event_volume,height=260,width=346,color=False, clip=3, forDisplay = True)

        saved_img_path = save_images_for_matchscore_calculation(labelled_output_path,file_id = idx ,img_array = desired_ev_gray_img,isImgFrame = False)

        desired_ann_array = labels[labels[:,0] == tstamp, :]    
        
        draw_labels_on_image(saved_img_path,desired_ann_array[:,1:])

                                 
def extract_image_frames_and_visualize_labels(image_h5py_db,label_npy_file,labelled_output_path):

    if not os.path.exists(image_h5py_db):
        return
    image_db = h5py.File(image_h5py_db,"r") 
    """for key in image_db.keys():
        print(key)"""
    frame_times = image_db["frame_timestamp"]
    frames = image_db["frames"]

    labels = np.load(label_npy_file)
    num_of_labels = labels.shape[0]


    if num_of_labels > required_samples:
        selected_rows = random.sample(list(range(num_of_labels)),required_samples)
    else:
        selected_rows = list(range(num_of_labels))

    for idx in selected_rows:
        current_desired = labels[idx]
        tstamp = current_desired[0]
        
        candidate_frame_times = frame_times[(frame_times <= tstamp)]
        if not len(candidate_frame_times) > 0:
            print("Cannot find corresponding image frame ")
            continue
        desired_frame_time = np.max(candidate_frame_times)

        desired_frame_idxs = np.argwhere(frame_times == desired_frame_time)
        desired_frame_idx = desired_frame_idxs[len(desired_frame_idxs) // 2].item()

        desired_frame_array = frames[desired_frame_idx]

        #illumination correction
        desired_frame_array =  cv2_illum_correction(desired_frame_array)
        desired_frame_array = (desired_frame_array - desired_frame_array.min())/desired_frame_array.max()

        saved_img_path = save_images_for_matchscore_calculation(labelled_output_path,file_id = desired_frame_idx ,img_array = desired_frame_array,isImgFrame = True)

        desired_ann_array = labels[labels[:,0] == tstamp, :]    
        
        draw_labels_on_image(saved_img_path,desired_ann_array[:,1:])


def read_imu_data(imu_h5_file):

    imu_database = h5py.File(imu_h5_file,"r")
    # imu-samples: One measurement per line: timestamp(us) ax(g) ay(g) az(g) gx(d/s) gy(d/s) gz(d/s)
    for key in imu_database.keys():
        print("key is ",key)
        print("value is ",np.array(imu_database[key]))
        print("shape of value array is ",np.array(imu_database[key]).shape)

def main_imu():

    imu_h5_file = "/media/udayanga/data_2/shon_data/DATA/Night_outdoor/night_outdoor_18th_Nov/crack/h5_files/ros_crack_21/ros_crack_21_imu.h5"
    read_imu_data(imu_h5_file)

def main():

    base_h5_folder_path = "/media/udayanga/data_2/shon_data/DATA/Laboratory_data/At_DTU_4/crack/h5_files"
    label_visulize_base_path = "/media/udayanga/data_2/shon_data/DATA/Laboratory_data/At_DTU_4/crack/label_visualize"

    for_event_db = True

    for folder in os.listdir(base_h5_folder_path):
        
        npy_label_files = glob.glob(os.path.join(base_h5_folder_path,folder) + "/*_label.npy")
        if not len(npy_label_files) > 0:
            print("No labelling file ")
            continue
    
        npy_label_file = npy_label_files[0]
        if for_event_db and  "frame" in npy_label_file:
            npy_label_file = npy_label_files[1]

        label_output_path = label_visulize_base_path + "/" + folder

        #if folder != "ros_crack_29":
        #    continue
        if not os.path.exists(label_output_path):
            os.makedirs(label_output_path)

        if for_event_db:
            event_h5_file = glob.glob(os.path.join(base_h5_folder_path,folder) + "/*_events.h5")[0]
            print("event h5 db is ",event_h5_file)
            create_event_frames_and_visualize_labels(event_h5_file,npy_label_file,label_output_path)
        else:
            frame_h5_file = glob.glob(os.path.join(base_h5_folder_path,folder) + "/*_frames.h5")[0]
            print("frame h5 db is ",frame_h5_file)
            extract_image_frames_and_visualize_labels(frame_h5_file,npy_label_file,label_output_path)

if __name__ == "__main__":
    main()








