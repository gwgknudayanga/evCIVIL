import torch
import os
import numpy as np
import h5py
import glob
import json

def get_annotations_as_numpy_arr_from_json_file(json_file_path,annotation_strategy = 0,is_for_h5 = True):
    image_index = -1
    temp_list = []

    if is_for_h5:
        desired_width = 346
        desired_height = 260
    #print("json file path ",json_file_path)
    with open(json_file_path,"r") as f:
        data = json.load(f)
        image_local_name = data["imagePath"]
        
        img_width = data["imageWidth"]
        img_height = data["imageHeight"]

        if not is_for_h5:
            desired_width = img_width
            desired_height = img_height

        if annotation_strategy == 0:
            image_index = image_local_name.rsplit(".",1)[0]
            if "_" in image_index:
                image_index = image_index.rsplit("_",1)[0]
                
        else:
            temp = image_local_name.rsplit(".",1)[0]
            image_index = temp.rsplit("_",1)[0]

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
        return None,-1,-1
    if is_for_h5:
        image_index = int(image_index)
    return np.vstack(temp_list),image_index,(desired_width,desired_height)

def get_eventTStamp_for_frameTStamp(frameTStamp,eventTStamp_array,isTStamp_return = True):

    candidates = np.argwhere(eventTStamp_array > frameTStamp)
    if not len(candidates) > 0:
        print("No corresponding event tstamp for frame tstamp ")
        return -1

    matching_event_index = np.min(candidates) #0 index based :  0 indexing
    event_time = eventTStamp_array[matching_event_index]
    #print("event id, event tstamp and frame tstamp ",matching_event_index," ",event_time," ",frameTStamp)
    retVal = event_time
    if not isTStamp_return:
        retVal = matching_event_index

    return retVal

def key_func(item):
    return int((item.rsplit("/",1)[1]).rsplit("_",1)[0])


def get_start_frame_number_for_ros_bag_extracts(images_with_json_files_folder):
    ev_frames_list = glob.glob(images_with_json_files_folder + "/*_evframe.png")
    frame_ids_list = []
    for ev_frame in ev_frames_list:
        local_name = ev_frame.rsplit("/",1)[1]
        frame_id = local_name.rsplit("_",1)[0]
        frame_ids_list.append(int(frame_id))
        
    sorted_ids_list = sorted(frame_ids_list)
    min_id = sorted_ids_list[0]
    #print("min id for {} is {}".format(images_with_json_files_folder,min_id))
    return min_id

#npy array format for a video file [event_index, class_id,tlx,tly,bbox_width,bbox_height]
def build_and_save_npy_ann_file_based_on_image_jsons_for_a_video(images_with_json_path_for_desired_video,h5_original_data_path_for_video,
                                                                 all_npz_files_path = None,annotation_strategy = 0,ros_sample_min_id = -1):
    local_base_name = h5_original_data_path_for_video.rsplit("/",1)[1]
    event_file_name = local_base_name + "_events.h5"
    image_file_name = local_base_name + "_frames.h5"
    label_file_name = local_base_name + "_label.npy"

    search_pattern = h5_original_data_path_for_video + "/*.npy"
    existing_npy_files = glob.glob(search_pattern)
    if existing_npy_files:
        for file in existing_npy_files:
            os.remove(file)

    #get all event timestamps
    
    event_db = h5py.File(os.path.join(h5_original_data_path_for_video,event_file_name),"r")
    #print("event db keys ",event_db.keys())
    event_tstamps = event_db["event_timestamp"]
    event_tstamps_arr = np.array(event_tstamps)
    #print("event_tstamps_arr ",event_tstamps_arr)
    event_db.close()

    frame_db = h5py.File(os.path.join(h5_original_data_path_for_video,image_file_name),"r")
    #print("frame db keys ",frame_db.keys())
    frame_tstamps = frame_db["frame_timestamp"]
    frame_tstamps_arr = np.array(frame_tstamps)
    frame_db.close()

    is_one_base_indexing = False

    crazy_suffix_for_json_folder = ""
    
    if annotation_strategy == 0: #Based on json files which contain in a image files folder only

        image_files_list = glob.glob(images_with_json_path_for_desired_video  + crazy_suffix_for_json_folder + "/*.png")
        #assert(len(frame_tstamps_arr) == len(image_files_list))

        #print("Number of images ",len(image_files_list), " ",images_with_json_path_for_desired_video)
        #print("Number of frame tstamps ",len(frame_tstamps_arr)," ",os.path.join(h5_original_data_path_for_video,image_file_name))
        
    elif annotation_strategy == 1: #Based on json file which contain in a image and ev images, but still we have images equal to evimages
        
        image_files_list = glob.glob(images_with_json_path_for_desired_video + "/*_evframe.png")
        
        #image_files_list = sorted(image_files_list,key=key_func)
        #print("Number of images ",len(image_files_list), " ",images_with_json_path_for_desired_video)
        #print("Number of frame tstamps ",len(frame_tstamps_arr)," ",os.path.join(h5_original_data_path_for_video,image_file_name))
        #assert(len(frame_tstamps_arr) == len(image_files_list))
        """first_img_idx = int((image_files_list[0].rsplit("/",1)[1]).rsplit("_",1)[0]) - 2
        
        if first_img_idx == ((len(frame_tstamps_arr) - len(image_files_list)) + 1):
            is_one_base_indexing = True
        elif first_img_idx == (len(frame_tstamps_arr) - len(image_files_list)):
            is_one_base_indexing = False
        else: 
            assert(len(frame_tstamps_arr) == len(image_files_list)) #Make sure we consider all the images in the labelling process."""
        
    elif annotation_strategy == 2:
        image_files_list = glob.glob(images_with_json_path_for_desired_video + "/*_evframe.png")
    
        
    #Get all the labels for the video together with frame id from jsons files and save them in a numpy array in following format
    # arr(frame_id,class_id,tlx,tly,brx,bry)
    all_anns_for_npy = []

    #print("aaaaaaaaaaaaaaa ",images_with_json_path_for_desired_video)
    if annotation_strategy == 0 or annotation_strategy == 1:
        
        json_files_list = glob.glob(images_with_json_path_for_desired_video + crazy_suffix_for_json_folder + "/*_evframe.json")
        #json_temp_files_list = glob.glob(images_with_json_path_for_desired_video + "/*_frame.json")
        if not len(json_files_list) > 0:
            return
        #print("json file list ",json_files_list)
        for json_file in json_files_list:
            print("jsn file is ",json_file)
            """if os.path.exists(json_file):
                base = json_file.rsplit("/",1)[0]
                desired_local_name = json_file.rsplit("/",1)[1].rsplit("_",1)[0] + "_frame.json"
                full_desired_name = os.path.join(base,desired_local_name)
                #print("BBBBBBBBBBBBB ",full_desired_name)
                if os.path.exists(full_desired_name):
                    json_file = full_desired_name
                    print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA ",json_file)
                else:
                    if len(json_temp_files_list) > 0:
                        continue"""
            ann_numpy_arr,image_idx,_ = get_annotations_as_numpy_arr_from_json_file(json_file,annotation_strategy,True)
            if image_idx == -1:
                continue
            if annotation_strategy == 1:
                if ros_sample_min_id != -1:
                    image_idx -= ros_sample_min_id
                else:   
                   image_idx -= 2

            frame_timestamp = frame_tstamps_arr[image_idx]
            
            matching_event_index = get_eventTStamp_for_frameTStamp(frame_timestamp,event_tstamps_arr)
            if matching_event_index == -1:
                continue
            num_of_anns = ann_numpy_arr.shape[0]
            #print("annotation array is ",np.concatenate((np.repeat(matching_event_index,num_of_anns).reshape(-1,1),ann_numpy_arr),axis=1))
            all_anns_for_npy.append(np.concatenate((np.repeat(matching_event_index,num_of_anns).reshape(-1,1),ann_numpy_arr),axis=1))
    else:
        json_files_list = glob.glob(images_with_json_path_for_desired_video + "/*_evframe.json")
        if not len(json_files_list) > 0:
            return
        for json_file in json_files_list:
            ann_numpy_arr,image_idx,_ = get_annotations_as_numpy_arr_from_json_file(json_file,annotation_strategy,True)
            if image_idx == -1:
                print("incorrect image or image idx for json ",json_file)
                continue
            #get the corresponding npz file
            relevant_npz_file_path = json_file.rsplit("/",2)[1] + "/" + json_file.rsplit("/",2)[1] + "_" + (json_file.rsplit("/",1)[1]).rsplit("_",1)[0] + ".npz"

            #all_npz_files_path.rsplit("/",1)[1]
            
            #+ "_" + (json_file.rsplit("/",1)[1]).rsplit("_",1)[0] + ".npz"
            full_npz_path = os.path.join(all_npz_files_path,relevant_npz_file_path)
            #print("for ann strategy 2 the reference npz path is ",full_npz_path)
            #print("relevant_npz_file_path ",relevant_npz_file_path)
            npz_data = np.load(full_npz_path)
            event_data = npz_data["events"]
            #get the mid event
            minimum_tstamp = event_data[2,0]
            maximum_tstamp = event_data[len(event_data) - 2,0]
            
  
            #Find the npz_mid_tstamp in original event h5 file and get the relevant index from the original event h5 files' tstamp array

            relevant_event_indices = np.argwhere((event_tstamps_arr >= minimum_tstamp) & (event_tstamps_arr <= maximum_tstamp))
            matching_original_event_index = relevant_event_indices[len(relevant_event_indices)//2]

            matching_original_time = event_tstamps_arr[matching_original_event_index]

            num_of_anns = ann_numpy_arr.shape[0]
   
            #print(np.repeat(matching_original_event_index,num_of_anns).reshape(-1,1))
            #np.concatenate((np.repeat(matching_original_event_index,num_of_anns).reshape(-1,1),ann_numpy_arr),axis=1)
            all_anns_for_npy.append(np.concatenate((np.repeat(matching_original_time,num_of_anns).reshape(-1,1),ann_numpy_arr),axis=1))

    
    full_npy_arr = np.concatenate(all_anns_for_npy,axis=0)
    #print("full npy arr ",full_npy_arr)
    #bbox to coco format
    full_npy_arr[:,4] -= full_npy_arr[:,2]
    full_npy_arr[:,5] -= full_npy_arr[:,3]
    label_npy_full_name = h5_original_data_path_for_video + "/" + label_file_name

    full_npy_arr = full_npy_arr[full_npy_arr[:,0].argsort(),:]
    np.save(label_npy_full_name,full_npy_arr)
    print("full_npy_arr ",full_npy_arr)

    #Get the frame times for those frame_ids,
    #For those frame timestamps get the corresponding event ids.

    #Replace the frame_id in the above array with event ids.


def main():
    path_for_dataset = "/media/udayanga/data_2/shon_data/DATA/Laboratory_data/At_DTU_part_2/spalling/h5_files"
    images_with_json_path_for_all_videos = "/media/udayanga/data_2/shon_data/DATA/Laboratory_data/At_DTU_part_2/spalling/json_labels/"
    referencing_npz_path = "" #"/media/udayanga/data_2/shon_data/DATA/tunnel_selected/crack/npz_files/" 
    annotation_strategy = 1
    ros_sample_min_id = -1
    ros_samples = False
    data_samples = glob.glob(path_for_dataset + "/*")
    for sample in data_samples:
        
        print("sample is ",sample)
        #if sample.rsplit("/",1)[1] != "ros_crack_29" :#or sample.rsplit("/",1)[1] == "crack_130": #   or sample.rsplit("/",1)[1] != "crack_27" :
        #    continue
        local_folder_name = sample.rsplit("/",1)[1]
        #if local_folder_name != "ros_crack_2":
        #    continue


        images_with_json_path_for_desired_video = images_with_json_path_for_all_videos + "/" + local_folder_name
        h5_original_data_path_for_desired_video = sample
        #print("h5_original_data_path_for_desired_video ",h5_original_data_path_for_desired_video)
        if ros_samples:
            min_id = get_start_frame_number_for_ros_bag_extracts(images_with_json_path_for_desired_video)
        else:
            min_id = -1
        build_and_save_npy_ann_file_based_on_image_jsons_for_a_video(images_with_json_path_for_desired_video,h5_original_data_path_for_desired_video,referencing_npz_path,annotation_strategy,min_id)
#build_and_save_npy_ann_file_based_on_image_jsons_for_a_video(images_with_json_path_for_desired_video,h5_original_data_path_for_desired_video,referencing_npz_path,2)

if __name__ == "__main__":
    main()