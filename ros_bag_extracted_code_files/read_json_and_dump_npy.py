import torch
import os
import numpy as np
import h5py
import glob
import json

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
        return None,-1
    return np.vstack(temp_list),int(image_index)

def get_eventTStamp_for_frameTStamp(frameTStamp,eventTStamp_array):
    matching_event_index = np.min(np.argwhere(eventTStamp_array > frameTStamp)) #0 index based :  0 indexing
    event_time = eventTStamp_array[matching_event_index]
    #print("event id, event tstamp and frame tstamp ",matching_event_index," ",event_time," ",frameTStamp)
    return matching_event_index

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
    print("min id for {} is {}".format(images_with_json_files_folder,min_id))
    return min_id


#npy array format for a video file [event_index, class_id,tlx,tly,bbox_width,bbox_height]
def build_and_save_npy_ann_file_based_on_image_jsons_for_a_video(images_with_json_path_for_desired_video,h5_original_data_path_for_video,
                                                                 all_npz_files_path = None,annotation_strategy = 0,ros_sample_min_id = -1,is_auto_exposure_true = True,is_auto_expose_frames = False,annotate_one_strategy_consider_json_type = 0):
    
    # annotate_one_strategy_consider_json_type = 0 means _frame.json
    # annotate_one_strategy_consider_json_type = 1 means _evframe.json
    # annotate_one_strategy_consider_json_type = 2 means consider both and chose one based on availability.

    local_base_name = h5_original_data_path_for_video.rsplit("/",1)[1]
    event_file_name = local_base_name + "_events.h5"
    image_file_name = local_base_name + "_frames.h5"

    if is_auto_exposure_true:
        if is_auto_expose_frames:
            label_file_name = local_base_name + "_frames_new_label.npy"
        else:
            label_file_name = local_base_name + "_events_new_label.npy"
    else:
        label_file_name = local_base_name + "_new_label.npy"

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
    original_frames = frame_db["frames"]
    original_frames_arr = np.array(original_frames)

    frame_tstamps_arr = np.array(frame_tstamps)
    frame_db.close()

    is_one_base_indexing = False
    
    if annotation_strategy == 0: #Based on json files which contain in a image files folder only

        image_files_list = glob.glob(images_with_json_path_for_desired_video + "/*.png")
        print("Number of images ",len(image_files_list), " ",images_with_json_path_for_desired_video)
        print("Number of frame tstamps ",len(frame_tstamps_arr)," ",os.path.join(h5_original_data_path_for_video,image_file_name))
        assert(len(frame_tstamps_arr) == len(image_files_list)) #Make sure we consider all the images in the labelling process.
    elif annotation_strategy == 1: #Based on json file which contain in a image and ev images, but still we have images equal to evimages
        
        image_files_list = glob.glob(images_with_json_path_for_desired_video + "/*_frame.png")
        
        #image_files_list = sorted(image_files_list,key=key_func)
        print("Number of images ",len(image_files_list), " ",images_with_json_path_for_desired_video)
        print("Number of frame tstamps ",len(frame_tstamps_arr)," ",os.path.join(h5_original_data_path_for_video,image_file_name))
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

    print("aaaaaaaaaaaaaaa ",images_with_json_path_for_desired_video)
    if annotation_strategy == 0 or annotation_strategy == 1:
        
        json_files_list = glob.glob(images_with_json_path_for_desired_video + "/*.json")
        unique_json_list = json_files_list
        if annotation_strategy == 1:
            if annotate_one_strategy_consider_json_type == 0:
                json_files_list = glob.glob(images_with_json_path_for_desired_video + "/*_frame.json")
                unique_json_list = json_files_list
            elif annotate_one_strategy_consider_json_type == 1:
                json_files_list = glob.glob(images_with_json_path_for_desired_video + "/*_evframe.json")
                unique_json_list = json_files_list
            else:
                json_files_list = glob.glob(images_with_json_path_for_desired_video + "/*.json")
                unique_json_list = json_files_list
                unique_json_list = []
                base_part = json_files_list[0].rsplit("/",1)[0]
                temp_list = [(j_file.rsplit("/",1)[1]).rsplit("_",1)[0] for j_file in json_files_list]
                temp_list = list(set(temp_list))
                temp_list = [base_part + "/"+ name + "_evframe.json" for name in temp_list]

                for file in temp_list:
                    if os.path.exists(file):
                        unique_json_list.append(file)
                    
                    else:
                        modified_for_ev_name = file.replace("_evframe","_frame")
                        unique_json_list.append(modified_for_ev_name)

        for json_file in unique_json_list:
            print("json_file ",json_file)
            ann_numpy_arr,image_idx = get_annotations_as_numpy_arr_from_json_file(json_file,annotation_strategy)
            if image_idx == -1:
                continue
            if annotation_strategy == 1:
                if ros_sample_min_id != -1:
                    image_idx -= ros_sample_min_id
                else:   
                    image_idx -= 0

            frame_timestamp = frame_tstamps_arr[image_idx]
            
            matching_event_index = get_eventTStamp_for_frameTStamp(frame_timestamp,event_tstamps_arr)
            num_of_anns = ann_numpy_arr.shape[0]
            print("annotation array is ",np.concatenate((np.repeat(matching_event_index,num_of_anns).reshape(-1,1),ann_numpy_arr),axis=1))
            all_anns_for_npy.append(np.concatenate((np.repeat(matching_event_index,num_of_anns).reshape(-1,1),ann_numpy_arr),axis=1))
    else:

        if is_auto_exposure_true and is_auto_expose_frames:
            json_files_list = glob.glob(images_with_json_path_for_desired_video + "/*_frame.json")

            for json_file in json_files_list:
                ann_numpy_arr,image_idx = get_annotations_as_numpy_arr_from_json_file(json_file,annotation_strategy)

                if image_idx == -1:
                    print("incorrect image or image idx for json ",json_file)
                    continue
                #get the corresponding npz file

                json_local_name = json_file.rsplit("/",1)[1]
                frame_idx_str_for_this_json = json_local_name.rsplit("_",1)[0]
                frame_idx_int_for_this_json = int(frame_idx_str_for_this_json)
                #print("for ann strategy 2 the reference npz path is ",full_npz_path)
                #print("relevant_npz_file_path ",relevant_npz_file_path)
                frame = original_frames_arr[frame_idx_int_for_this_json]
                frame_time = frame_tstamps_arr[frame_idx_int_for_this_json]
    
                """relevant_event_indices = np.argwhere(event_tstamps_arr <= frame_time)
                matching_original_event_index = np.max(relevant_event_indices)"""
                num_of_anns = ann_numpy_arr.shape[0]
                print("annotation numpy array is ",ann_numpy_arr)
                print(np.repeat(frame_time,num_of_anns).reshape(-1,1))
                all_anns_for_npy.append(np.concatenate((np.repeat(frame_time,num_of_anns).reshape(-1,1),ann_numpy_arr),axis=1))
        else:
            json_files_list = glob.glob(images_with_json_path_for_desired_video + "/*_evframe.json") # change the json accordingly as with _evframe.json or _frame.json . You can use this to dump both frames and images annotations seperately when auto exposure is true.
                                                                                            # When auto exposure is true we have to annotate annotate both event histograms and frames seperately.
            for json_file in json_files_list:
                ann_numpy_arr,image_idx = get_annotations_as_numpy_arr_from_json_file(json_file,annotation_strategy)
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
                
                #mid_location = len(event_data) // 2
                #print("printing event data ",npz_data)
                #npz_mid_tstamp = event_data[mid_location,0]
                #Find the npz_mid_tstamp in original event h5 file and get the relevant index from the original event h5 files' tstamp array
                relevant_event_indices = np.argwhere((event_tstamps_arr >= minimum_tstamp) & (event_tstamps_arr <= maximum_tstamp))
                matching_original_event_index = relevant_event_indices[len(relevant_event_indices)//2]
                #print("mid event index is ",matching_original_event_index)
                num_of_anns = ann_numpy_arr.shape[0]
                print("annotation numpy array is ",ann_numpy_arr)
                print(np.repeat(matching_original_event_index,num_of_anns).reshape(-1,1))
                #np.concatenate((np.repeat(matching_original_event_index,num_of_anns).reshape(-1,1),ann_numpy_arr),axis=1)
                all_anns_for_npy.append(np.concatenate((np.repeat(matching_original_event_index,num_of_anns).reshape(-1,1),ann_numpy_arr),axis=1))

    if  len(all_anns_for_npy) > 0:
        full_npy_arr = np.concatenate(all_anns_for_npy,axis=0)
        #print("full npy arr ",full_npy_arr)
        #bbox to coco format
        full_npy_arr[:,4] -= full_npy_arr[:,2]
        full_npy_arr[:,5] -= full_npy_arr[:,3]
        label_npy_full_name = h5_original_data_path_for_video + "/" + label_file_name
        np.save(label_npy_full_name,full_npy_arr)
        print("full_npy_arr ",full_npy_arr)
    #Get the frame times for those frame_ids,
    #For those frame timestamps get the corresponding event ids.

    #Replace the frame_id in the above array with event ids.


def main():

    defect_type = "crack"
    base_path = "/media/udayanga/data_2/Final_Data_set/To_Labelled/New_night_indoor_jaer/25_feb"
    annotate_one_strategy_consider_json_type_11 = 2 # 
    path_for_dataset = base_path + "/h5_sub_files/" + defect_type
    images_with_json_path_for_all_videos = base_path + "/json_labels/" + defect_type
    referencing_npz_path =  None
    is_auto_exposure_enabled = False
    is_auto_exposure_frames = False #This is only applicable when  is_auto_exposure_enabled = True
    annotation_strategy = 1
    if annotation_strategy == 2:
        referencing_npz_path = "/media/udayanga/data_2/Final_Data_set/To_Labelled/tunnel_selected/npz_files/" + defect_type
        is_auto_exposure_enabled = True
        is_auto_exposure_frames = True #This is only applicable when  is_auto_exposure_enabled = True
    ros_sample_min_id = -1
    ros_samples = False

    data_samples = glob.glob(path_for_dataset + "/*")
    for sample in data_samples:
        
        print("sample is ",sample)
        local_folder_name = sample.rsplit("/",1)[1]
        #if local_folder_name != "t_spall_10":
        #    continue
        #if local_folder_name != "ros_crack_2":
        #    continue
            
        images_with_json_path_for_desired_video = images_with_json_path_for_all_videos + "/" + local_folder_name
        h5_original_data_path_for_desired_video = sample
        print("h5_original_data_path_for_desired_video ",h5_original_data_path_for_desired_video)
        if ros_samples:
            min_id = get_start_frame_number_for_ros_bag_extracts(images_with_json_path_for_desired_video)
        else:
            min_id = -1
        #build_and_save_npy_ann_file_based_on_image_jsons_for_a_video(images_with_json_path_for_desired_video,h5_original_data_path_for_desired_video,None,annotation_strategy,min_id)
        build_and_save_npy_ann_file_based_on_image_jsons_for_a_video(images_with_json_path_for_desired_video,h5_original_data_path_for_desired_video,referencing_npz_path,annotation_strategy,ros_sample_min_id,is_auto_exposure_enabled,is_auto_exposure_frames,annotate_one_strategy_consider_json_type=annotate_one_strategy_consider_json_type_11)

if __name__ == "__main__":
    main()
