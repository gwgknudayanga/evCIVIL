import glob
import os
import numpy as np

from frame_and_event_volume_formation import select_event_volume_and_create_event_frame,get_intensity_frame_related_to_event_index, dump_frame_npz_files_separately_for_auto_exposure
#Now we have all the annotations for each video
#So it is time to build the dataset so that we need npz files for the the samples with json.
def get_npz_files_for_the_chosen_labels_of_given_video(original_h5_datases_path_for_video,destination_npz_path_event_based,
                                                       destination_images_path,destination_npz_path_frame_based,defect_type,
                                                       is_auto_expose_enabled = True, is_auto_expose_frames = False):
    
    local_base_name = original_h5_datases_path_for_video.rsplit("/",1)[1]
    event_file_name = local_base_name + "_events.h5"
    image_file_name = local_base_name + "_frames.h5"

    if is_auto_expose_enabled:
        if is_auto_expose_frames:
            label_file_name = local_base_name + "_frames_new_label.npy"
        else:
            label_file_name = local_base_name + "_events_new_label.npy"
    else:
        label_file_name = local_base_name + "_new_label.npy"

    if not os.path.exists(os.path.join(original_h5_datases_path_for_video,label_file_name)):
        return
    label_np_arr = np.load(os.path.join(original_h5_datases_path_for_video,label_file_name))
    event_indices = label_np_arr[:,0]
    print("label array ",label_np_arr)

    for event_index in event_indices:
        #Build the event histogram/voxel grid or otherwise event volume for snn around that
        
        #What is the minimum and maximum event indices and get the labels in that range and paste in a seperate npy file
        
        
        if is_auto_expose_enabled:

            if is_auto_expose_frames:
                
                #in this case actually event index mean event time stamp
                dump_frame_npz_files_separately_for_auto_exposure(os.path.join(original_h5_datases_path_for_video,image_file_name),event_index,label_np_arr,destination_npz_path_frame_based,destination_images_path)
                
            else:

                select_event_volume_and_create_event_frame(int(event_index),label_np_arr,os.path.join(original_h5_datases_path_for_video,event_file_name),
                                                   destination_npz_path_event_based,destination_images_path,defect_type)
            #will code later
            #use the _frames from origina_json_files_path and the corresponding json labels for those labels.
            # Need to dump frames seperately
        else:
            select_event_volume_and_create_event_frame(int(event_index),label_np_arr,os.path.join(original_h5_datases_path_for_video,event_file_name),
                                                   destination_npz_path_event_based,destination_images_path,defect_type)
            
            get_intensity_frame_related_to_event_index(int(event_index),os.path.join(original_h5_datases_path_for_video,image_file_name),
                                                        os.path.join(original_h5_datases_path_for_video,event_file_name),
                                                       destination_images_path,destination_npz_path_frame_based,
                                                       destination_npz_path_event_based)
            #User can use the same labels npy file
            #provide the <image_num>_<i k>
            #Get the corresponding frames if there are 1 to 1 mapping between evframes and frames, elese obtai

def main():
    desired_temp_list = ["t_spall_10"]
    is_auto_expose_enabled = False
    is_auto_expose_frames = False
    defect_typee = "crack"

    base_path = "/media/udayanga/data_2/Final_Data_set/To_Labelled/New_night_indoor_jaer/25_feb"
    path_for_the_all_videos_with_labels = base_path + "/h5_sub_files/" + defect_typee
    all_video_folders = glob.glob(path_for_the_all_videos_with_labels + "/*")
    destination_npz_base_path_event_based = base_path + "/npz_files_event_based/" + defect_typee
    destination_npz_base_path_frame_based = base_path + "/npz_files_image_based/" + defect_typee
    destination_images_base_path = base_path + "/img_evimg_pair_files/" + defect_typee
    for h5_files_for_video_path in all_video_folders:
        print("h5_files_for_video_path ",h5_files_for_video_path)
        local_folder_name = h5_files_for_video_path.rsplit("/",1)[1]
        if is_auto_expose_enabled and is_auto_expose_frames:
            if os.path.exists(os.path.join(destination_npz_base_path_frame_based,local_folder_name)):
                print("npz file ",local_folder_name," exists ")
                continue
        elif os.path.exists(os.path.join(destination_npz_base_path_event_based,local_folder_name)):
            print("npz file ",local_folder_name," exists")
            continue
        #if  local_folder_name in desired_temp_list:
        #    continue
        #    print("local folder name ",local_folder_name)
        get_npz_files_for_the_chosen_labels_of_given_video(h5_files_for_video_path,destination_npz_base_path_event_based,
                                                           destination_images_base_path,destination_npz_base_path_frame_based,defect_typee,
                                                           is_auto_expose_enabled,is_auto_expose_frames)


if __name__ == "__main__":
    main()