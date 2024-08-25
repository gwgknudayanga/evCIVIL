import h5py
import os
import shutil
import numpy as np



def select_subset_from_h5_file(original_h5_frame_path,original_h5_event_path,original_h5_imu_path,destination_h5_file_folder_path,desired_start_frame_number,number_of_desired_frames = -1,check_non_monotomic_only = False):
    
    #desired frame details selection

    desired_end_frame_number = -1

    original_frame_database = h5py.File(original_h5_frame_path,"r")

    if check_non_monotomic_only:
        if len(original_frame_database) >= 200:
            check_non_monotomic_only = False


    if number_of_desired_frames > 0:
        if len(original_frame_database["frames"]) < number_of_desired_frames:
            number_of_desired_frames = len(original_frame_database["frames"]) - desired_start_frame_number - 1
        desired_end_frame_number = desired_start_frame_number + number_of_desired_frames

    desired_frames_h5_subset = original_frame_database["frames"][desired_start_frame_number + 1 :desired_end_frame_number]
    #print(desired_frames_h5_subset.shape)
    original_frame_time_stamp_dataset = original_frame_database["frame_timestamp"]
    desired_frame_timestamp_h5_subset = original_frame_time_stamp_dataset[desired_start_frame_number + 1: desired_end_frame_number]



    #writing desired frames to new h5 file
    target_h5_frame_file_name = os.path.join(destination_h5_file_folder_path,original_h5_frame_path.rsplit("/",1)[1])
    target_frame_h5_file = h5py.File(target_h5_frame_file_name,"w")
    target_frame_tstamp_dataset = target_frame_h5_file.create_dataset("frame_timestamp",shape=(len(desired_frame_timestamp_h5_subset),),maxshape=(None,),dtype="uint64")
    target_frame_tstamp_dataset.write_direct(desired_frame_timestamp_h5_subset)
    target_frame_dataset = target_frame_h5_file.create_dataset("frames",shape=(len(desired_frames_h5_subset),260,346),maxshape=(None, 260,346),dtype="uint8")
    target_frame_dataset.write_direct(desired_frames_h5_subset)
    target_frame_h5_file.close()

    #desired event details selection
    event_collection_start_time = original_frame_database["frame_timestamp"][desired_start_frame_number]
    event_collection_end_time = original_frame_database["frame_timestamp"][desired_end_frame_number] 
    original_frame_database.close()

    original_h5_event_file = h5py.File(original_h5_event_path,"r")

    if check_non_monotomic_only:
        event_collection_start_time = original_h5_event_file["event_timestamp"][0]
        event_collection_end_time = original_h5_event_file["event_timestamp"][-1]

    non_monotomic_start_arr = np.argwhere(original_h5_event_file["event_timestamp"] < original_h5_event_file["event_timestamp"][0])

    start_from_here_idx = 0
    if np.any(non_monotomic_start_arr):
        start_from_here_idx = np.min(non_monotomic_start_arr)

    candidate_start_idx_arr = np.argwhere(original_h5_event_file["event_timestamp"] >= event_collection_start_time)
    mask_1 = candidate_start_idx_arr > start_from_here_idx
    desired_event_start_index = np.min(candidate_start_idx_arr[mask_1])

    candidate_end_idx_arr = np.argwhere(original_h5_event_file["event_timestamp"] <= event_collection_end_time)
    mask_2 = candidate_end_idx_arr > start_from_here_idx
    desired_event_end_index = np.max(candidate_end_idx_arr[mask_2])


    desired_event_time_stamps = original_h5_event_file["event_timestamp"][desired_event_start_index:desired_event_end_index]
    desired_event_x_coords = original_h5_event_file["x"][desired_event_start_index:desired_event_end_index]
    desired_event_y_coords = original_h5_event_file["y"][desired_event_start_index:desired_event_end_index]
    desired_event_pol_coords = original_h5_event_file["polarity"][desired_event_start_index:desired_event_end_index]

    #writing desired events to new h5 file
    target_h5_event_file_name = os.path.join(destination_h5_file_folder_path,original_h5_event_path.rsplit("/",1)[1])
    target_event_h5_file = h5py.File(target_h5_event_file_name,"w")


    target_event_times_dataset = target_event_h5_file.create_dataset("event_timestamp",shape=(len(desired_event_time_stamps),),maxshape=(None,),dtype="uint64")
    target_event_times_dataset.write_direct(desired_event_time_stamps)
    target_events_x_dataset = target_event_h5_file.create_dataset("x",shape=(len(desired_event_x_coords),),maxshape=(None,),dtype="uint16")
    target_events_x_dataset.write_direct(desired_event_x_coords)
    target_events_y_dataset = target_event_h5_file.create_dataset("y",shape=(len(desired_event_y_coords),),maxshape=(None,),dtype="uint16")
    target_events_y_dataset.write_direct(desired_event_y_coords)
    target_events_pol_dataset = target_event_h5_file.create_dataset("polarity",shape=(len(desired_event_pol_coords),),maxshape=(None,),dtype="uint8")
    target_events_pol_dataset.write_direct(desired_event_pol_coords)

    target_event_h5_file.close()

    if not original_h5_imu_path:
        return

    original_imu_database = h5py.File(original_h5_imu_path,"r")
    imu_collection_start_time = event_collection_start_time # original_frame_database["frame_timestamp"][desired_start_frame_number]
    imu_collection_end_time = event_collection_end_time # original_frame_database["frame_timestamp"][desired_end_frame_number] 

    desired_imu_start_index = np.min(np.argwhere(original_imu_database["imu_timestamp"] >= imu_collection_start_time))
    desired_imu_end_index = np.max(np.argwhere(original_imu_database["imu_timestamp"] <= imu_collection_end_time))

    desired_imu_h5_subset = original_imu_database["imu"][desired_imu_start_index :desired_imu_end_index]
    desired_imu_timestamp_subset = original_imu_database["imu_timestamp"][desired_imu_start_index :desired_imu_end_index]


    target_h5_imu_file_name = os.path.join(destination_h5_file_folder_path,original_h5_imu_path.rsplit("/",1)[1])
    target_imu_h5_file = h5py.File(target_h5_imu_file_name,"w")
    imu_tstamp_database = target_imu_h5_file.create_dataset("imu_timestamp",shape=(len(desired_imu_timestamp_subset),),maxshape=(None,),dtype="uint64")
    imu_tstamp_database.write_direct(desired_imu_timestamp_subset)

    imu_database = target_imu_h5_file.create_dataset("imu",shape=(len(desired_imu_h5_subset),10),maxshape=(None,10),dtype="float32")
    imu_database.write_direct(desired_imu_h5_subset)

    target_imu_h5_file.close()


if __name__ == "__main__":
    
    """base_path = "/media/udayanga/OS/Users/gwgkn/Research_work/object_tracking/At_ETH_crack_frames/h5_files_subset"
    original_h5_frame_path = base_path + "/selected_cracks" + "/ros_crack_10" + "/ros_crack_10_frames.h5"
    original_h5_event_path = base_path + "/selected_cracks" + "/ros_crack_10" + "/ros_crack_10_events.h5"
    destination_h5_file_folder_path = base_path + "/selected_subset_cracks/ros_crack_10"""
    
    subfiles = os.listdir("/media/udayanga/data_2/Final_Data_set/To_Labelled/New_night_outdoor_jaer/Dec_25_Aedat/aedat_data/h5_files/crack/")
    
    for sub_dir in subfiles:
        #sub_dir = "spalling_1"

        base_path = "/media/udayanga/data_2/Final_Data_set/To_Labelled/New_night_outdoor_jaer/Dec_25_Aedat/aedat_data/h5_files/crack/" + sub_dir
        original_h5_frame_path =  base_path + "/"  + sub_dir + "_frames.h5"
        original_h5_event_path = base_path + "/"  + sub_dir + "_events.h5"
        original_h5_imu_path = None #base_path + "/"  + sub_dir + "_imus.h5"

        destination_h5_file_folder_path = "/media/udayanga/data_2/Final_Data_set/To_Labelled/New_night_outdoor_jaer/Dec_25_Aedat/aedat_data/h5_files_subset/crack/" + sub_dir
        
        desired_start_frame_number = 0
        desired_number_of_frames = 250

        if not os.path.exists(destination_h5_file_folder_path):
            os.makedirs(destination_h5_file_folder_path)

        select_subset_from_h5_file(original_h5_frame_path,original_h5_event_path,original_h5_imu_path,destination_h5_file_folder_path,desired_start_frame_number,desired_number_of_frames,check_non_monotomic_only=True)