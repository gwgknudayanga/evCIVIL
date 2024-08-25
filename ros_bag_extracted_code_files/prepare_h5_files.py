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
import argparse


# Set these values to limit the number of frames that should be included inside the h5 file
min_frame_id_thres = 5
max_frame_id_thres = 205

def video_to_frames(video_path,output_frame_path):
    
    if not os.path.exists(video_path):
        print("Error : video_path {} not found".format(video_path))
        
    if  os.path.exists(output_frame_path):
        shutil.rmtree(output_frame_path)
    
    os.makedirs(output_frame_path)

    cap = cv2.VideoCapture(video_path)
    if cap.isOpened():
        current_frame = 0
        ret = 1
        while ret: 
            ret, frame = cap.read()
            if ret:
                name = f'{output_frame_path}/frame{current_frame}.jpg'
                #print(f"Creating file... {name}")
                cv2.imwrite(name, frame)
            current_frame += 1
    cap.release()
    return True


def comma_to_dot(value):
    return float(value.decode('utf-8').replace(',', '.'))


def create_database_for_imu_data(imu_file_path,h5_file_path,isFromJaer = True):

    if os.path.exists(h5_file_path):
        os.remove(h5_file_path)
    
    imu_dataset = h5py.File(h5_file_path,"w")

    print("imu_file_path ",imu_file_path)
    
    #make this so that first three points are (x,y,z) - linear acceleration and last three points angular velocity

    imu_data = imu_dataset.create_dataset("imu",shape=(0,6),maxshape=(None,6),dtype="float32") #(x,y,z,w) orientation., (x,y,z) angular velocity , linear acceleration (x,y,z)'
    imu_time_data = imu_dataset.create_dataset("imu_timestamp",shape=(0,),maxshape=(None,),dtype="uint64")

    try:
        imu_data_array = np.loadtxt(imu_file_path)
    except:
        print("imu file with comma delimiter ")
        try:
            retval = os.system(f"sed -i s/,/./g {imu_file_path}")
            if not retval:
                imu_data_array = np.loadtxt(imu_file_path)
        except:
            print(f"file {imu_file_path} cannot be loaded .")

    num_of_imu_samples = imu_data_array.shape[0]

    imu_time_data.resize(num_of_imu_samples,axis=0)
    imu_data.resize(num_of_imu_samples,axis=0)
    
    imu_time_data[-num_of_imu_samples:] = imu_data_array[:,0]
    
    imu_data[-num_of_imu_samples:] = imu_data_array[:,1:]

    imu_dataset.close()


def create_dataset_for_events(event_file_path,h5_file_path,isFromJaer = True,v_flip_required = True):
    
    if os.path.exists(h5_file_path):
        os.remove(h5_file_path)
    event_dataset = h5py.File(h5_file_path,"w")
    
    event_times = event_dataset.create_dataset("event_timestamp",shape=(0,),maxshape=(None,),dtype="uint32")
    events_x = event_dataset.create_dataset("x",shape=(0,),maxshape=(None,),dtype="uint16")
    events_y = event_dataset.create_dataset("y",shape=(0,),maxshape=(None,),dtype="uint16")
    events_pol = event_dataset.create_dataset("polarity",shape=(0,),maxshape=(None,),dtype="uint8")

    event_stat_array = np.loadtxt(event_file_path)
    #print("Maximum of event stat array y x ",event_stat_array[:,2].max(),event_stat_array[:,1].max())

    if v_flip_required:
        event_stat_array[:,2] = 259 - event_stat_array[:,2]
    
    if (event_stat_array[:,0].max()*1e6) > 4294967295:
        print("Error : maximum time stamp {} doesn't fit into uint32 max value".format(event_stat_array[:,0].max()*1e6))
        print("Event file path ",event_file_path)
        sys.exit(0)

    number_of_events = len(event_stat_array[:,1])

    if isFromJaer:
        event_stat_array[:,0] = event_stat_array[:,0]*1e6   #bringing seconds to micro seconds
        event_times.resize(event_times.shape[0]+number_of_events, axis=0)
        event_times[-number_of_events:] = event_stat_array[:,0]
        events_x.resize(events_x.shape[0]+number_of_events,axis=0)
        events_x[-number_of_events:] = event_stat_array[:,1]
        events_y.resize(events_y.shape[0]+number_of_events,axis=0)
        events_y[-number_of_events:] = event_stat_array[:,2]
        events_pol.resize(events_pol.shape[0]+number_of_events,axis=0)
        events_pol[-number_of_events:] = event_stat_array[:,3]
        #pp = np.array(event_dataset["x"])
        #print("ppppppppppppppp ",event_stat_array[:,1])
    event_dataset.close()

def create_dataset_for_frames(h5_file_name,video_path,frame_time_file_path,isFromJaer = True):
    frame_dataset = h5py.File(h5_file_name,"w")
    frame_data = frame_dataset.create_dataset("frames",shape=(0, 260, 346),maxshape=(None, 260, 346),dtype="uint8")
    frame_time_data = frame_dataset.create_dataset("frame_timestamp",shape=(0,),maxshape=(None,),dtype="uint32")
    if isFromJaer:
        
        idx = video_path.find("video.avi")
        frames_path = video_path[:idx] + "/frames"
        retVal = False
        if not os.path.exists(frames_path):
            retVal = video_to_frames(video_path,frames_path)
        else:
            retVal = True
        if retVal:
            frame_time_stamp_array = np.loadtxt(frame_time_file_path)
            number_of_frames = len(frame_time_stamp_array[:,1])

            modified_id = 0
            for frame_idx,frame_time in enumerate(frame_time_stamp_array[:,1]):
                if frame_idx < min_frame_id_thres or frame_idx > max_frame_id_thres: #(number_of_frames - 3):
                    continue
                
                img_frame_name_suffix = int(frame_time_stamp_array[frame_idx,0])
                img_local_name = str(img_frame_name_suffix) + ".jpg"
                idx = video_path.find("video.avi")
                img_frame_full_name = frames_path + "/frame" + img_local_name

                frame_data.resize(frame_data.shape[0]+1, axis=0)
                img = cv2.imread(img_frame_full_name, cv2.IMREAD_GRAYSCALE)
                frame_data[-1] = img
                frame_time_data.resize(frame_time_data.shape[0] + 1,axis=0)
                frame_time_data[-1] = frame_time
                modified_id += 1
    frame_dataset.close()

def create_3_databases_from_aedat(base_path,defect_type,desired_folder_id = -1,is_both_events_and_imgs = True):
    #one database for events
    #another for frames
    #another for labels (tstamp,bbox,class) : this is in .npz format

    desired_aedat_path = os.path.join(base_path,"aedat_data",defect_type)

    temp_count = 0
    for candidate_folder in sorted(os.listdir(desired_aedat_path)):
        print("candidate folder ",candidate_folder)

        if desired_folder_id != -1:
            current_folder_idx = candidate_folder.rsplit('_',1)[1]
            print(current_folder_idx)
            if current_folder_idx != str(desired_folder_id):
                continue                

        #aedat_file_folder = base_path + "/aedat_data/" + defect_type + "/" + candidate_folder
        
        aedat_file_folder = desired_aedat_path + "/" + candidate_folder
        
        if is_both_events_and_imgs:
            video_path = aedat_file_folder + "/video.avi" 
            frame_time_file_path = aedat_file_folder + "/video-timecode.txt"
        
        event_file_path = os.path.join(aedat_file_folder,"events-events.txt")
        
        if not os.path.exists(event_file_path):
            event_file_path = os.path.join(aedat_file_folder,"events-events.txt")
            
       
        sub_folder_full_path = base_path + "/h5_files/" + defect_type + "/" + candidate_folder

        if os.path.exists(sub_folder_full_path):
            continue

        if not os.path.exists(sub_folder_full_path):
            os.makedirs(sub_folder_full_path)
        else:
            print("exist the path ",sub_folder_full_path)
            continue
            
        h5_file_name_events = sub_folder_full_path + "/" + candidate_folder + "_events" + ".h5"
        
        if is_both_events_and_imgs:
            h5_file_name_frames = sub_folder_full_path + "/" + candidate_folder + "_frames" + ".h5"
        #npy_file_name_labels = base_path + "/h5_files/" + defect_type + "/" + candidate_folder + "_labels" + ".npy"s
        
        """is_event_file_exist = False
        is_image_file_exist = False
        
        if os.path.exists(h5_file_name_events):
            is_event_file_exist = True
        
        if os.path.exists(h5_file_name_frames):
            is_image_file_exist = True
        
        if is_image_file_exist and is_event_file_exist:
            print("h5 files already exist.. hence continue..")
            continue"""
            
        create_dataset_for_events(event_file_path,h5_file_name_events)

        process_imu = True
        
        if process_imu:
            imu_file_path = os.path.join(aedat_file_folder,"events-imu.txt")
            h5_file_name_imu = sub_folder_full_path + "/" + candidate_folder + "_imu" + ".h5"
            create_database_for_imu_data(imu_file_path,h5_file_name_imu,isFromJaer=True)

        if is_both_events_and_imgs:          
            create_dataset_for_frames(h5_file_name_frames,video_path,frame_time_file_path)
        #create_labels_for_dataset()
        #break

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str,default="")
    parser.add_argument("--defect_type",type=str,default="")
    args = parser.parse_args()
    
    base_path = "/media/udayanga/data_2/Final_Data_set/To_Labelled/New_night_indoor_jaer/8th_march_laser/"
    defect_type = "crack" #args.defect_type #"texture"
    
    create_3_databases_from_aedat(base_path,defect_type,desired_folder_id = -1,is_both_events_and_imgs = True)

if __name__ == "__main__":
    main()
