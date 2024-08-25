#from __future__ import print_function, absolute_import

import argparse

import numpy as np
import rosbag
import h5py
from cv_bridge import CvBridge
import rosbag
import ctypes
import os
from PIL import Image
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str,default="/home/udayanga/ros_day_indoor_20th_dec_2023/ros_bags/")
parser.add_argument("--output_path", type=str)
parser.add_argument("--color", action="store_true")

args = parser.parse_args()

#bag = rosbag.Bag(args.data_path, "r")
#bridge = CvBridge()

#dataset = h5py.File(args.output_path, "w")
#topics = ["/dvs/image_raw", "/camera/color/image_raw","/dvs/events"]

topics = ["/dvs/image_raw","/dvs/events","/dvs/imu"] #/camera/color/image_raw

src_base_path = "/home/udayanga/ros_day_indoor_20th_dec_2023/ros_bags/"   #"/media/udayanga/OS/Users/gwgkn/Research_work/Ros_recording/Outdoor_Ros_Rec_12_Nov_2023/crack"
ros_bag_file_list = sorted(os.listdir(src_base_path))
output_base_path = "/home/udayanga/ros_day_indoor_20th_dec_2023/h5_files/spalling/" 

defect_type = "spalling"

current_idx = 0

for bag_file in ros_bag_file_list:

    ros_bags_file = os.path.join(src_base_path,bag_file)
    args.data_path = ros_bags_file

    bag = rosbag.Bag(args.data_path, "r")
    bridge = CvBridge()

    sub_dir_name = bag_file.rsplit(".",1)[0] #"ros_" + defect_type + "_" + str(current_idx)
    

    print("sub_dir_name ",sub_dir_name)

    #print("aaaaaaaaaaaaaaaa" , sub_dir_name)
    
    args.output_path = os.path.join(output_base_path,sub_dir_name)

    if not os.path.exists(args.output_path):
        print("creating dir ")
        os.makedirs(args.output_path)

    event_h5_file_path = args.output_path + "/" + sub_dir_name + "_events.h5" # "/ros_" + defect_type + "_" + str(current_idx) + "_events.h5"

    event_dataset = h5py.File(event_h5_file_path,"w")
    event_times = event_dataset.create_dataset("event_timestamp",shape=(0,),maxshape=(None,),dtype="uint64")
    events_x = event_dataset.create_dataset("x",shape=(0,),maxshape=(None,),dtype="uint16")
    events_y = event_dataset.create_dataset("y",shape=(0,),maxshape=(None,),dtype="uint16")
    events_pol = event_dataset.create_dataset("polarity",shape=(0,),maxshape=(None,),dtype="uint8")

    image_h5_file_path = args.output_path + "/" + sub_dir_name + "_frames.h5" # "/ros_" + defect_type + "_" + str(current_idx) + "_frames.h5"

    frame_dataset = h5py.File(image_h5_file_path,"w")
    frame_data = frame_dataset.create_dataset("frames",shape=(0,260,346,3),maxshape=(None, 260,346,3),dtype="uint8")
    frame_time_data = frame_dataset.create_dataset("frame_timestamp",shape=(0,),maxshape=(None,),dtype="uint64")

    imu_h5_file_path = args.output_path + "/" + sub_dir_name +  "_imus.h5" #"/ros_" + defect_type + "_" + str(current_idx) + "_imus.h5"
    imu_dataset = h5py.File(imu_h5_file_path,"w")
    #make this so that first three points are (x,y,z) - linear acceleration and last three points angular velocity
    imu_data = imu_dataset.create_dataset("imu",shape=(0,10),maxshape=(None,10),dtype="float32") #(x,y,z,w) orientation., (x,y,z) angular velocity , linear acceleration (x,y,z)'
    imu_time_data = imu_dataset.create_dataset("imu_timestamp",shape=(0,),maxshape=(None,),dtype="uint64")                                            # uint64 

    first_time_stamp = 0
    first_event_need_to_set = True
     
    number_of_imu_msgs = 0
    for topic, msg, t in bag.read_messages(topics=topics):
        if topic == topics[1]:
            events = msg.events
            num_events = len(events)
            #print("number of events ",num_events)

            event_data = np.array(
				[[x.x, x.y,x.ts.to_nsec()/1e3,
				x.polarity] for x in events],
				dtype="uint64")
            
            if first_event_need_to_set:
                first_time_stamp = event_data[0,2]
                first_event_need_to_set = False

            event_times.resize(event_times.shape[0]+num_events, axis=0)
            event_times[-num_events:] = event_data[:,2] - first_time_stamp
            events_x.resize(events_x.shape[0]+num_events, axis=0)
            events_x[-num_events:] = event_data[:,0]
            events_y.resize(events_y.shape[0]+num_events, axis=0)
            events_y[-num_events:] = event_data[:,1]
            events_pol.resize(events_pol.shape[0]+num_events, axis=0)
            events_pol[-num_events:] = event_data[:,3]
            #print("events_pol shape ",events_pol.shape)

        elif topic in topics[2]:
               
               if first_event_need_to_set:
                   continue
               #print("printing imu message ",msg)

               current_imu_ts = msg.header.stamp.to_nsec()/1e3 #float(msg.header.stamp.to_nsec())/1e9
               current_imu_ts -= first_time_stamp
               imu_time_data.resize(imu_time_data.shape[0] + 1,axis=0)
               imu_time_data[-1] = current_imu_ts

               orient_arr = np.array([msg.orientation.x,msg.orientation.y,msg.orientation.z,msg.orientation.w]).reshape(-1,4)
               angular_vel_arr = np.array([msg.angular_velocity.x,msg.angular_velocity.y,msg.angular_velocity.z]).reshape(-1,3)
               linear_acc_arr = np.array([msg.linear_acceleration.x,msg.linear_acceleration.y,msg.linear_acceleration.z]).reshape(-1,3)
               current_imu_data = np.concatenate((orient_arr,angular_vel_arr,linear_acc_arr),axis=1).reshape(-1,10)
               imu_data.resize(imu_data.shape[0]+1,axis=0)
               imu_data[-1] = current_imu_data
               #print("size of imu database ",len(imu_data))

               number_of_imu_msgs += 1

        elif topic in topics[0]:
            if first_event_need_to_set:
                print("continueeeeeeeeeeeeeeee .....")
                continue

            im_rgb = bridge.imgmsg_to_cv2(msg, "rgb8")
            try:

                current_frame_ts = msg.header.stamp.to_nsec()/1e3 #float(msg.header.stamp.to_nsec())/1e9
                current_frame_ts -= first_time_stamp

                if current_frame_ts < 0:
                    continue

                frame_time_data.resize(frame_time_data.shape[0] + 1,axis=0)

                frame_time_data[-1] = current_frame_ts

                im = Image.fromarray(im_rgb)
                im = im.resize((346,260),Image.Resampling.BILINEAR)
                #print("im ",im)
                frame_data.resize(frame_data.shape[0]+1, axis=0)
                #print("im shape ",np.array(im).shape)
                frame_data[-1] = np.array(im)
                #print("frame data size ",frame_data.reshape[0])
                
                print(frame_time_data)


            except:
                print("Some error")
                continue

    current_idx += 1
    event_dataset.close()
    frame_dataset.close()        



	
    
