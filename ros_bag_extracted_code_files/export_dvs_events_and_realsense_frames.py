from __future__ import print_function, absolute_import

import argparse

import numpy as np
import rosbag
import h5py
from cv_bridge import CvBridge
import rosbag
import ctypes
import os

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str,default="/home/udayanga/davis_rosbags_2023-10-28-19-29-52.bag")
parser.add_argument("--output_path", type=str)
parser.add_argument("--color", action="store_true")

args = parser.parse_args()

bag = rosbag.Bag(args.data_path, "r")
bridge = CvBridge()

#dataset = h5py.File(args.output_path, "w")
#topics = ["/dvs/image_raw", "/camera/color/image_raw","/dvs/events"]

topics = ["/camera/color/image_raw","/dvs/events","/dvs/imu"]

"""img_data = dataset.create_dataset(
    "davis/left/image_raw",
    shape=(0, 260, 346, 3),
    maxshape=(None, 260, 346, 3),
    dtype="uint8")

img_tstamp = dataset.create_dataset("davis/left/image_tstamp",shape=(0,1),maxshape=(None,1),dtype="float64")

dvs_data = dataset.create_dataset(
    "davis/left/events",
    shape=(0, 4),
    maxshape=(None, 4),
    dtype="uint32")"""



src_base_path = "/media/udayanga/OS/Users/gwgkn/Research_work/Ros_recording/outdoor_16th_Nov_2023/to_tobi/crack"
ros_bag_file_list = os.listdir(src_base_path)
output_base_path = "/media/udayanga/OS/Users/gwgkn/Research_work/Ros_recording/outdoor_16th_Nov_2023/to_tobi/h5_files/"

defect_type = "crack"

current_idx = 26

for bag_file in ros_bag_file_list:
	
	ros_bags_file = os.path.join(src_base_path,bag_file)
	
	args.data_path = ros_bags_file
	
	sub_dir_name = "ROS_" + defect_type + "_" + current_idx
	
	args.output_path = os.path.join(output_base_path,sub_dir_name)

	event_h5_file_path = args.output_path + "/ros_images_" + str(current_idx) + ".h5"

	event_dataset = h5py.File(event_h5_file_path,"w")
	event_times = event_dataset.create_dataset("event_timestamp",shape=(0,),maxshape=(None,),dtype="uint64")
	events_x = event_dataset.create_dataset("x",shape=(0,),maxshape=(None,),dtype="uint16")
	events_y = event_dataset.create_dataset("y",shape=(0,),maxshape=(None,),dtype="uint16")
	events_pol = event_dataset.create_dataset("polarity",shape=(0,),maxshape=(None,),dtype="uint8")
	

	image_h5_file_path = args.output_path + "/ros_images_" + str(current_idx) + ".h5"
	
	frame_dataset = h5py.File(image_h5_file_path,"w")
	frame_data = frame_dataset.create_dataset("frames",shape=(0,480,640,3),maxshape=(None, 480,640,3),dtype="uint8")
	frame_time_data = frame_dataset.create_dataset("frame_timestamp",shape=(0,),maxshape=(None,),dtype="uint64")

	"""imu_h5_file_path = args.output_path +"/ros_imus_" + str(idx) + ".h5"
	imu_dataset = h5py.File(imu_h5_file_path,"w")
	imu_data = imu_data.create_dataset("",shape=())
	imu_time_data = imu_dataset. """
	
	
	fist_time_stamp = 0
	first_event_need_to_set = True

	for topic, msg, t in bag.read_messages(topics=topics):


		#print("topic ",topic)
	    if topic == topics[1]:
			events = msg.events
			num_events = len(events)
			print("number of events ",num_events)

		# save events
	     
			event_data = np.array(
				[[x.x, x.y,x.ts.to_nsec()/1e3,
				x.polarity] for x in events],
				dtype="uint64")

			if first_event_need_to_set:
				fist_time_stamp = event_data[0,2]
				first_event_need_to_set = False

			event_times.resize(event_times.shape[0]+num_events, axis=0)
			event_times[-num_events:] = event_data[:,2] - fist_time_stamp

			events_x.resize(events_x.shape[0]+num_events, axis=0)
			events_x[-num_events:] = event_data[:,0]
			events_y.resize(events_y.shape[0]+num_events, axis=0)
			events_y[-num_events:] = event_data[:,1]
			events_pol.resize(events_pol.shape[0]+num_events, axis=0)
			events_pol[-num_events:] = event_data[:,3]

			print("events_pol shape ",events_pol.shape)
			
			#dvs_data[-num_events:] = event_data

			"""event_ts_collector = np.append(
				event_ts_collector, [float(x.ts.to_nsec())/1e9 for x in events])
			print("Processed {} events".format(num_events))"""

	    elif topic in topics[2]:
	    
			if first_event_need_to_set:
				continue
			
	    elif topic in topics[0]:
	    
			if first_event_need_to_set:
				print("continueeeeeeeeeeeeeeee .....")
				continue
			im_rgb = bridge.imgmsg_to_cv2(msg, "rgb8")
			try:
				# save image
				frame_data.resize(frame_data.shape[0]+1, axis=0)
				frame_data[-1] = im_rgb
				
				current_frame_ts = msg.header.stamp.to_nsec()/1e3 #float(msg.header.stamp.to_nsec())/1e9
				current_frame_ts -= fist_time_stamp
				print("current img tstamp is ",np.uint32(current_frame_ts))
				#print("image data ",img_data)
				frame_data.resize(frame_data.shape[0]+1, axis=0)
				frame_data[-1] = current_frame_ts 
				"""frame_ts_collector = np.append(
					frame_ts_collector, [current_frame_ts], axis=0)
				print("Processed frame.")"""
			except TypeError:
				print("Some error")
				continue

	current_idx += 1
	event_dataset.close()
	frame_dataset.close()
	break
	   
	
		
	

#Relationship of imu tstamp and event tstamps
#In event tstamps
#timeeeeeeeeeeeeeeeeeeeeeeeeeeeee stamp  1699475777893303043
#in imu tstamps
#    secs: 1699475784
#    nsecs: 373900043
#in image tstamps


