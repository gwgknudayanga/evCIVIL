import h5py
import numpy as np
import cv2


LABELMAP = ["crack", "spalling"]

def draw_bboxes(img, labels, labelmap=LABELMAP):

    #colors = cv2.applyColorMap(np.arange(0, 255).astype(np.uint8), cv2.COLORMAP_HSV)
    #colors = [tuple(*item) for item in colors.tolist()]

    for i in range(labels.shape[0]):

        pt1 = (int(labels[i,1]), int(labels[i,2]))
        size = (int(labels[i,3]), int(labels[i,4]))
        pt2 = (pt1[0] + size[0], pt1[1] + size[1])
        class_id = int(labels[i,0])
        class_name = labelmap[class_id % len(labelmap)]
        color = (0, 255, 255)
        if class_id == 1:
            color = (0, 255, 0)
        center = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
        cv2.rectangle(img, pt1, pt2, color, 1)
        cv2.putText(img, class_name, (center[0], pt2[1] - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)


def main(h5_file,npy_file):

    # Define video properties
    frame_height = 260
    frame_width = 346
    fps = 20

    label_array = np.load(npy_file)

    #label_array = label_array[label_array[:,0].argsort()]

    frame_db = h5py.File(h5_file,"r")

    frame_tstamps = np.array(frame_db["frame_timestamp"])
    frames = np.array(frame_db["frames"],dtype=np.uint8)

    num_frames = frames.shape[0]


    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, fps, (frame_width, frame_height))

    if not out.isOpened():
        print("Error: Could not open VideoWriter.")

    for i in range(num_frames):

        frame = cv2.cvtColor(frames[i], cv2.COLOR_GRAY2BGR)

        frame_time = frame_tstamps[i]
        candidates = label_array[(label_array[:,0] >= frame_time),0]
        if candidates.size > 0:
            matching_event_tstamp = np.min(candidates)
            matching_boxes = label_array[label_array[:,0] == matching_event_tstamp,:].reshape(-1,6)
            matching_boxes = matching_boxes[:,1:]
            if matching_boxes.size > 0:
                draw_bboxes(frame,matching_boxes)
        out.write(frame)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1000 // fps) & 0xFF == ord('q'):
            break

        """if cv2.waitKey(1000 // fps) & 0xFF == ord('q'):
            break"""

    # Release the VideoWriter and close any OpenCV windows
    out.release()
    #cv2.destroyAllWindows()

if __name__ == "__main__":

    path = "/media/udayanga/data_2/shon_data/DATA_TO_HOST/ev-CIVIL_dataset/Field_dataset/"
    #sub_folder = "Night_outdoor/night_outdoor_31st_dec/31_crack_46/"
    #h5_file = "31_crack_46_frames.h5"
    #npy_file = "31_crack_46_label.npy"

    #sub_folder = "field_data_day/dset_2/spalling_36/"
    #h5_file = "spalling_36_frames.h5"
    #npy_file = "spalling_36_label.npy"

    sub_folder = "day_time_challenge/crack_88/"
    h5_file = "crack_88_frames.h5"
    npy_file = "crack_88_frame_label.npy"

    sub_folder = "Night_outdoor/night_outdoor_18th_Nov/ros_crack_21/"
    h5_file = "ros_crack_21_frames.h5"
    npy_file = "ros_crack_21_frame_label.npy"

    sub_folder = "/tunnel_selected/t_spall_5/"
    h5_file = "t_spall_5_frames.h5"
    npy_file = "t_spall_5_label.npy"


    h5_full_path = path + sub_folder + h5_file
    npy_full_path = path + sub_folder + npy_file
    main(h5_full_path,npy_full_path)