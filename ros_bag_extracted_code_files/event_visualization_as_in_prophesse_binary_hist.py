import os
import numpy as np
import cv2
import h5py
import math
import argparse


class H5EventVisualizer(object):

    def __init__(self,h5File_path):

        self.h5File = h5py.File(h5File_path,"r")
        timestamps = self.h5File["event_timestamp"]
        self.current_event_pos = 0
        self.current_timestamp = timestamps[0]
        self._decode_dtype = [('t','u8'),('x', 'u2'), ('y', 'u2'), ('p', 'u1')]
        self.tot_events = len(timestamps)
        print("Total number of events ",self.tot_events)
        self.done = False
    
    def seek_event(self,ev_count):
        if ev_count <= 0:
            self.current_time = self.h5File["event_timestamp"][0]
        elif ev_count >= self.tot_events:
            self.current_event_pos = self.tot_events
            self.current_time = self.h5File["event_timestamp"][self.tot_events - 1]
        else:
            self.current_event_pos = ev_count
            self.current_time = self.h5File["event_timestamp"][ev_count]
        if self.current_event_pos == self.tot_events - 1:
            self.done = True

    def seek_time(self,final_time,term_criterion=100000):

        if final_time > self.h5File["event_timestamp"][self.tot_events - 1]:
            print("skip all the events ")
            self.done = True
            return
        if final_time < 0:
            return
        low = 0
        high = self.tot_events

        while high - low > term_criterion:
            middle = (low + high) // 2
            mid = self.h5File["event_timestamp"][middle]

            if mid > final_time:
                high = middle
            elif mid < final_time:
                low = middle + 1
            else:
                self.current_time = final_time
                return
            
        final_buffer = self.h5File["event_timestamp"][low:high]
        final_index = np.searchsorted(final_buffer, final_time)

        self.current_event_pos = low + final_index
        self.done = self.current_event_pos >= self.tot_events


    def stream_td_data(self,buffer,pos,count):

        buffer['t'][:count] = self.h5File["event_timestamp"][pos:pos + count]
        buffer['x'][:count] = self.h5File["x"][pos:pos + count]
        buffer['y'][:count] = self.h5File["y"][pos:pos + count]
        buffer['p'][:count] = self.h5File["polarity"][pos:pos + count]

    def load_n_events(self,ev_count):
        
        buffer = np.empty((ev_count+1,), dtype=self._decode_dtype)

        if (self.current_event_pos + ev_count) > self.tot_events:
            self.done = True
            ev_count = self.tot_events - self.current_event_pos
            
            self.stream_td_data(buffer,self.current_event_pos,ev_count)
            self.current_event_pos = self.tot_events - 1
            self.current_timestamp = buffer['t'][ev_count]

        else:
            self.stream_td_data(buffer,self.current_event_pos,ev_count)
            self.current_event_pos = self.current_event_pos + ev_count
            self.current_timestamp = buffer['t'][ev_count]
    
        return buffer[:ev_count]

    def load_delta_t(self,delta_t):

        if delta_t < 1:
            raise ValueError("load_delta_t(): delta_t must be at least 1 micro-second: {}".format(delta_t))

        if self.done or (self.current_event_pos >= self.tot_events):
            self.done = True

        final_time = self.current_timestamp + delta_t
        tmp_time = self.current_timestamp
        batch = 100000
        event_buffer = []
        pos = self.current_event_pos

        while tmp_time < final_time and pos < self.tot_events:
            print(self.tot_events)
            print(batch)
            print(pos)
            count = min(self.tot_events,pos + batch) - pos
            buffer = np.empty((count,), dtype=self._decode_dtype)
            self.stream_td_data(buffer,pos,count)
            print("kkkkkkkkkkkkk ",buffer)
            tmp_time = buffer['t'][-1]
            event_buffer.append(buffer)
            pos += count

        if tmp_time >= final_time:
            self.current_timestamp = final_time
        else:
            self.current_timestamp = tmp_time
        
        assert len(event_buffer) > 0
        idx = np.searchsorted(event_buffer[-1]['t'], final_time)
        event_buffer[-1] = event_buffer[-1][:idx]
        event_buffer = np.concatenate(event_buffer)
        idx = len(event_buffer)
        self.current_event_pos += idx
        self.done = self.current_event_pos >= self.tot_events
        
        return event_buffer


    #def __del__(self):
    #    self.h5File.close()

LABELMAP = ["car", "pedestrian"]

def make_binary_histo(events, img=None, width=304, height=240):
    """
    simple display function that shows negative events as blacks dots and positive as white one
    on a gray background
    args :
        - events structured numpy array
        - img (numpy array, height x width x 3) optional array to paint event on.
        - width int
        - height int
    return:
        - img numpy array, height x width x 3)
    """
    if img is None:
        img = 127 * np.ones((height, width, 3), dtype=np.uint8)
    else:
        # if an array was already allocated just paint it grey
        img[...] = 127
    if events.size:
        assert events['x'].max() < width, "out of bound events: x = {}, w = {}".format(events['x'].max(), width)
        assert events['y'].max() < height, "out of bound events: y = {}, h = {}".format(events['y'].max(), height)

        img[events['y'], events['x'], :] = 255 * events['p'][:, None]
        #Keep the binary histogram always up.
        
    return img


def draw_bboxes(img, boxes, labelmap=LABELMAP):
    """
    draw bboxes in the image img
    """
    colors = cv2.applyColorMap(np.arange(0, 255).astype(np.uint8), cv2.COLORMAP_HSV)
    colors = [tuple(*item) for item in colors.tolist()]

    for i in range(boxes.shape[0]):
        pt1 = (int(boxes['x'][i]), int(boxes['y'][i]))
        size = (int(boxes['w'][i]), int(boxes['h'][i]))
        pt2 = (pt1[0] + size[0], pt1[1] + size[1])
        score = boxes['class_confidence'][i]
        class_id = boxes['class_id'][i]
        class_name = labelmap[class_id % len(labelmap)]
        color = colors[class_id * 60 % 255]
        center = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
        cv2.rectangle(img, pt1, pt2, color, 1)
        cv2.putText(img, class_name, (center[0], pt2[1] - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)
        cv2.putText(img, str(score), (center[0], pt1[1] - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)


def play_files_parallel(h5_files, labels=None, delta_t=10000, skip=5000):
    """
    Plays simultaneously files and their boxes in a rectangular format.
    """
    # open the video object for the input files
    videos = [H5EventVisualizer(h5_file) for h5_file in h5_files]
    # use the naming pattern to find the corresponding box file
    #box_videos = [PSEELoader(glob(td_file.split('_td.dat')[0] +  '*.npy')[0]) for td_file in td_files]
    #print(box_videos)

    height, width = 260,346 #videos[0].get_size()
    #labelmap = vis.LABELMAP if height == 240 else vis.LABELMAP_LARGE

    # optionally skip n minutes in all videos
    #print(videos)
    #print(box_videos)
    for v in videos:
        v.seek_time(skip)

    # preallocate a grid to display the images
    size_x = int(math.ceil(math.sqrt(len(videos))))
    size_y = int(math.ceil(len(videos) / size_x))
    frame = np.zeros((size_y * height, width * size_x, 3), dtype=np.uint8)

    cv2.namedWindow('out', cv2.WINDOW_NORMAL)

    # while all videos have something to read
    num_ev_bundles = 0
    while not sum([video.done for video in videos]):

        # load events and boxes from all files
        #
        events = [video.load_delta_t(delta_t) for video in videos]
        #print("events ",events[0])
        #box_events = [box_video.load_delta_t(delta_t) for box_video in box_videos]

        print("Number of events ",len(events))
        #print("box_events ",len(box_events))
        
        for index,evs in enumerate(events):
            y, x = divmod(index, size_x)
            # put the visualization at the right spot in the grid
            im = frame[y * height:(y + 1) * height, x * width: (x + 1) * width]
            # call the visualization functions
            im = make_binary_histo(evs, img=im, width=width, height=height)
            #print("boxes ",boxes)
            #vis.draw_bboxes(im, boxes, labelmap=labelmap)
        
        num_ev_bundles += 1

        # display the result
        cv2.imshow('out', frame)
        cv2.waitKey(1)
    print("num_ev_bundles ",num_ev_bundles)

#history of the chaos.

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='visualize one or several event files along with their boxes')
    parser.add_argument('records', nargs="+",
                        help='input event files, annotation files are expected to be in the same folder')
    parser.add_argument('-s', '--skip', default=1000, type=int, help="skip the first n microseconds")
    parser.add_argument('-d', '--delta_t', default=30000, type=int, help="load files by delta_t in microseconds")

    return parser.parse_args()

if __name__ == "__main__":
    
    #/media/udayanga/OS/Users/gwgkn/Research_work/Synthetic_Event_Data/crack_aedata_synthesized_h5_files/
    # h5_files/crack/synth_crack_1/synth_crack_1_events.h5

    ARGS = parse_args()
    play_files_parallel(ARGS.records,labels=None,delta_t = ARGS.delta_t,skip=ARGS.skip)
