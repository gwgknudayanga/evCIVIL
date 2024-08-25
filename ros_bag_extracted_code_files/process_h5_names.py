import h5py
import numpy as np

import os
import glob

def main():
    base_path = "/media/udayanga/OS/Users/gwgkn/Research_work/Ros_recording/outdoor_16th_Nov_2023/h5_files/"
    image_db = glob.glob(base_path + "/*/*_images*.h5")
    for file in image_db:
        local_name = file.rsplit("/",1)[1]
        idx = local_name.rsplit("_",1)[1]
        new_name = "ros_crack_" + idx + "_frames.h5"
        new_full_name = file.rsplit("/",1)[0] + new_name
        print(new_full_name)
        #os.rename(file,new_full_name)


if __name__ == "__main__":
    main()
