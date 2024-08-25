import h5py
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import glob
import os
import shutil


def cv2_illum_correction(src_img_path,isImg = True):

    if isImg:
        img = src_img_path
    
    else:
        img = cv2.imread(src_img_path)

    num_of_dims = len(img.shape)

    if num_of_dims > 2:
        img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        
    # Load the image
 
    # Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    #print("shape of the image is ",img.shape)
    clahe_result = clahe.apply(img)

    #clahe_result = cv2.fastNlMeansDenoising(clahe_result, None, h=10, templateWindowSize=7, searchWindowSize=21)
    # Display the original and processed images side by side
    """plt.subplot(121), plt.imshow(image, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
 
    plt.subplot(122), plt.imshow(clahe_result, cmap='gray')
    plt.title('CLAHE Result'), plt.xticks([]), plt.yticks([])"""
    #cv2.imshow(clahe_result)
    return clahe_result

def save_images_for_matchscore_calculation(modified_images_to_matchtest_path,file_id,img_array,isImgFrame,isvoxelgrid = False):
    
    modifier = "_evframe"
    if isImgFrame:
        modifier = "_frame"

    relative_out_fname = str(file_id) + modifier + ".png"
    output_full_fname = os.path.join(modified_images_to_matchtest_path,relative_out_fname)

    fig,ax = plt.subplots(frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    
    fig.add_axes(ax)
    if isvoxelgrid:
        ax.imshow(img_array[0],cmap="gray",aspect="auto")
    else:
        ax.imshow(img_array,cmap="gray",aspect="auto")

    #plt.close()
    #fig.savefig(output_full_fname)

    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    data = cv2.cvtColor(data, cv2.COLOR_RGB2GRAY)

    plt.close()
    #print("data.shape",data)
    im = Image.fromarray(data)
    im.save(output_full_fname)


def read_h5_extract_frames_and_save(src_img_h5_path,destination_img_files_path):

    data = h5py.File(src_img_h5_path,"r")
    frames = data["frames"]
    #frame_array = np.array(frames)

    for img_idx in range(len(frames)):
        desired_img = frames[img_idx]
        desired_img =  cv2_illum_correction(desired_img) #illum_correction_for_frames(desired_img)
        #desired_img = (desired_img - desired_img.min())/desired_img.max()
        save_images_for_matchscore_calculation(destination_img_files_path,file_id = img_idx ,img_array = desired_img,isImgFrame = True)

def main():

    h5_file_base_path = "/media/udayanga/data_2/Final_Data_set/To_Labelled/at_DTU_4/h5_files/crack/"
    destination_img_base_path = "/media/udayanga/data_2/Final_Data_set/To_Labelled/at_DTU_4/aedat_data/negetive_time_crack_frames/"

    h5_sub_folders_for_samples = glob.glob(h5_file_base_path + "/*")

    for h5_sub_folder in h5_sub_folders_for_samples:
        local_folder_name = h5_sub_folder.rsplit("/",1)[1]
        local_h5_name = local_folder_name+ "_frames.h5"
        h5_full_name = os.path.join(h5_sub_folder,local_h5_name)

        folder_path_for_all_imgs_to_this_h5 = os.path.join(destination_img_base_path,local_folder_name)
        if not os.path.exists(folder_path_for_all_imgs_to_this_h5):
            os.makedirs(folder_path_for_all_imgs_to_this_h5)
            read_h5_extract_frames_and_save(h5_full_name,folder_path_for_all_imgs_to_this_h5)


def select_json_file_names_and_copy_to_relevant_all_imgs_folder(json_files_parents_base_path,all_imgs_parents_base_path):

    json_files_parents = glob.glob(json_files_parents_base_path + "/*")

    for json_file_parent in json_files_parents:

        corresponding_all_imgs_folders_local_name = json_file_parent.rsplit("/",1)[1]

        json_files_for_current_sample = glob.glob(json_file_parent + "/*.json")

        for json_file in json_files_for_current_sample:
            json_local_file_name = json_file.rsplit("/",1)[1]
            json_file_incremented_index = int(json_local_file_name.split("_",1)[0])
            json_file_remainder_part = json_local_file_name.split("_",1)[1]
            json_file_corrected_index = json_file_incremented_index - 2
            modifed_json_file_name = str(json_file_corrected_index) + "_" + json_file_remainder_part
            #if not os.path.exists(os.path.join(all_imgs_parents_base_path,corresponding_all_imgs_folders_local_name)):
            #    continue
            modified_json_file_full_name_for_destination = os.path.join(all_imgs_parents_base_path,corresponding_all_imgs_folders_local_name,modifed_json_file_name)
            shutil.copy(json_file,modified_json_file_full_name_for_destination)

if __name__ == "__main__":
    main()
    #json_files_parents_base_path = "/media/udayanga/data_2/Final_Data_set/To_Labelled/Real_Final/ev_dataset_1/images_evimages_files/crack/"
    #all_imgs_parents_base_path = "/media/udayanga/data_2/Final_Data_set/To_Labelled/Real_Final/ev_dataset_1/all_frame_images_full/crack/"
    #select_json_file_names_and_copy_to_relevant_all_imgs_folder(json_files_parents_base_path,all_imgs_parents_base_path)
    #main()
