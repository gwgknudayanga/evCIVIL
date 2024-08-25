import os
import glob

def main():
    base_path = "/media/udayanga/data_2/Final_Data_set/To_Labelled/9th_indoor_december_best/images_evimages_files"

    sub_folders = os.listdir(base_path)
    for sub_folder in sub_folders:
        file_full_names = glob.glob(base_path+"/" + sub_folder + "/*.png")

        for file_name in file_full_names:
            #print(file_name)
            local_name = file_name.rsplit("/",1)[1]
            modified_name = local_name.rsplit("_",1)[0] + ".png"
            modified_full_name = file_name.rsplit("/",1)[0] + "/" + modified_name
            print(modified_full_name)
            os.rename(file_name,modified_full_name)

if __name__ == "__main__":
    main()

