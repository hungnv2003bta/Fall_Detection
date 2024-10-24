"""
Script used to prepare the URFD dataset: 

1- It takes the downloaded RGB images of the camera 0, divided into two
    folders: 'Falls' and 'ADLs', where all the images of a video (comprised in a
    folder) are inside one of those folders.

2- It creates another folder for the new dataset with the 'Falls' and
    'NotFalls' folders. All the ADL videos are moved to the 'NotFalls' folder.
    The images within the original 'Falls' folder are divided in three stages:
    (i) the pre-fall ADL images (they go to the new 'NotFalls' folder),
    (ii) the fall itself (goes to the new 'Falls' folder) and
    (iii) the post-fall ADL images (to 'NotFalls').
    
All the images are resized to size (W,H) - both W and H are variables of the
script.

The script should generate all the necessary folders.
"""

import cv2
import os
import csv

# Path where the image are stored 
fall_detection_folder = '/Users/hungnguyen/UIT/XuLiAnh/Fall_Detection'
data_folder = fall_detection_folder + '/dataset/URFD/'
adl_folder = 'NotFalls/'
fall_folder = 'Falls/'
# Path to save output images
output_path = fall_detection_folder + '/dataset_preprocessing/URFD/'
# Label files, download them from the dataset's site 
falls_labels =  fall_detection_folder + '/dataset/URFD/urfall-cam0-falls.csv'
notfalls_labels = fall_detection_folder + '/dataset/URFD/urfall-cam0-adls.csv'

W, H = 224, 224 

#---------------------
# READ LABELS AND STORE THEM
#---------------------

labels = {'falls': dict(), 'notfalls': dict()}

# For falls videos: read the CSV where frame-level labels are given
with open(falls_labels, 'r') as csvfile: 
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    event_type = 'falls'
    for row in spamreader:
        elems = row[0].split(',')  # read a line in the csv 
        if not elems[0] in labels[event_type]:
            labels[event_type][elems[0]] = []
        if int(elems[2]) == 1 or int(elems[2]) == -1:
            labels[event_type][elems[0]].append(0)
        elif int(elems[2]) == 0:
            labels[event_type][elems[0]].append(1)

# For ADL videos: read the CSV where frame-level labels are given
with open(notfalls_labels, 'r') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    event_type ='notfalls'
    for row in spamreader:
        elems = row[0].split(',')  # read a line in the csv 
        if not elems[0] in labels[event_type]:
            labels[event_type][elems[0]] = []
        if int(elems[2]) == 1 or int(elems[2]) == -1:
            labels[event_type][elems[0]].append(0)
        elif int(elems[2]) == 0:
            labels[event_type][elems[0]].append(1)
            
print('Label files processed')

#----------------------------------
# PROCESS THE DATASET
#----------------------------------

if not os.path.exists(output_path):
  os.makedirs(output_path)
if not os.path.exists(output_path + 'Falls'):
  os.makedirs(output_path + 'Falls')
if not os.path.exists(output_path + 'NotFalls'):
  os.makedirs(output_path + 'NotFalls')

# Get all the folders: each one contains the set of images of the video
folders = [f for f in os.listdir(data_folder)
            if os.path.isdir(os.path.join(data_folder, f))]

for folder in folders:
    print('{} videos =============='.format(folder))
    events = [f for f in os.listdir(data_folder + folder) 
                if os.path.isdir(os.path.join(data_folder + folder, f))]
    events.sort() 
    for nb_event, event, in enumerate(events):
        # Create the appropriate folder
        if folder == 'NotFalls':
            event_id = event[:6]
            #new_folder = output_path + 'NotFalls/notfall_{}'.format(event_id)
            new_folder = output_path + 'NotFalls/{}'.format(event)
            if not os.path.exists(new_folder):
                os.makedirs(new_folder)       
        elif folder == 'Falls':
            event_id = event[:7]
            # "No falls" come before and after the fall, so the respective
            # folders must be created
            #new_folder = output_path + 'Falls/fall_{}'.format(event_id)
            new_folder = output_path + 'Falls/{}'.format(event)
            if not os.path.exists(new_folder):
                os.makedirs(new_folder)    
        #folder_created = False
        path_to_images = data_folder + folder + '/' + event + '/'
        
        # Load all the images of the video
        images = [f for f in os.listdir(path_to_images) 
                    if os.path.isfile(os.path.join(path_to_images, f))]
        images.sort()
        fall_detected = False # whether a fall has been detected in the video
        for nb_image, image in enumerate(images):
            x = cv2.imread(path_to_images + image)
            
            # If the image is part of an ADL video no fall need to be
            # considered
            if folder == 'NotFalls':
                # Save the image
                save_path = (output_path +
                    #'NotFalls/notfall_{}'.format(event_id) + 
                    'NotFalls/{}'.format(event) + 
                    '/frame{:04}.jpg'.format(nb_image))
                cv2.imwrite(save_path,
                            cv2.resize(x, (W,H)),
                            [int(cv2.IMWRITE_JPEG_QUALITY), 95]) 
            elif folder == 'Falls':
                event_type = 'falls'
                if nb_image < len(labels[event_type][event_id]) and labels[event_type][event_id][nb_image] == 0:
                    if fall_detected:
                        # Create another folder for an ADL event,
                        # i.e. the post-fall ADL event
                        new_folder = (output_path +
                                    #'NotFalls/notfall_{}_post'.format(
                                    #event_id))
                                    'NotFalls/{}_post'.format(event))
                        if not os.path.exists(new_folder):
                            os.makedirs(new_folder) 
                        
                        save_path = (output_path +
                                    #'NotFalls/notfall_{}_post'.format(
                                    #event_id) +
                                    'NotFalls/{}_post'.format(event) +
                                    '/frame{:04}.jpg'.format(nb_image))
                    else:
                        new_folder = (output_path +
                                    #'NotFalls/notfall_{}_pre'.format(event_id))
                                    'NotFalls/{}_pre'.format(event))
                        if not os.path.exists(new_folder):
                            os.makedirs(new_folder) 
                        save_path = (output_path +
                                    #'NotFalls/notfall_{}_pre'.format(
                                    #event_id) +
                                    'NotFalls/{}_pre'.format(event) +
                                    '/frame{:04}.jpg'.format(nb_image))
                    cv2.imwrite(save_path,
                                cv2.resize(x, (W,H)),
                                [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                    
                elif nb_image < len(labels[event_type][event_id]) and labels[event_type][event_id][nb_image] == 1: # actual fall
                    save_path = (output_path + 
                                #'Falls/fall_{}'.format(event_id) +
                                'Falls/{}'.format(event) +
                                '/frame{:04}.jpg'.format(nb_image))
                    cv2.imwrite(save_path,
                                cv2.resize(x, (W,H)),
                                [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                    # If fall is detected in a video set the variable to True
                    # used to discern between pre- and post-fall ADL events
                    fall_detected = True
 
print('End of the process, all the images stored within the {} folder'.format(
      output_path))