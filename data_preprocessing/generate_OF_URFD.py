import os 
import cv2
import numpy as np

# Opfical_flow_folder = "/Users/hungnguyen/UIT/XuLiAnh/Fall_Detection/Optical_Flow_Images/OF"
# preprocessing_data = "/Users/hungnguyen/UIT/XuLiAnh/Fall_Detection/dataset_preprocessing/URFD"

# image_preprocessing = []
# # get name of all images in preprocessing data
# for folder in os.listdir(preprocessing_data):
#     folder_path = os.path.join(preprocessing_data, folder)

#     for event in os.listdir(folder_path):
#         event_path = os.path.join(folder_path, event)

#         for image in os.listdir(event_path):
            

def optical_flow_tvl1_generator(data_folder, output_folder, frame_size=(224, 224), L=10):
    """
    Generator function to compute TV-L1 optical flow for L consecutive frames and 
    generate motion patterns as stacks of optical flow images.
    
    Args:
    - data_folder: path to the folder containing 'Falls' and 'NotFalls' folders.
    - output_folder: path where the stacked optical flow images will be saved.
    - frame_size: size of the output optical flow image (default is 224x224).
    - L: number of consecutive frame pairs to stack optical flows from.
    
    Yields:
    - A stacked optical flow tensor O of shape (224, 224, 2*L).
    """
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # TV-L1 Optical Flow algorithm from OpenCV
    tvl1 = cv2.optflow.DualTVL1OpticalFlow_create()
    
    # Iterate over "Falls" and "NotFalls" folders
    categories = ['Falls', 'NotFalls']
    
    for category in categories:
        category_path = os.path.join(data_folder, category)
        event_folders = [f for f in os.listdir(category_path) if os.path.isdir(os.path.join(category_path, f))]
        event_folders.sort()
        
        for event_folder in event_folders:
            input_path = os.path.join(category_path, event_folder)
            output_path = os.path.join(output_folder, category, event_folder)
            
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            
            images = sorted([img for img in os.listdir(input_path)])

            # Ensure we have at least L+1 images to process
            if len(images) < L + 1:
                continue
            
            # Process L consecutive image pairs
            for i in range(len(images) - L):
                stacked_flow = []

                for j in range(L):
                    img1_path = os.path.join(input_path, images[i + j])
                    img2_path = os.path.join(input_path, images[i + j + 1])

                    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
                    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
                    
                    img1 = cv2.resize(img1, frame_size, interpolation=cv2.INTER_AREA)
                    img2 = cv2.resize(img2, frame_size, interpolation=cv2.INTER_AREA)

                    if img1 is None or img2 is None:
                      print(f"Skipping frame pair {images[i+j]} and {images[i+j+1]} due to read error.")
                      continue
                    
                    # Compute TV-L1 optical flow
                    flow = tvl1.calc(img1, img2, None)

                    # Separate the horizontal (dx) and vertical (dy) components
                    dx, dy = flow[..., 0], flow[..., 1]

                    # Stack dx and dy into a 2-channel image
                    stacked_flow.append(dx)
                    stacked_flow.append(dy)

                # Stack all L pairs of (dx, dy) to create a tensor O with shape (224, 224, 2*L)
                flow_stack = np.stack(stacked_flow, axis=-1)
                
                # Save or yield the stacked tensor for further processing
                output_flow_path = os.path.join(output_path, f"flow_stack_{i:05d}.npy")
                np.save(output_flow_path, flow_stack)

                yield flow_stack

data_folder = '/Users/hungnguyen/UIT/XuLiAnh/Fall_Detection/dataset_preprocessing/URFD'  # Assuming "Falls" and "NotFalls" folders are inside
output_folder = '/Users/hungnguyen/UIT/XuLiAnh/Fall_Detection/Optical_FLow_Images'

# Ensure the output folder exists
if not os.path.exists(output_folder):
  os.makedirs(output_folder)

# Initialize the generator
optical_flow_gen = optical_flow_tvl1_generator(data_folder, output_folder, L=10)

# Consume the generator
for flow_stack in optical_flow_gen:
    print(f"Generated stacked flow of shape: {flow_stack.shape}")