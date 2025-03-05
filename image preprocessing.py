import cv2, os
from glob import glob


input_folder = "Training Images"
output_folder = "Training Images 2"
os.makedirs(output_folder, exist_ok=True)

def process_image(image_path, output_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    
    crop_x_start = (image.shape[1]) // 2
    crop_y_start = (image.shape[0]) // 2
    crop_x_start = crop_x_start - crop_y_start
    
    cropped_image = image[0:0+image.shape[0], crop_x_start:crop_x_start+image.shape[0]]
    
    lab_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_image)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)
    
    processed_lab = cv2.merge((cl, a_channel, b_channel))
    
    processed_image = cv2.cvtColor(processed_lab, cv2.COLOR_LAB2BGR)

    resized_image = cv2.resize(processed_image, (512, 512))
    
    cv2.imwrite(output_path, resized_image)
    

image_paths = glob(os.path.join(input_folder, "*.jpg"))
for image_path in image_paths:
    filename = os.path.basename(image_path)
    output_path = os.path.join(output_folder, filename)
    process_image(image_path, output_path)
    