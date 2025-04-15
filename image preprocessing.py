import cv2, os
from glob import glob

img_size = 128
input_folder = "Testing Images orig"
output_folder = "Testing Images"
os.makedirs(output_folder, exist_ok=True)

def processing(image, output, size):
    image = cv2.imread(image, cv2.IMREAD_COLOR)
    
    crop_x = (image.shape[1]) // 2
    crop_y = (image.shape[0]) // 2
    crop_x = crop_y - crop_x
    
    cropped_image = image[0:0+image.shape[0], crop_x:crop_x+image.shape[0]]
    
    lab = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    processed_lab = cv2.merge((cl, a, b))
    
    processed_image = cv2.cvtColor(processed_lab, cv2.COLOR_LAB2BGR)

    resized_image = cv2.resize(processed_image, (size, size))
    
    cv2.imwrite(output, resized_image)
    

image_paths = glob(os.path.join(input_folder, "*.jpg"))
for i in image_paths:
    file = os.path.basename(i)
    output_path = os.path.join(output_folder, file)
    processing(i, output_path, img_size)
    