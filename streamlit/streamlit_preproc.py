import cv2

def streamlit_processing(image, size):
    h, w = image.shape[:2]
    min_dim = min(h, w)
    crop_x = (w - min_dim) // 2
    crop_y = (h - min_dim) // 2

    cropped_image = image[crop_y:crop_y + min_dim, crop_x:crop_x + min_dim]

    lab = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    processed_lab = cv2.merge((cl, a, b))
    processed_image = cv2.cvtColor(processed_lab, cv2.COLOR_LAB2BGR)

    resized_image = cv2.resize(processed_image, (size, size))
    return resized_image
