import cv2
import numpy as np
from PIL import Image
import io

def load_image_from_bytes(image_bytes):
    """从字节流加载图像"""
    image = Image.open(io.BytesIO(image_bytes))
    return np.array(image)

def resize_image_for_display(image, max_size=400):
    """调整图像大小以便于显示"""
    h, w = image.shape[:2]
    if h > w:
        new_h, new_w = max_size, int(w * max_size / h)
    else:
        new_h, new_w = int(h * max_size / w), max_size
    
    return cv2.resize(image, (new_w, new_h))
