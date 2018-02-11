import os
import shutil
from PIL import Image
import numpy as np

new_dir_path = os.getcwd() + "/jpeg_images/"

try:
    os.mkdir(new_dir_path)
except OSError:
    print "Directory exists."

for i in os.listdir("/Users/upamanyughose/Documents/Rito/cell_Volume/images"):
    if not i.endswith("C2.tif"):
        continue
    img_path = os.getcwd() + "/images/" + i
    img = Image.open(img_path)
    img = img.convert("RGB")
    new_img_path = new_dir_path + i.split(".")[0] + ".jpg"
    img.save(new_img_path, "JPEG", quality=80)
