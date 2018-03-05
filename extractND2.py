from pims import ND2_Reader
import matplotlib.pyplot as plt
import os
import shutil

home_folder = 'ND2/'
file_name = 'day_10_ctrl002.nd2'
folder_name = 'Day10_c1/'
frames = ND2_Reader(home_folder + file_name)
frames.default_coords['c'] = 0


def extract_from_ND2(folder_name, file_name):
    try:
        os.mkdir(folder_name)
    except OSError:
        shutil.rmtree(folder_name)
        os.mkdir(folder_name)

    print frames[0].shape

    for i, f in enumerate(frames[0]):
        plt.imsave(folder_name + str(i) + '_' + file_name + '.png', f, cmap='gray')
        print i
