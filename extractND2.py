from pims import ND2_Reader
import matplotlib.pyplot as plt
import os
import shutil

mDir = 'Data/no_cells_lam_perl_Data'

try:
    os.mkdir(mDir)
except OSError:
    None


def extract_from_ND2(file_name, c):
    global calib
    print file_name
    frames = ND2_Reader(file_name)
    calib = float(frames.metadata['calibration_um'])

    f = ''
    if c == 1:
        f = mDir + '/c1/'
    else:
        f = mDir + '/c2/'

    frames.default_coords['c'] = c - 1

    num_slices = frames[0].shape[0]

    try:
        os.mkdir(f)
        for j, fr in enumerate(frames[0]):
            fn = file_name.split('.')
            print f + str(j) + '_' + fn[0].split('/')[1] + '.png'
            plt.imsave(f + str(j) + '_' + fn[0].split('/')[1] + '.png', fr, cmap='gray')
            print("Progress: [%f]" % ((100.0 * j) / num_slices))

    except OSError:
        None


extract_from_ND2('/home/upamanyu/Documents/Rito/Q-Pi/Data/no_cells_lam_perl.nd2', 1)
extract_from_ND2('/home/upamanyu/Documents/Rito/Q-Pi/Data/no_cells_lam_perl.nd2', 2)
