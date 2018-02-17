import os
import sys
import matplotlib.pyplot as plt
import numpy as np

membrane_z = 0


def handle_matplotlib_mouse(event):
    global membrane_z
    membrane_z = event.y
    print event.ydata
    print membrane_z


# function to create a z stack from a set of slices
def create_z_stack(path):
    z_stack = np.array([])

    # search for existing stack, else create
    try:
        print("Trying to load existing Z Stack.")
        z_stack = np.load(path + '/Z_STACK.npy')
    except IOError:
        print("Z Stack doesn't exist. Creating now.")

        # loops through slices to create stack using np.vstack
        for ind, i in enumerate(os.listdir(path)):
            if i.endswith('.jpeg'):
                img_name = path + '/' + i
                img = plt.imread(img_name)
                if z_stack.shape[0] == 0:
                    z_stack = np.expand_dims(img, axis=0)
                else:
                    z_stack = np.vstack((z_stack, np.expand_dims(img, axis=0)))
            print "Progress: [%d%%]\r" % (((ind + 1) / 60.) * 100)
            sys.stdout.flush()

        np.save(path + '/Z_STACK.npy', z_stack, allow_pickle=True)

    return z_stack


z_stack = create_z_stack('/Users/upamanyughose/Documents/Rito/cellVolume/membrane_cell/c2')
lateral_cs = z_stack[:, 100, :, :]

fig = plt.figure()
fig.canvas.mpl_connect('button_press_event', handle_matplotlib_mouse)
plt.imshow(lateral_cs)
plt.show()

print membrane_z
