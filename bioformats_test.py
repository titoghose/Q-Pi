import javabridge
import bioformats
import matplotlib.pyplot as plt

javabridge.start_vm(class_path=bioformats.JARS)

with bioformats.ImageReader(path='Data/ctrl001.nd2') as reader:
    for i in range(40, 60):
        plt.imsave('img'+str(i)+'.png', reader.read(c=1, z=i, rescale=False), cmap='gray')
        # plt.show()

javabridge.kill_vm()
