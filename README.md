
## Q-Pi

Q-Pi is an algorithm designed for enhanced visualisation and high-throughput quantification of percentage invasion.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine. See 'deployment' for notes on how to deploy the project on a live system. Currently, Q-Pi is supported **only on *OSX* and *Linux*** systems. Due to some conflicts with installation of the dependencies, *Windows* support is currently not available but will be provided soon.

### Dependencies

These are the dependencies needed for running this program and installation instructions given below.

```
Python 2.7.*
numpy
scipy
matplotlib
opencv
pims_nd2
python-bioformats
vtk
mayavi
```

### Installing

Follow these steps to get your system ready to use the program.

#### 1. Download and install Python 2.7.*

By default, OSX and Linux come installed with Python. In case it is not present in the system, it can be found [here](https://www.python.org/downloads/release/python-2714/).

Download and install it following the instructions at the given link.

#### 2. Download and install dependency libraries

1. Steps to follow for ***Mac OSX*** users

Open a terminal window at, or change directory to, the Q-Pi folder. Then, run the below mentioned code. It runs a shell script that installs all the required dependencies. This will install the following dependncies:
+ numpy
+ scipy
+ matplotlib
+ opencv
+ pims_nd2
+ python-bioformats

```
- sudo easy_install pip
- sudo pip install numpy
- sudo pip install scipy
- sudo pip install matplotlib
- sudo pip install opencv-python==3.3.0.10
- sudo pip install pims_nd2
```

To install **python-bioformats**, requires Java to be present on the system. Java should be pre-installed on OSX. So just the following command should install the library:
```
- sudo pip install python-bioformats
```

To install **mayavi**, execute these commands on a terminal window:
```
- pip install vtk
- pip install envisage
- pip install -U wxPython
- pip install mayavi
```

**Note**: OSX users will be required to additionally download and install XCode if not already present; in most cases your system should automatically prompt you, if required.

2. Steps to follow for ***Linux*** users
Open a terminal window at, or change directory to, the Q-Pi folder. Then, run the below mentioned code. It runs a shell script that installs all the required dependencies. This will install the following dependncies:
+ numpy
+ scipy
+ matplotlib
+ opencv
+ pims_nd2
+ python-bioformats

```
- sudo apt-get update
- sudo apt-get install python-setuptools python-dev build-essential
- sudo easy_install pip
- sudo pip install numpy
- sudo pip install scipy
- sudo pip install matplotlib
- sudo pip install opencv-python==3.3.0.10
- sudo pip install pims_nd2
```

To install **python-bioformats**, requires OpenJDK Java distribution to be present on the system. Other distributions will cause linking errors with this library. Execute the following:

```
apt-get install openjdk-8-jdk
pip install python-bioformats
```

To install **mayavi**, execute these commands on a terminal window:

```
- pip install vtk
- apt-get update
- apt-get install python-qt4 python-qt4-gl python-setuptools python-c
- easy-install EnvisageCore EnvisagePlugins
- pip install envisage
- pip install mayavi
```

## Using Q-Pi

### Running the programme

In order to test the programmeme, clone or download this repository to your system. Place your .nd2, .lif or .czi data file in a folder called *Data* inside the main folder.

Now to run the code, open a terminal at the main folder and execute the following.

```
python qpi.py [file_name]
```

*file_name* must be the full path to the location of the data file. All intermediate outputs will be saved in a folder with the name *Data_file_name*.

A few advanced options are provided:

```
python qpi.py [Data/file_name.nd2] -lb [LB] -ub [UB] -p -w [WIN]
```

These are optional arguments and mean the following:
+ -lb : Lower bound of the z slices where the cell is expected to start from. This can be used when there is fluorescence reflection below the actual bottom of the cell.
+ -ub : Upper bound of the z slices where the cell is expected to end. This can be used when there is fluorescence reflection above the actual top of the cell.
+ -p : Plot the cell. If not mentioned, the cell will not be plotted, only quantification will be done.
+ -w : Window size for membrane level selection. In case of	bleed-through in membrane channel, we recommend this option. Recommended value is 0.25. If no bleed-through exists, do not mention this option and the programme should automatically select the membrane level.

This is an example of how to use these options:

```
python qpi.py [Data/file_name.nd2] -lb 10 -ub 80 -p -w 0.25
```

The following command can be used to get help with these options while running the programme:

```
python qpi.py --help
```


### Sequence of Events

1. Wait till the files are extracted and an image is displayed on the screen. The window can be resized to own convenience.

2. Using your mouse, draw the bounding boxes around the cells you wish to analyse. Keep a small margin between the cell and the bounding box. You can draw upto 15 boxes. When happy with a box, press the key **n** on your keyboard to draw the next box. Once done drawing, press the key **x** to submit the selections.

3. The programme will sequentially analyse each cell. Look at the terminal for intermediate output information.

4. The lateral cross section of the ZX and ZY planes of the Z-Stack will pop up with the auto selected membrane level. If not satisfied with the selection, you can manually use your mouse to select the correct level. The image can be zoomed into for a clearer view. Every click is recorded and the subsequent membrane level is displayed on the terminal. Close the window to finalise selection.

5. If plot option was selected then wait for cell to be plot without membrane first and then with the membrane at the correct level.

6. Final percentage invsasion will be displayed on terminal after shutting the plots.

## Sample Data

A sample .nd2 file can be found [here](link) and used to test the program.

## Built With

* [OpenCV](https://opencv.org/) - Image Processing Library
* [Mayavi](http://docs.enthought.com/mayavi/mayavi/) - 3D Plotting Library
* [Python-Bioformats](https://pythonhosted.org/python-bioformats/) - Used to extract raw microscope images to .png format

## License

This project is licensed under the ??? License - see the [LICENSE.md](LICENSE.md) file for details

## Citation

In order to cite this code please use the following DOI :
