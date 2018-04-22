

## Q-Pi

Q-Pi is an algorithm designed for enhanced visualisation and high-throughput quantification of percentage invasion.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine. See 'deployment' for notes on how to deploy the project on a live system. Currently, Q-Pi is supported **only on *OSX* and *Linux*** systems. Due to some conflicts with installation of the dependencies, *Windows* support is currently not available but will be provided soon.

### Input file
This programme supports various input formats. The input file is expected to be a z-stack containing two channels - membrane and channel respectively, where channel 1 is the membrane, and channel 2 is the cell. Input file format is any microscopy image format supported by python-bioformats and can be checked [here](https://docs.openmicroscopy.org/bio-formats/5.8.1/supported-formats.html).

### Sample Data
A test file generated from NIKON _NIS Elements_ suite can be found [here](https://drive.google.com/open?id=1--SQ_OiZU9fH9Ob6OwfODR9Rdu5TCs_f). 

### Dependencies

These are the dependencies needed for running this programme and installation instructions given below.

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

Follow these steps to get your system ready to use the programme.

#### 1. Download and install Python 2.7.*

By default, OSX and Linux come installed with Python. Running the following code in a terminal window will reveal if installed, and which version.
```
python
```
In case it is not present in the system, it can be found [here](https://www.python.org/downloads/release/python-2714/). Download and install it following the instructions at the given link.

#### 2. Download and install dependency libraries (see '1.' for Mac OSX and skip to '2.' for Linux)

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

Installation of **python-bioformats** requires Java to be present on the system. Java should normally be pre-installed on OSX. So just the following command should install the library:

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

In order to test the programme, clone or download this repository to your system. Place your data file in a folder called *Data* inside the main folder.

Now to run the code, open a terminal at the main folder and execute the following.

```
python qpi.py [file_name]
```

*file_name* must be the full path to the location of the data file. All intermediate outputs will be saved in a folder with the name *Data_file_name*.

A few advanced optional user-based modifications are available:

```
python qpi.py [Data/file_name.nd2] -lb [LB] -ub [UB] -p -w [WIN]
```

These are optional arguments and mean the following:
+ -lb : Lower bound of the z slices where the cell is expected to start from. This can be used when there is fluorescence reflection below the actual bottom of the cell.
+ -ub : Upper bound of the z slices where the cell is expected to end. This can be used when there is fluorescence reflection above the actual top of the cell.
+ -p : Plot the cell. If not mentioned, the cell will not be plotted, only quantification will be done.
+ -w : Window size for membrane level selection. In case of	bleed-through in membrane channel, we recommend this option. Recommended value is 0.25. If no bleed-through exists, do not mention this option and the programme should automatically select the membrane level.

Below is an example of how to use these options, in this case, specific to the sample data provided above:

```
python qpi.py [Data/file_name.nd2] -lb 23 -ub 116 -p -w 0.25
```

The following command can be used to get help with these options while running the programme:

```
python qpi.py --help
```


### Sequence of Events

1. Wait till the files are extracted and an image is displayed on the screen. The window can be resized to own convenience.

2. Using your mouse, draw the bounding boxes (ROI) around the cells you wish to analyse. Keep a small margin between the cell and the ROI. You can draw upto 15 ROIs. When happy with an ROI, press the key **n** on your keyboard to draw the next box. Once done drawing, press the key **x** to submit the selections.

3. The programme will sequentially analyse each cell. See terminal for intermediate output information.

4. The lateral cross section of the ZX and ZY planes of the Z-Stack will pop up with the auto selected membrane level. If not satisfied with the selection, you can manually use your mouse to select the correct level. The image can be zoomed into for a clearer view. Every click is recorded and the subsequent membrane level is displayed on the terminal. Close the window to finalise selection.

5. If plot option was selected then wait for cell to be plotted without membrane first and then with the membrane at the correct level.

6. Final percentage invsasion will be displayed on terminal after shutting the plots.

## Built With

* [OpenCV](https://opencv.org/) - Image Processing Library
* [Mayavi](http://docs.enthought.com/mayavi/mayavi/) - 3D Plotting Library
* [Python-Bioformats](https://pythonhosted.org/python-bioformats/) - Used to extract raw microscope images to .png format

## Citing the programme

Please use the following DOI: _'pending'_, and cite the following publication: _'pending'_

