## Q-Pi

Q-Pi is an algorithm designed for enhanced visualisation and high-throughput quantification of percentage invasion.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine. See 'deployment' for notes on how to deploy the project on a live system. Currently, Q-Pi is supported **only on *OSX* and *Linux*** systems. Due to some conflicts with installation of the dependencies, instructions to install dependencies and run the program on *Windows* has not yet been provided, but will be soon.

### Input file
This programme supports various input formats. The input file is expected to be a z-stack containing two channels - membrane and channel respectively, where channel 1 is the membrane, and channel 2 is the cell. Input file format is any microscopy image format supported by python-bioformats and can be checked [here](https://docs.openmicroscopy.org/bio-formats/5.8.1/supported-formats.html).

### Sample Data
A test file generated from NIKON _NIS Elements_ suite can be found [here](https://drive.google.com/open?id=1--SQ_OiZU9fH9Ob6OwfODR9Rdu5TCs_f). 
It can be viewed using the NIS Elements Viewer which can be downloaded from [here](https://www.nikoninstruments.com/Products/Software/NIS-Elements-Advanced-Research/NIS-Elements-Viewer).

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
python --version
```
In case it is not present in the system, it can be found [here](https://www.python.org/downloads/release/python-2714/). Download and install it following the instructions at the given link.

#### 2. Download and install dependency libraries (see '1.' for Mac OSX and skip to '2.' for Linux)

1. Steps to follow for ***Mac OSX*** users
---
Open a terminal window at, or change directory to, the Q-Pi folder. Then, run the below mentioned code. This will install the following dependencies:
+ numpy
+ scipy
+ matplotlib
+ opencv
+ pims_nd2

```
- easy_install pip
- pip install numpy
- pip install scipy
- pip install matplotlib
- pip install opencv-python==3.3.0.10
- pip install pims_nd2
```

Installation of **python-bioformats** requires Java  Development Kit (JDK) to be present on the system. Java should normally be pre-installed on OSX 10.6 and older versions. This is not the case for the versions after 10.6. Hence for this case, JDK can be installed from [here](http://www.oracle.com/technetwork/java/javase/downloads/jdk8-downloads-2133151.html). 

To confirm the presence of JDK on your system the following command should print the version.
```
javac -version
```
Once JDK is installed, run the following command on your terminal to get **python-bioformats**.
```
- pip install python-bioformats
```
To install **mayavi**, execute these commands on a terminal window:
```
- pip install vtk
- pip install envisage
- pip install -U wxPython
- pip install mayavi
```
Mayavi requires wxPython, PyQt4 or PySide but we have opted to go with wxPython. After installation of wxPython, if PyQt is still present on the system, it may cause conflicts while importing **mayavi**. We recommend ensuring that PyQt is not installed on the system to prevent any issues.
 
**Note**: OSX users will be required to additionally download and install XCode command line tools if not already present; in most cases your system should automatically prompt you, if required.

2. Steps to follow for ***Linux*** users
---
Open a terminal window at, or change directory to, the Q-Pi folder. Then, run the below mentioned code. It runs a shell script that installs all the required dependencies. This will install the following dependncies:
+ numpy
+ scipy
+ matplotlib
+ opencv
+ pims_nd2
+ python-bioformats

```
- apt-get update
- apt-get install python-setuptools python-dev build-essential
- easy_install pip
- pip install numpy
- pip install scipy
- pip install matplotlib
- pip install opencv-python==3.3.0.10
- pip install pims_nd2
```

To install **python-bioformats**, requires JDK provided by OpenJDK to be present on the system. Other distributions may cause errors while setting up this library. Execute the following:

```
apt-get install openjdk-8-jdk
pip install python-bioformats
```

To install **mayavi**, execute these commands on a terminal window:

```
- pip install vtk
- apt-get update
- apt-get install python-qt4 python-qt4-gl
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
python qpi.py [Data/file_name.nd2] -lb [LB] -ub [UB] -p
```

These are optional arguments and mean the following:
+ -lb : Lower bound of the z slices where the cell is expected to start from. This can be used when there is fluorescence reflection below the actual bottom of the cell.
+ -ub : Upper bound of the z slices where the cell is expected to end. This can be used when there is fluorescence reflection above the actual top of the cell.
+ -p : Plot the cell. If not mentioned, the cell will not be plotted, only quantification will be done.

Below is an example of how to use these options, in this case, specific to the sample data provided above:

```
python qpi.py [Data/file_name.nd2] -lb 23 -ub 116 -p
```

The following command can be used to get help with these options while running the programme:

```
python qpi.py --help
```

If running with the **python** command gives errors for those with Conda distributions of python, try the same commands with **pythonw**.

**Note**: Due to involvement of 3D plotting and graphics functions, systems without good processing capabilities may take a little time to produce the plots. Systems with good graphics cards and processors will benefit and plotting times will be lower.

### Sequence of Events

1. Wait till the files are extracted and an image is displayed on the screen. The window can be resized to own convenience.

2. Using your mouse, draw the bounding boxes (ROI) around the cells you wish to analyse. Keep a small margin between the cell and the ROI. You can draw upto 15 ROIs. When happy with an ROI, press the key **n** on your keyboard to draw the next one. Once done drawing, press the key **x** to submit the selections.

3. The programme will sequentially analyse each cell. See terminal for intermediate output information.

4. The lateral cross section of the ZX and ZY planes of the Z-Stack will pop up with the auto selected membrane level. If not satisfied with the selection, you can manually use your mouse to select the correct level. The image can be zoomed into for a clearer view. Just select the zoom option and box the region to zoom into with your mouse. We recommend zooming in because some files have very few slices and hence the stack appears very thin. Every click is recorded and the subsequent membrane level is displayed on the terminal. Close the window to finalise selection.

5. If plot option was selected then wait for cell to be plotted without membrane first and then with the membrane at the correct level.

6. Final percentage invsasion will be displayed on terminal after shutting the plots.

## Built With

* [OpenCV](https://opencv.org/) - Image Processing Library
* [Mayavi](http://docs.enthought.com/mayavi/mayavi/) - 3D Plotting Library
* [Python-Bioformats](https://pythonhosted.org/python-bioformats/) - Used to extract raw microscope images to .png format

## Citation

In order to access and cite this code, please use the following DOI: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1239826.svg)](https://doi.org/10.5281/zenodo.1239826)
