# Q-Pi

Q-Pi is an algorithm designed for enhanced visualisation and high-throughput quantification of percentage invasion.

This program supports various input formats. The input file is expected to be a z-stack containing two channels - membrane and channel respectively, where channel 1 is the membrane, and channel 2 is the cell. Input file format is any microscopy image format supported by **python-bioformats** and can be checked [here](https://docs.openmicroscopy.org/bio-formats/5.8.1/supported-formats.html).

# Sample Data
A **sample data file** generated from NIKON _NIS Elements_ suite can be found [here](https://drive.google.com/open?id=1--SQ_OiZU9fH9Ob6OwfODR9Rdu5TCs_f).

It can be viewed using the **NIS Elements Viewer** (only on Mac OS and Windows) which can be downloaded from [here](https://www.nikoninstruments.com/Products/Software/NIS-Elements-Advanced-Research/NIS-Elements-Viewer).


# Installing

Follow the relevant section (Windows or Linux) to get your system ready to use the program.


## Windows Installation

Follow these steps sequentially to setup the system to run Q-Pi:

1. Q-Pi uses libraries that need the Microsoft C++ Build Tools to be installed on the system. The 2008, 2010 and 2015 versions need to be installed. The links to the same are given below. Once downloaded, run the executable files to install the tools on the system.
     * [C++ Build Tools 2008](https://www.microsoft.com/en-in/download/details.aspx?id=15336)
     * [C++ Build Tools 2010](https://www.microsoft.com/en-in/download/details.aspx?id=14632)
     * [C++ Build Tools 2015](https://www.microsoft.com/en-in/download/details.aspx?id=48145)

2. Oracle Java Development Kit 8 needs to be downloaded and installed. The link to download it can be found here:
     * [Oracle JDK 8](https://www.oracle.com/technetwork/java/javase/downloads/jdk8-downloads-2133151.html)

3. Install Python 3.x on the system. The Anaconda 3 distribution is recommended because some 3D visualization libraries may create issues with installation on a normal Python distribution. The link to download it is given below.
     * [Windows Anaconda 3](https://www.anaconda.com/distribution/)

4. Download this repository (link on top of page) and extract the files at a desired location on your system. Now follow these steps:
   * From the start menu, open an **Anaconda Prompt**
   * Navigate to the Q-Pi folder on your system and find the **qpi_windows_installation.bat** file.
   * Hold down the *Shift* key and *right-click* the file. Select the *Copy as path* option and paste it on the **Anaconda Prompt** opened earlier.
   * Press *Enter* to execute the command.
   * Wait for all the python libraries to get installed on your system. This may take some time.

After executing these steps, Q-Pi should be ready to run on the Windows system. Make sure the steps were executed sequentially.


## Linux installation

Download or Clone this repository. Open a terminal, navigate inside the Q-Pi folder and execute the following steps to install all the dependencies for python:

```shell
chmod +x qpi_linux_installation.sh
./qpi_linux_installation.sh
```

After executing these steps, Q-Pi should be ready to run on the Linux system. Make sure the steps were executed sequentially.

# Using Q-Pi

In order to test the program, clone or download this repository to your system. Place your data file in a folder called *Data* inside the main folder.

Now to run the code, open a terminal (Linux) or anaconda prompt (Windows) inside the the main folder and execute the following.

**NOTE:** All the *python* commands refer to Python 3 and not Python 2.

```shell
python qpi.py [file_name]
```

*[file_name]* must be the full path to the location of the data file. All intermediate outputs will be saved in a folder with the name *Data_[file_name]*.

A few advanced optional user-based modifications are available:

```shell
python qpi.py [Data/file_name.nd2] -lb [LB] -ub [UB] -p
```

These are optional arguments and mean the following:
* **-lb:** Lower bound of the z slices where the cell is expected to start from. This can be used when there is fluorescence reflection below the actual bottom of the cell.
* **-ub:** Upper bound of the z slices where the cell is expected to end. This can be used when there is fluorescence reflection above the actual top of the cell.
* **-p:** Plot the 3D reconstruction of the cell and the membrane. If not mentioned, the cell will not be plotted, only quantification will be done.
* **-i:** Plot the intermediate image processing applied on the different z slices. If not mentioned, these plots will not be generated.

The following command can be used to get help with these options while running the program:

```shell
python qpi.py --help
```

## Running Q-Pi on the sample data file

```shell
python qpi.py Data/2036_quantify.nd2 -lb 23 -ub 116 -p
```

**NOTE**: Due to involvement of 3D plotting and graphics functions, systems without good processing capabilities may take a little time to produce the plots. Systems with good graphics cards and processors will benefit and plotting times will be lower.

## Sequence of events after running

1. Wait till the files are extracted and a z-slice is displayed on the screen. The window can be resized to own convenience.

2. Using your mouse, draw the bounding boxes (ROI) around the cells you wish to analyse. Keep a small margin between the cell and the ROI.  When happy with an ROI, press the key **N** on your keyboard to draw the next one. Once done drawing, press the key **X** to submit the selections.

   **NOTE:** Scrolling up and down will change the z-slice displayed.

3. The program will sequentially analyse each cell. See terminal for intermediate output information.

4. The lateral cross section of the ZX and ZY planes of the Z-Stack will pop up with the auto selected membrane level. If not satisfied with the selection, you can manually use your mouse to select the correct level. Close the window to finalise selection.

   **NOTE:** The check buttons can be used to hide or show a particular channel.

5. If **-p** option was selected then wait for cell to be plotted without membrane first and then with the membrane at the correct level.

6. Final percentage invasion will be displayed on the terminal and saved in a text file after shutting the plots.

## Built With

* [OpenCV](https://opencv.org/) - Image Processing Library
* [Mayavi](http://docs.enthought.com/mayavi/mayavi/) - 3D Plotting Library
* [Python-Bioformats](https://pythonhosted.org/python-bioformats/) - Used to extract raw microscope images to .png format

## Citation

In order to access and cite this code, please use the following DOI: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1239826.svg)](https://doi.org/10.5281/zenodo.1239826)
