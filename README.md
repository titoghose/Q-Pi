## Q-Pi

Q-Pi is an algorithm designed for enhanced visualisation and high-throughput quantification of percentage invasion.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system. Currently, Q-Pi is supported **only on *OSX* and *Linux*** systems. Due to some conflicts with installation of the dependencies, *Windows* support is currently not available but will be provided soon.

### Prerequisites

These are the prerequisites for running this program.

```
Python 2.7.*
numpy
scipy
matplotlib
opencv
pims_nd2
vtk
mayavi
```

### Installing

Follow these steps to get your system ready to use the program.

#### 1. Download and install Python 2.7.* 
	
By default, OSX and Linux come installed with Python. In case it is not present in the system, it can be found [here](https://www.python.org/downloads/release/python-2714/).

Download and install it following the instructions.

#### 2. Download and install dependency libraries
Next, execute the following commands on a terminal at the main Q-Pi folder. It runs a shell script that installs all the required dependencies.

**Note**: sudo rights needed

```
- chmod +x setup.sh
- ./setup.sh
```

## Using Q-Pi

### Running the program

In order to test the program, clone or download this repository to your system. Place your .nd2 data file in a folder called *Data* inside the main folder. 

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
+ -lb : Lower bound of the z slices where the cell is expected to start from.
+ -ub : Upper bound of the z slices where the cell is expected to end.
+ -p : Plot the cell. If not mentioned, the cell will not be plot, only quantification will be done.
+ -w : Window size to analyse and auto predict membrane level. If no bleed through exists, do not mention this option. In case of 	bleed through in experiment, recommended value is 0.25 .

This is an example of how to use these options:

```
python qpi.py [Data/file_name.nd2] -lb 10 -ub 80 -p -w 0.25
```

The following command can be used to get help with these options while running the program:

```
python qpi.py --help
```


### Sequence of Events

1. Wait till the files are extracted and an image is displayed on the screen. The window can be resized to own convenience.

2. Using your mouse, draw the bounding boxes around the cells you wish to analyse. Keep a small margin between the cell and the bounding box. You can draw upto 15 boxes. When happy with a box, press the key **n** on your keyboard to draw the next box. Once done drawing, press the key **x** to submit the selections.

3. The program will sequentially analyse each cell. Look at the terminal for intermediate output information. 

4. The lateral cross section of the ZX and ZY planes of the Z-Stack will pop up with the auto selected membrane level. If not satisfied with the selection, you can manually use your mouse to select the correct level. The image can be zoomed into for a clearer view. Every click is recorded and the subsequent membrane level is displayed on the terminal. Close the window to finalise selection.

5. If plot option was selected then wait for cell to be plot without membrane first and then with the membrane at the correct level.

6. Final percentage invsasion will be displayed on terminal after shutting the plots.

## Built With

* [OpenCV](https://opencv.org/) - Image Processing Library
* [Mayavi](http://docs.enthought.com/mayavi/mayavi/) - 3D Plotting Library
* [Pims_ND2](https://github.com/soft-matter/pims_nd2/) - Used to extract .nd2 files to .png

## License

This project is licensed under the ??? License - see the [LICENSE.md](LICENSE.md) file for details

## Citation

In order to cite this code please use the following DOI : 