apt-get update
apt update
# clear

printf "Installing pip python package manager \n\n"
apt install python3-pip
# clear

printf "Installing Java Development Kit (JDK) \n\n"
apt-get install openjdk-8-jdk
# clear

printf "Install PyQt4 (Visualization Backend) \n\n"
apt install python3-pyqt4
# clear

printf "Installing python dependencies \n\n"
pip3 install numpy scipy matplotlib opencv-python>=3.3.0.0 pims_nd2 python-bioformats vtk envisage
# clear

printf "Installing Mayavi (3D visualization library for python) \n\n"
pip3 install mayavi