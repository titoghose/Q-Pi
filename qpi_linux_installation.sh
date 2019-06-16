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
pip3 install --user numpy
pip3 install --user scipy
pip3 install --user matplotlib
pip3 install --user opencv-python>=3.3.0.0
pip3 install --user pims_nd2
pip3 install --user python-bioformats
pip3 install --user vtk envisage
# clear

printf "Installing Mayavi (3D visualization library for python) \n\n"
pip3 install --user mayavi