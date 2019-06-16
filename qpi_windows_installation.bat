:: install JDK and set path first
:: install Visual Studio C++ Build Tools

:: installing python dependencies
pip install --user numpy 
pip install --user scipy 
pip install --user matplotlib 
pip install --user "opencv-python>=3.3.0.0"
pip install --user pims_nd2
pip install --user python-bioformats   
pip install --user vtk
pip install --user envisage

:: installing Mayavi (3D visualization library for python)
pip install --user mayavi