from setuptools import setup

setup(
   name='Q-Pi',
   version='1.0',
   description='A useful module',
   author='Upamanyu Ghose',
   author_email='titoghose@gmail.com',
   packages=['Q-Pi'],  #same as name
   install_requires=['numpy', 'scipy', 'matplotlib', 'pims', 'mayavi', 'opencv-python'], #external packages as dependencies
)