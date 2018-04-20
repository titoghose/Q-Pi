from setuptools import setup

setup(
   name='Q-Pi',
   version='1.0',
   description='A useful module',
   author='Man Foo',
   author_email='foomail@foo.com',
   packages=['Q-Pi'],  #same as name
   install_requires=['mayavi', 'librosa'], #external packages as dependencies
)