## Q-Pi

Q-Pi is an algorithm designed for enhanced visualisation and high-throughput quantification of percentage invasion.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system. Currently, Q-Pi is supported **only on *OSX* and *Linux*** systems. We plan to add *Windows* support soon.

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

A step by step series of examples that tell you have to get a development env running

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

## Running the tests

In order to test the program, place your .ND2 data file in a folder called *Data* inside the main folder.

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone who's code was used
* Inspiration
* etc
