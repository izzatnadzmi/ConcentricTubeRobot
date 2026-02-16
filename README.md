# ConcentricTubeRobot

Concentric Tube Continuum Robot distal-end tip position controller cascaded with BVP controller.

Final MSc Dissertation: [Closed-loop System for Concentric-Tube Continuum Robots](CTR__Final.pdf)  
Model based on [A geometrically exact model for externally loaded concentric-tube continuum robots](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3091283/)

## Documentation

1. controller.py

    Main code for CTR system. Calls CTR model and trajectory generator.
    
    Also contains classes for Jacobian Linearisation and Controller.

2. CTR_model.py

    Model for a three-tubed concentric tube continuum robot class.
    
3. CurvatureController.py

    End curvature BVP controller.

4. TrajectoryGenerator.py

    Generates a quintic\quadratic\cubic polynomial, or linear trajectory.


## Requirements

- Python 3.7.x (2.7 is not supported)

- pathos

    For multiprocessing
    
- numpy

- matplotlib

- scipy

## How to use

1. Clone the repo.

> git clone https://github.com/izzatnadzmi/ConcentricTubeRobot.git

> cd ConcentricTubeRobot/

2. Install required, and missing libraries.

3. Execute python script.


## License

MIT

