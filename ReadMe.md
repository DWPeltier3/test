# Swarm Characteristics Classification Using Neural Networks

## Purpose
This repository hosts the source code used in the research paper titled "Swarm Characteristics Classification Using Neural Networks." The project aims to share the methodologies and code that were developed to classify swarm characteristics using advanced neural network techniques applied to time series data.

## Features

### Data Generation
- The `Matlab` folder contains code that generates swarm on swarm engagement information, including position and velocity data. This dataset serves as the foundation for training the neural network model.

### Neural Network Training and Evaluation
- The `class.py` script is responsible for training the neural network and evaluating its performance in Time Series Classification (TSC) of swarm characteristics. It outlines the steps taken to prepare the data, construct the model, and assess its accuracy and efficiency in classification tasks.

## How to Install and Use

### Installation
1. The project relies on several dependencies, which are listed in the `swarm.yml` file. This file can be used to build a conda environment specifically for running the code in this repository.
   
To create the environment, run:

    conda env create -f swarm.yml

Activate the newly created environment:

    conda activate swarm


### Running the Code
The code was designed to run on a High Power Computing (HPC) SLURM resource-managed system. The repository includes shell `.sh` files which are scripts intended to submit jobs to the SLURM scheduler. These scripts detail how the code should be executed in such an environment.

## Disclaimer
All code in this repository is provided "as is", with no warranty (even implied) that it will work as advertised. There is no support available, and the authors or contributors to this code cannot be held responsible for any damages or losses incurred from its use.

## License
Copyright 2024 Donald Peltier

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

