# mixencodings
# Genetic Algorithm for Mixed-Variable Optimization

This project demonstrates the implementation of a genetic algorithm for solving a mixed-variable optimization problem. The algorithm aims to minimize the total assignment cost by efficiently allocating tasks to workers while considering various types of decision variables.

## Problem Description

The problem involves assigning tasks to workers with the objective of minimizing the overall assignment cost. The algorithm employs a hybrid encoding strategy to represent assignment decisions. The decision variables include binary task assignments for mandatory tasks, estimated completion times, task permutations, and the number of optional tasks assigned.

## Features

- Hybrid encoding of mixed-variable decision-making scenarios.
- Implementation of a genetic algorithm to evolve solutions over generations.
- Utilization of a neural network model to predict the fitness of solutions.
- Correction of permutation assignments to ensure uniqueness.

## Getting Started

To run the implementation:

1. Install the required libraries: `numpy` and `tensorflow`.
2. Execute the provided Python script using a Python interpreter.

Feel free to experiment with parameters, fitness functions, and other aspects to adapt the algorithm to your specific optimization problem.

## Code Explanation

The repository contains the following files:

- `genetic_algorithm.py`: The main script that implements the genetic algorithm and hybrid encoding.
- `README.md`: This README file providing an overview of the project.

## Acknowledgments

This project is inspired by the field of optimization and genetic algorithms. It showcases how hybrid encoding can effectively address problems with mixed-variable decision-making scenarios.

Please note that this README provides a high-level overview. For detailed code and execution, refer to the actual implementation in the script.

For questions or contributions, feel free to open an issue or pull request!
