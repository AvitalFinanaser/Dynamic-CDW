# Simulation Framework for CDW

This repository accompanies the paper  
**"A Dynamic Approach to Collaborative Document Writing"**, accepted for publication at **ECAI 2025** (European Conference on Artificial Intelligence, Bologna, Italy).  
It provides the extended version and supplementary materials for the paper, including detailed proofs, additional experiments, and the full simulation framework code.

## Directory Structure

- **agents/**  
  Contains implementations of the three `Agent` types used in the study: unstructured, Euclidean, and LLM-based agents. 

- **datasets/**  
  Input data for population profiles, including demographic distributions and a synthetic dataset on climate actions.

- **events/**  
  Defines the `Event` and `Event List` classes and handles the generation of event lists used to simulate sequential agent actions. Includes all the functions discussed in the paper.

- **figures/**  
  Output directory for all figures produced during simulation, including Pareto frontiers, coverage plots, and comparative analyses.

- **paragraphs/**  
  Implements `Paragraph` objects used as candidate proposals in the document. Includes basic and interval paragraph types.

- **results/**  
  Stores all simulation results, intermediate outputs, and serialized objects generated during runs. Due to space limitations, only community information and figures are included.

- **rules/**  
  Contains rule configuration files (static and dynamic CSFs). 

- **simulation/**  
  Core logic for `Simulation Configuration`, `Scheduler`, metrics evaluation and aggregation. 

- **tests/**  
  Unit tests and validation scripts for major components in the pipeline.

-**Event_List_Example_5/**
    The scheduled event list presented in the paper as example 5 with JSON files describing its agents, events, and paragraphs. Note that there is a log file of API calls and responses.


## Usage

To run the main simulation pipeline, we used parallel computation with two scripts, one for LLM agents and another for the rest.

## Citation

If you use this code or data, please cite:

Avital Finanser and Nimrod Talmon.  
*A Dynamic Approach to Collaborative Document Writing.*  
Proceedings of ECAI 2025.
