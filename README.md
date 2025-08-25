# Simulation Framework for CDW

This repository supports the simulation and evaluation of collaborative document writing (CDW) using agent-based models. It was developed as part of an academic study exploring collective constitutional writing with diverst agent populations.

## Directory Structure

- **agents/**  
  Contains implementations of the three `Agent` types used in the study: unsturactured, Euclidiean, and LLM-based agents. 

- **datasets/**  
  Input data for population profiles, including demographic distributions and syntethic dataset on climate actions.

- **events/**  
  Defines the `Event` and `Event List` classes and handles generation of event lists used to simulate sequential agent actions. Includes all the functions discussed in the paper.

- **figures/**  
  Output directory for all figures produced during simulation, including Pareto frontiers, coverage plots, and comparative analyses.

- **paragraphs/**  
  Implements `Paragraph` objects used as candidate proposals in the document. Includes basic and interval paragraph types.

- **results/**  
  Stores all simulation results, intermediate outputs, and serialized objects generated during runs. Due to space limitation, only communities information and figures are included.

- **rules/**  
  Contains rule configuration files (static and dynamic CSFs). 

- **simulation/**  
  Core logic for `Simulation Configuration`, `Scheduler`, metrics evaluation and aggregation. 

- **tests/**  
  Unit tests and validation scripts for major components in the pipeline.

-**Event_List_Example_5/**
    The scheduled event list presented in the paper as example 5 with JSON files describing its agents, events, paragraphs. Note that there is a log file of API calls and response.


## Usage

To run the main simulation pipeline we used parallel computation with two scripts, one fore llm agents and another fo the rest.