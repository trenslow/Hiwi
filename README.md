# NemexOIE

This repository contains scripts used to evaluate the NemexOIE system.

## System Requirements

The scripts were developed and tested on Python 3.6. The evaluate.py script requires the matplotlib library for graphing purposes and the graph.py script requires neo4j to be installed on the system, as well as the neo4j Python driver.

## Installation

Installation instructions for Matplotlib can be found [here](https://matplotlib.org/users/installing.html).

Installation instructions for neo4j can be found [here](https://neo4j.com/docs/operations-manual/current/installation/).

Installation instructions for the neo4j Python driver can be found [here](https://neo4j.com/developer/python/).

## Running the System

### Evaluation

The following command will run the evaluation of the NemexOIE system:

```
python evaluate.py
```

This script compares the system's outputs to the results of other benchmark OIE systems and presents them graphically in the same way as Del Corro and Gemulla (2013). It will display the graphs as the script is run, but will also save the graphs in the corresponding 'nemexOutputs' directory for each data set.

### Creating Graph Database with Nemex Outputs

The Nemex outputs can also be represented graphically using the graph database neo4j. First, the neo4j server needs to be started, which can be done with the following command:

```
location_of_neo4j_installation/neo4j-community-3.x/bin/neo4j start
```

Please refer to the neo4j documentation for issues regarding starting the server.

Before running the graph.py script, wait a minute or so for the local neo4j server to boot up, otherwise the script will crash. The graph.py script can be run with the following command:

```
python graph.py file_with_relations.txt
```
For now, the script assumes that the relation arguments are tab-separated and it only graphs binary relations. Script will crash if no argument is provided.

To view the graph database, navigate to 'http://localhost:7474/browser/' in your favorite browser. On the first run, it may ask for a username and password. Please set these equal to the variables of the same name in the graph.py script. Now you can view the relations in your browser, as well as submit Cypher queries to retrieve necessary information about the relations. Refer to neo4j's documentation for more detailed information.

When finished, please run the following command to shut down the local neo4j server:

```
location_of_neo4j_installation/neo4j-community-3.x/bin/neo4j stop
```

## Citation

> Del Corro, Luciano, and Rainer Gemulla. "Clausie: clause-based open information extraction." Proceedings of the 22nd international conference on World Wide Web. ACM, 2013.