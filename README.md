# Node2Vec
This directory is an implementation of multilayer Node2Vec algorithm and a normal version of Node2Vec. Moreover, it contains link prediction tests we used to verify the result
## Build
Need tensorflow, sklearn, networkx library in python 3
## Running the modele
Run the test with the tester.py with a edge list, an output file, and a flag to decide whether to run the second test. Of course there are many variables you can input e.g. number of walks, walk length, probability p for biased sampling etc.
```
python3 tester.py <--graph Graph file> <--directory directory> <--test>
```
### Input
The input of the python script should be a pickle file in which should be a networkx graph object.
#### running the multilayer link predictino tests on the input directory where should contain multiple layers of graphs
```
python3 tester.py --directory data_set/ --NM
```
### Output
The output will be the AUC and the Accuracy and f-measure of the data-set trained by the algorithm.
