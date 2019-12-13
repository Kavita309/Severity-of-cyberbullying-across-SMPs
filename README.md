# BTP_Project


## How to run the project?

#### Convert the CSV files in Dataset folder to PKL file using python file 'csv_to_pkl.pkl'

## Running Traditional Models
1. Set paths for PKL files of Formspring and Twitter in the traditional_ml.py.
2. Set model paramaters according to requirement ( Type_of_model, type_of_dataset, type_of_embedding)
3. In the command line line, run python traditional_ml.py. The output for the model can be seen on the command line.

## Running deep learning models

1. Download SSWE Embedding from http://ir.hit.edu.cn/~dytang/paper/sswe/embedding-results.zip and GLOVE embeddings from https://nlp.stanford.edu/projects/glove/'
2. Set paths for PKL files and embedding files of Formspring and Twitter in the deeplearning.py.
3.  Set model paramaters according to requirement ( Type_of_model, type_of_dataset, type_of_embedding)
4. In the command line line, run python deeplearning.py. The output for the model can be seen on the command line.

## Visualizing Results

Run file traditional_results_with_graph.py to see the plots for comparison between different types of models.


## Versions compatible with codebase

1. Python : 3.7.5
2. Tensorflow : 1.14
