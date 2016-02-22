# Holographic Embeddings of Knowledge Graphs

This repository holds the code for experiments in the paper 

```
Holographic Embeddings of Knowledge Graphs
Maximilian Nickel, Lorenzo Rosasco, Maximilian Nickel, AAAI 2016.
```

## Install 

To run the experiments, first install [scikit-kge](https://github.com/mnick/scikit-kge),
An open-source python library to compute various knowledge graph embeddings including

- Holographic Embeddings (HolE)
- RESCAL
- TransE
- TransR
- ER-MLP

After `scikit-kge` is installed, simply clone this repository via 

```
git clone git@github.com:mnick/holographic-embeddings.git
```

and run the experiments as detailed in the next section

## Experiments 

The repository holds scripts of the form 

```
run_<model>_<dataset>.sh
```

which runs the experiments for `dataset` with the best parameters for `model`.

The full code for the experiments can be found in the `kg` and `countries` subfolders. The python scripts in these subfolders should be easy to use for grid search.
