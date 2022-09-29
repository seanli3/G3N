# G3N: Graph Neighbourhood Neural Networks

## Requirements
As listed in `requirements.txt`:
```
networkx==2.7.1
numpy==1.21.5
ogb==1.3.3
scikit_learn==1.1.0
scipy==1.7.3
torch==1.11.0
torch_geometric==2.0.4
torch_scatter==2.0.9
tqdm==4.64.0
```

Due to an [issue](https://github.com/snap-stanford/ogb/issues/329) associated with importing the ogb package, you may need to run in your environment `pip uninstall setuptools`

## Synthetic datasets
For isomorphism tasks on EXP, SR25, graph8c and CSL, run 
```
python3 iso.py --dataset exp --t <t> --d <d>
python3 iso.py --dataset sr25 --t <t> --d <d>
python3 iso.py --dataset graph8c --t <t> --d <d>
python3 iso.py --dataset csl --t <t> --d <d>
```
where d and t are parameters of G3N denoting neighbourhood size and neighbourhood subgraph dimension.

For substructure counting tasks run 
```
python3 counting.py --ntask <n_task> --t <t> --d <d>
```
where `n_task` selects what substructure you want to count: 0: triangle, 1: tailed_triangle; 2: star; 3: 4-cycle.

## Real world datasets
For TU datasets, you may look into `tu.py` or run the grid search by
```
python3 grid_tu.py --t 2 --d 1
```

For graph classification on MolHIV, run
```
python3 mol.py --t 3 --d 3
```

For graph regression on ZINC, run 
```
python3 zinc.py --t 3 --d 3
```

Note that the default parameters in the `.py` files may not be the optimal hyperparameter configurations. Please refer to the paper or supplementary material for more information on hyperparameter selection.
