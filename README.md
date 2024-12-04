# disulfiNNAte

an edge conditioned graph convolutional network trained to predict possible disulfide-bridge positions in proteins.

So far, this neural network is trained on the CATH-s40 database.

```
arguments:
  -h, --help            shows help message and exit
  -i, --input           path to the input pdb file
  -o, --ouput           output file name. default=out.csv
  -m, --model           model name. default=v0.keras
```

The output.csv contains the residues pairs with the corresponding probability.