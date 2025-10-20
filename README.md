# disulfiNNate

an edge conditioned graph convolutional network trained to predict possible disulfide-bridge positions in proteins.

So far, this neural network is trained on the CATH-s40 database.

```
arguments:
  -h, --help            shows help message and exit
  -i, --input           path to the input pdb file
  -o, --ouput           output file name. default=out.csv
  -m, --model           model name. default=v0.keras
  --fastrelax           executes fastrelax before the relaxation
```

The output.csv contains the residues pairs with the corresponding probability.

### docker execution

here is how you can build the docker image:
```
docker build -t disulfinnate:v1 .
```

the following command executes a test run:
```
docker run --rm -v $(pwd):/disulfiNNate/data disulfinnate:v1 python disulfiNNate.py -i test/3ft7.pdb -o test/out.csv
```