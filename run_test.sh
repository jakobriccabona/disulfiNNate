#!/bin/bash

docker run --rm -v $(pwd):/disulfiNNate/data disulfinnate:v1 python disulfiNNate.py \
 -i data/3ft7.pdb --csv data/out.csv