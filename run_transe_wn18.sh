#!/bin/sh

python kg/run_transe.py --fin data/wn18.bin \
       --test-all 50 --nb 100 --me 500 \
       --margin 2.0 --lr 0.1 --ncomp 50
