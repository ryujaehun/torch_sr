#!/bin/bash
		for j in {2..4}
			do
		 CUDA_VISIBLE_DEVICES=$j-1 python3 train.py --cuda -m 5 -u $j
			done

