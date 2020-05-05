#!/bin/sh
python3 create_data_script.py --train_size=5000
python3 create_data_script.py --train_size=10000
python3 create_data_script.py --train_size=50000
python3 create_data_script.py --train_size=100000
python3 create_data_script.py --train_size=200000
