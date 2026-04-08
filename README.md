Download MIMIC-IV from PhysioNet into data/mimic-iv/
Run python setup_data.py validate to confirm the data is in place
Run python setup_data.py seed to load public base rates
Run python setup_data.py train --n 500 to train the model on MIMIC cases
Run python setup_data.py eval --n 10 to evaluate the pipeline on held-out cases