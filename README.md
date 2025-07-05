# OpenMAP-BrainAge

This repository is the anomymous folder, containing the code for the OpenMAP-BrainAge model. We will update more details later and please wait for the formal repository releasing out.

## Environments:

Please use anaconda and the following command to install the required environments and packages for running:

`conda env create -f environment.yml`

## Instructions:

To train the model, please refer to the file [train_ADNI_multiview.sh](./train_ADNI_multiview.sh), with the default setting of all hyperparameters. You have to modify the dataloader in [dataADNI_multiview.py](./dataADNI_multiview.py) file to fit into your data config format and data directory.
