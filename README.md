# EM-PLA
EM-PLA: Environment-aware Heterogeneous Graph-based Multimodal Protein-Ligand Binding Affinity Prediction
## Install
```bash
git clone git@github.com:littlemou22/EM-PLA.git
cd EM-PLA
```

## Dataset
This paper conducts experiments using the following datasets: the PDBbind 2016 dataset(http://pdbbind.org.cn/), the CASF-2013 dataset, the CSAR-HIQ (51) dataset, and the CSAR-HIQ (36) dataset. After downloading the datasets, they should be placed in the /data folder.

## Process Data
Before starting the training, please use `process.py` to convert the raw data into the format required by the model. The file paths in the code may need to be adjusted.
```bash
python process.py
```

## Run & Test
### Train your own model.
```bash
python train_lba.py
```

### Test your own model.
```bash
python test_lba.py
```

### Our trained model.
Our trained model is located in the `best_model.zip` folder. You can modify the model path in `test_lba.py` accordingly.
