# CS5340_Uncertainty_Modelling_in_AI
CS5340 AY2023/2024 Project Group 4

### Changes made from OOD code
In openood.utils, dataset_config.num_workers, .gpu, .num_machines have been commented out. The latter 2 is used
in deciding if a sampler from torch is necessary, and the former is used for determining number of subprocesses
to use when loading in data. For more information, visit https://pytorch.org/docs/stable/data.html 

Preprocessors are specific to datapipelines in the paper, for now we set default 
to base preprocessor and testpreprocessor. 

### Data
Data from openood is hosted in their wiki page. For this implementation please specifiy the absolute
path of the data folder when calling dataloaders(will change later if have time). 