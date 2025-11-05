# Changelog
All notable changes to this project will be documented here.

## [v2.0: model_predictions fixes] - 2025-10-28
model_predictions:
- Section 2 (data loading and exploration): Reduced runtime error rate from 21% -> 0% ([# cells resulting in runtime error]/[# total cells])
- fixed unsafe CUDA checks; added AllenSDK hooks; standardized data load + logging

## [v1.1: model_predictions baseline] - 2025-10-24
- set up UNC's Longleaf HPC jupyter virtual environment via Open Ondemand, copied repo and authenticated git, gitignore'd large output folder.

model_predictions:
- Uploaded baseline jupyter notebook from original pipeline that does data exploration and model predictions

pictures:
- Uploaded baseline saved images folder.

## [v1.0: data processing fixes] - 2025-10
data_processors/pull_and_process data:
- Fixed GPU→CPU offload & caching in batch processing; eliminated redundant device moves
- Corrected tensor handling; fixed .cpu() call; robust DataFrame build
- Simplified frame assignment
- Cleaned master_function
- Performance:
  - Speed: 2411.713s -> 1011.369s, ~1.20 it/s -> ~2.48 it/s
  - Memory: allocated - ~0.41/0.59 GB -> ~0.19/0.24 GB; cached - ~0.52/0.75 GB -> ~0.24/0.31 GB

data_processors/load_processed_data:
- Fixed missing-file bug in master_cleaning_and_saving (no more undefined return)

data_processors/data_splitter:
- Enabled batch shuffling for train loader

## [v0.0: data processing baseline] - 2025-9
- Uploaded baseline data processing code from original pipeline.
