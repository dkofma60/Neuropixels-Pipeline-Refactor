# Changelog
All notable changes to this project will be documented here.

## [v4.0: Final Reports] - 2025-12-16
model_predictions:
- ran analysis for final report (Neuropixels-Pipeline-Refactor/docs/RTG_Networks_Final_Report___Daniel_Kofman.pdf)

cLSTM:
- ran analysis for final report (Neuropixels-Pipeline-Refactor/docs/cLSTM_final_reportpdf)

## [v3.0: LSTM integration] - 2025-11-25
model_predictions:
- modularized LSTM and data processing

models/LSTM:
- uploaded refactored vesion of baseline LSTM architecure

models/LSTM:
- created cLSTM, a modifed LSTM archteceture with a new module that injects learned regularization patterns defined by a multivariate distribution

data_processors/data_splitter_new
- fixed major data misalignent and shuffling bugs by creating new data splitter version with random shuffling and stratifcation without corrupting test and validation sets

Added performance testing in Neuropixels-Pipeline-Refactor/tests.

## [v2.0: model_predictions fixes] - 2025-10-28
model_predictions:
- Section 2 (data loading and exploration): Reduced runtime error rate from 21% -> 0% ([# cells resulting in runtime error]/[# total cells])
- fixed unsafe CUDA checks; added AllenSDK hooks; standardized data load + logging

## [v1.1: model_predictions baseline] - 2025-10-24
- set up UNC's Longleaf HPC jupyter virtual environment via Open Ondemand, copied repo and authenticated git, gitignore'd large output folder.

model_predictions:
- Uploaded baseline jupyter notebook from original pipeline that does data exploration and model predictions

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
