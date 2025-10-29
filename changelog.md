# Changelog
All notable changes to this project will be documented here.

## [v2.0: model_predictions fixes] - 2025-10-28
model_predictions:
- fixed 5 runtime error cells and other design flaws/bugs, streamlined code in data exploration section (section 2)

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

data_processors/load_processed_data:
- Fixed missing-file bug in master_cleaning_and_saving (no more undefined return)

data_processors/data_splitter:
- Enabled batch shuffling for train loader

## [v0.0: data processing baseline] - 2025-9
- Uploaded baseline data processing code from original pipeline.
