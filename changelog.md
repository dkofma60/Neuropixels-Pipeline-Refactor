# Changelog
All notable changes to this project will be documented here.

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
