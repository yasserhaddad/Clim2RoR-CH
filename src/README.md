# `src` folder of Clim2RoR-CH
This is the `src` folder of the repository linked to the paper "Impacts of climate variability on run-of-river hydropower and electricity systems planning in Switzerland".

The folder contains the following files:
- `readbin_wsl.py` contains the function to read the binary output files of the PREVAH model.
- `extract_runoff_prevah.py` contains all the functions to extract runoff data from the binary output files of the PREVAH model.
- `utils_polygons.py` contains utilitary functions to process polygons of Swiss catchments.
- `utils_streamflow_hydropower.py` contains utilitary functions to convert runoff/streamflow into hydropower generation.
- `var_attributes.py` contains the attributes of the xarray Datasets that are created in the pipeline.
- `paper_data_processing.py` contains a pipeline with all the necessary functions to process data from the extraction of data from binary output files of PREVAH, to the extraction of runoff from this data, to its processing and conversion into hydropower generation.
- `paper_match_wasta_nexuse.py` contains a pipeline with all the necessary functions to integrate the results into a database for Nexus-e simulations.
- `paper_analysis_hydropower.py` contains a pipeline with all the necessary functions to analyze the estimated hydropower generation results and produce its related figures of the paper and the supplementary material.
