model="CNRM-ALADIN63_CNRM-CERFACS-CNRM-CM5_r1i1p1"
# model="CNRM-ALADIN63_MOHC-HadGEM2-ES_r1i1p1"
# model="CNRM-ALADIN63_NCC-NorESM1-M_r1i1p1"
# model="CNRM-ALADIN63_MPI-M-MPI-ESM-LR_r1i1p1"

# model="MOHC-HadREM3-GA7-05_CNRM-CERFACS-CNRM-CM5_r1i1p1"
# model="MOHC-HadREM3-GA7-05_MOHC-HadGEM2-ES_r1i1p1"
# model="MOHC-HadREM3-GA7-05_MPI-M-MPI-ESM-LR_r1i1p1"
# model="MOHC-HadREM3-GA7-05_NCC-NorESM1-M_r1i1p1"
# model="MOHC-HadREM3-GA7-05_ICHEC-EC-EARTH_r12i1p1"

# scenarios=("rcp26" "rcp45" "rcp85")
# scenarios=("rcp26")
scenarios=("rcp45")
# scenarios=("rcp85")

for scenario in "${scenarios[@]}"; do
    # Create a directory for logs if it doesn't exist
    mkdir -p "logs/${model}_${scenario}"
    echo "Processing model: $model, scenario: $scenario"
    nice -n 10 python -m src.data_processing_projections \
        --climate_model_chain "$model" \
        --climate_scenario "$scenario" \
        --weighted_sum 2>&1 | tee "logs/${model}_${scenario}/processing_prevah_hydropower_${current_date}.log"
done