model="CNRM-ALADIN63_CNRM-CERFACS-CNRM-CM5_r1i1p1"
# model="CNRM-ALADIN63_MOHC-HadGEM2-ES_r1i1p1"
# model="CNRM-ALADIN63_NCC-NorESM1-M_r1i1p1"
# model="CNRM-ALADIN63_MPI-M-MPI-ESM-LR_r1i1p1"

# model="MOHC-HadREM3-GA7-05_CNRM-CERFACS-CNRM-CM5_r1i1p1"
# model="MOHC-HadREM3-GA7-05_MOHC-HadGEM2-ES_r1i1p1"
# model="MOHC-HadREM3-GA7-05_MPI-M-MPI-ESM-LR_r1i1p1"
# model="MOHC-HadREM3-GA7-05_NCC-NorESM1-M_r1i1p1"
# model="MOHC-HadREM3-GA7-05_ICHEC-EC-EARTH_r12i1p1"

scenarios=("rcp26" "rcp45" "rcp85")

# model=("None")
# scenarios=("None")
current_date=$(date +"%Y%m%d")
for scenario in "${scenarios[@]}"; do
    # Create a directory for logs if it doesn't exist
    if [ "$scenario" = "None" ] || [ "$model" = "None" ]; then
        folder="obs"
        scenario_arg=""
        model_arg=""
    else
        folder="${model}_${scenario}"
        scenario_arg="--climate_scenario $scenario"
        model_arg="--climate_model_chain $model"
    fi
    mkdir -p "logs/${folder}"
    echo "Processing model: $model, scenario: $scenario"

    nice -n 10 python -u -m src.data_processing \
        $model_arg \
        $scenario_arg \
        --weighted_sum 2>&1 | tee "logs/${folder}/processing_prevah_hydropower_${current_date}.log"
done