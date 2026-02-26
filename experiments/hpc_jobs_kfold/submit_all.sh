#!/bin/bash
# Submit all K-Fold jobs to HPC cluster
# Generated from: experiments/configs/kfold_best_specs_als.json

mkdir -p logs

bsub < job_kfold_abalone_btt.sh
bsub < job_kfold_abalone_cpd.sh
bsub < job_kfold_abalone_lmpo2.sh
bsub < job_kfold_abalone_mpo2.sh
bsub < job_kfold_ai4i_cpd.sh
bsub < job_kfold_ai4i_lmpo2.sh
bsub < job_kfold_ai4i_mpo2.sh
bsub < job_kfold_appliances_cpd.sh
bsub < job_kfold_appliances_lmpo2.sh
bsub < job_kfold_appliances_mpo2.sh
bsub < job_kfold_bike_btt.sh
bsub < job_kfold_bike_cpd.sh
bsub < job_kfold_bike_lmpo2.sh
bsub < job_kfold_bike_mpo2.sh
bsub < job_kfold_concrete_btt.sh
bsub < job_kfold_concrete_cpd.sh
bsub < job_kfold_concrete_lmpo2.sh
bsub < job_kfold_concrete_mpo2.sh
bsub < job_kfold_energy_efficiency_btt.sh
bsub < job_kfold_energy_efficiency_cpd.sh
bsub < job_kfold_energy_efficiency_lmpo2.sh
bsub < job_kfold_energy_efficiency_mpo2.sh
bsub < job_kfold_obesity_btt.sh
bsub < job_kfold_obesity_cpd.sh
bsub < job_kfold_obesity_lmpo2.sh
bsub < job_kfold_obesity_mpo2.sh
bsub < job_kfold_realstate_btt.sh
bsub < job_kfold_realstate_cpd.sh
bsub < job_kfold_realstate_lmpo2.sh
bsub < job_kfold_realstate_mpo2.sh
bsub < job_kfold_seoulBike_btt.sh
bsub < job_kfold_seoulBike_cpd.sh
bsub < job_kfold_seoulBike_lmpo2.sh
bsub < job_kfold_seoulBike_mpo2.sh
bsub < job_kfold_student_perf_btt.sh
bsub < job_kfold_student_perf_cpd.sh
bsub < job_kfold_student_perf_lmpo2.sh
bsub < job_kfold_student_perf_mpo2.sh
