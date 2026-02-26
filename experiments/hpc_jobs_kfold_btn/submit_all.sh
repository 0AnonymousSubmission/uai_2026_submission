#!/bin/bash
# Submit all K-Fold BTN jobs to HPC cluster
# Generated from: experiments/configs/kfold_best_specs_btn.json

mkdir -p logs

bsub < job_kfold_btn_abalone_btt.sh
bsub < job_kfold_btn_abalone_cpd.sh
bsub < job_kfold_btn_abalone_lmpo2.sh
bsub < job_kfold_btn_abalone_mpo2.sh
bsub < job_kfold_btn_ai4i_cpd.sh
bsub < job_kfold_btn_ai4i_lmpo2.sh
bsub < job_kfold_btn_ai4i_mpo2.sh
bsub < job_kfold_btn_appliances_lmpo2.sh
bsub < job_kfold_btn_appliances_mpo2.sh
bsub < job_kfold_btn_bike_btt.sh
bsub < job_kfold_btn_bike_cpd.sh
bsub < job_kfold_btn_bike_lmpo2.sh
bsub < job_kfold_btn_bike_mpo2.sh
bsub < job_kfold_btn_concrete_btt.sh
bsub < job_kfold_btn_concrete_cpd.sh
bsub < job_kfold_btn_concrete_lmpo2.sh
bsub < job_kfold_btn_concrete_mpo2.sh
bsub < job_kfold_btn_energy_efficiency_btt.sh
bsub < job_kfold_btn_energy_efficiency_cpd.sh
bsub < job_kfold_btn_energy_efficiency_lmpo2.sh
bsub < job_kfold_btn_energy_efficiency_mpo2.sh
bsub < job_kfold_btn_obesity_btt.sh
bsub < job_kfold_btn_obesity_lmpo2.sh
bsub < job_kfold_btn_obesity_mpo2.sh
bsub < job_kfold_btn_realstate_btt.sh
bsub < job_kfold_btn_realstate_cpd.sh
bsub < job_kfold_btn_realstate_lmpo2.sh
bsub < job_kfold_btn_realstate_mpo2.sh
bsub < job_kfold_btn_seoulBike_btt.sh
bsub < job_kfold_btn_seoulBike_cpd.sh
bsub < job_kfold_btn_seoulBike_lmpo2.sh
bsub < job_kfold_btn_seoulBike_mpo2.sh
bsub < job_kfold_btn_student_perf_btt.sh
bsub < job_kfold_btn_student_perf_lmpo2.sh
bsub < job_kfold_btn_student_perf_mpo2.sh
