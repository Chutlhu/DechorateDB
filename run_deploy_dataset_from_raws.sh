#!/usr/bin/env bash

# Paths
path_to_data_dir="/home/dicarlod/Documents/Dataset/dEchorate"
outdir="./outputs_16k"

mkdir -p $outdir

path_to_database="${outdir}/dEchorate_database.csv"
path_to_calibrated_positions_notes="${outdir}/dEchorate_calibrated_elements_positions.csv"
path_to_chirps="${outdir}/dEchorate_chirp.h5"

# # # Annotation files to database
# python dechorate/main_geometry_from_measurements.py --outdir ${outdir} --datadir $path_to_data_dir

# # # Final calibrated geometry
# python dechorate/main_geometry_from_echo_calibration.py --outdir ${outdir}

# # # Build the complete database with recordings 
# python dechorate/main_build_annotation_database.py --outdir ${outdir} --datadir $path_to_data_dir --calibnote $path_to_calibrated_positions_notes

# # # Build sound dataset: from zips to hdf5
# python dechorate/main_build_sound_datasets.py --outdir ${outdir} --signal speech  --fs 16000 --datadir $path_to_data_dir --dbpath $path_to_database --comp 7
# python dechorate/main_build_sound_datasets.py --outdir ${outdir} --signal chirp   --fs 48000 --datadir $path_to_data_dir --dbpath $path_to_database --comp 7
# python dechorate/main_build_sound_datasets.py --outdir ${outdir} --signal silence --fs 16000 --datadir $path_to_data_dir --dbpath $path_to_database --comp 7
# python dechorate/main_build_sound_datasets.py --outdir ${outdir} --signal babble  --fs 16000 --datadir $path_to_data_dir --dbpath $path_to_database --comp 7
# python dechorate/main_build_sound_datasets.py --outdir ${outdir} --signal noise   --fs 16000 --datadir $path_to_data_dir --dbpath $path_to_database --comp 7
# # echo "you may want to delete the content of .cache folder"

# # # # Estimate RIRs
# python dechorate/main_estimate_rirs.py --outdir ${outdir} --dbpath ${path_to_database} --chirps ${path_to_chirps} --comp 7

# Convert it into Sofa format
mkdir -p "${outdir}/sofa/"
python dechorate/main_build_sofa_database.py \
                 --outdir "${outdir}/sofa/" \
                 --echo   "${outdir}/dEchorate_annotations.h5"\
                 --csv    "${outdir}/dEchorate_database.csv" \
                 --hdf    "${outdir}/dEchorate_rir.h5"

# # # Preprare deliverable
delivdir='./deliverable'
mkdir -p $delivdir
for signal in silence babble noise speech chirp rir; do
    cp ${outdir}/dEchorate_${signal}.h5 ${delivdir}/dEchorate_${signal}.h5
done
cp ${outdir}/dEchorate_database.csv    ${delivdir}/dEchorate_database.csv
cp ${outdir}/dEchorate_annotations.h5  ${delivdir}/dEchorate_annotations.h5
cp ./dechorate/main_geometry_from_echo_calibration.py ${delivdir}/main_geometry_from_echo_calibration.py
