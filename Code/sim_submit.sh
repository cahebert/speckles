#!/bin/

for i in {1..40} # 6 8
do
seed=$i*50
simpath='/Users/clairealice/Documents/research/speckles/sim_cubes/sim_562_'$i'.fits'
parampath='/Users/clairealice/Documents/research/speckles/sim_cubes/sim_params_'$i'.p'

echo "starting sim number "$i
python simulate_speckle_psf.py --save_psfs_path=$simpath --color=562 --save_params_path=$parampath

done
