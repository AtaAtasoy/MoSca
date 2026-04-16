cd /home/atasoy/MoSca

export CUDA_VISIBLE_DEVICES=0
export MPLCONFIGDIR=/tmp/mpl

mkdir -p ./runs/vipe_cowboy_cat
ln -s /dss/dssmcmlfs01/pn25pi/pn25pi-dss-0000/di35dov/monst3r/demo_tmp/vipe_results/121frames/cowboy-cat_colmap/851422b7258e44a7ad04a9789be82299/images ./runs/vipe_cowboy_cat/images

./.venv/bin/python mosca_precompute.py \
  --cfg ./profile/vipe/cowboy_cat_prep_vipe_depth.yaml \
  --ws ./runs/vipe_cowboy_cat

./.venv/bin/python mosca_reconstruct.py \
  --cfg ./profile/vipe/cowboy_cat_fit_init.yaml \
  --ws ./runs/vipe_cowboy_cat
