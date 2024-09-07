CUDA_VISIBLE_DEVICES=0 python scripts/rgb2ir_vqf8.py --steps 200 \
--indir dataset/KAIST512/visible/set07_V001_I00039.png \
--outdir dataset/KAIST512/visible/out1.png \
--config configs/latent-diffusion/kaist512-vqf8.yaml \
--checkpoint /path/to/checkpoint \
--ddim_eta 0.0