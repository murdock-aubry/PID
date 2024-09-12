CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 TORCH_USE_CUDA_DSA=1 python3 scripts/rgb2ir_vqf8.py --steps 400 \
--indir /w/246/murdock/PID/dataset/KAIST512/visible \
--outdir /w/246/murdock/PID/output_image \
--config /w/246/murdock/PID/configs/latent-diffusion/kaist512-vqf8.yaml \
--checkpoint /w/246/murdock/PID/'epoch=000235-step=000059999.ckpt' \
--ddim_eta 0.0