CUDA_VISIBLE_DEVICES=0 \
python /w/246/murdock/PID/TeVNet/test.py \
--weights-file /w/246/murdock/PID/epoch_1000.pth \
--image-dir /w/246/murdock/PID/dataset/KAIST512/visible \
--smp_model Unet --smp_encoder resnet18 \
--output-dir /w/246/murdock/PID/output_image_decomposed/ \
--vnums 4