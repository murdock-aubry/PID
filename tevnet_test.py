import argparse, os, sys, glob
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
import matplotlib.pyplot as plt
import yaml
import sys
from ldm.modules.HADARNet.modules import HADARNet

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


config_path = "/w/246/murdock/PID/configs/latent-diffusion/tev_net_config.yaml"
output_path = "/w/246/murdock/PID/output_image/out_test.jpg"


with open(config_path, 'r') as file:
    config = yaml.safe_load(file)
tev_net_config = config['model']['tev_net_config']

tev_net = HADARNet(
    in_channels=tev_net_config['params']['in_channels'],
    out_channels=tev_net_config['params']['out_channels'],
    smp_model=tev_net_config['params']['smp_model'],
    smp_encoder=tev_net_config['params']['smp_encoder']
)

ckpt_path = tev_net_config['params']['ckpt_path']
checkpoint = torch.load(ckpt_path, map_location=device)
tev_net.load_state_dict(checkpoint['state_dict'], strict=False)
tev_net.eval()



# image = "/w/246/murdock/PID/prepared_image/leighia_test.jpg"
image = "/w/246/murdock/PID/output_image/set09_V000_I03159.png"
image = np.array(Image.open(image).convert("RGB")).astype(np.uint8)
image = image.astype(np.float32)
image = image[None].transpose(0,3,1,2)

# image = torch.from_numpy(image - np.min(image) / (np.max(image) - np.min(image)))
image = torch.from_numpy(image) * 2.0 - 1.0




temp = tev_net.tevnet(image)[:, 1, :, :]


# temp = (temp - np.min(temp)) / (np.max(temp) - np.min(temp))


temp_image = torch.clamp((temp+1.0)/2.0, min=0.0, max=1.0)
predicted_image = temp_image.cpu()

print(predicted_image)
quit()

cmap = plt.get_cmap('hsv')
colored_image = cmap(predicted_image.detach().numpy()) 



# colored_temp_image = plt.cm.hot(colored_image.detach().numpy())

colored_temp_image = (colored_image[0, :, :, :3] * 255).astype(np.uint8)
# predicted_image = predicted_image.detach().numpy()

Image.fromarray(colored_temp_image).save(output_path)

# Image.fromarray(predicted_image.astype(np.uint8)).save(outpath)


# print(temp[:, 1, :, :])



quit()

