from dotenv import load_dotenv
import os
import torch

if not load_dotenv():
    raise Exception("Couldn't load .env file. Please make sure that the .env file exists according to the README.md file.")

if not os.getenv("MT_DEVICE"):
    raise Exception("MT_DEVICE is not set in the .env file. Please make sure that the .env file exists according to the README.md file.")

__device = os.getenv("MT_DEVICE")

if not __device.startswith("cuda") and not __device.startswith("cpu"):
    raise Exception("MT_DEVICE should start with 'cuda' or 'cpu'.")

if __device.startswith("cuda") and not torch.cuda.is_available():
    raise Exception("MT_DEVICE is set to 'cuda' but no GPU is available.")

if __device.startswith("cuda") and not __device.split(":")[1].isdigit():
    raise Exception("MT_DEVICE should be in the format 'cuda:{GPU_ID}'. Have a look in the README.md file for more information.")

if __device.startswith("cuda") and int(__device.split(":")[1]) >= torch.cuda.device_count():
    raise Exception("MT_DEVICE is set to a GPU that does not exist. Please make sure that the GPU ID is correct.")

DEVICE = torch.device(__device)