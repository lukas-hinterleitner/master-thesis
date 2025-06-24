from dotenv import load_dotenv
import os

if not load_dotenv():
    raise Exception("Couldn't load .env file. Please make sure that the .env file exists according to the README.md file.")

if not os.getenv("MT_MODEL_NAME"):
    raise Exception("MT_MODEL_NAME is not set in the .env file. Please make sure that the .env file exists according to the README.md file.")

MODEL_NAME = os.getenv("MT_MODEL_NAME")