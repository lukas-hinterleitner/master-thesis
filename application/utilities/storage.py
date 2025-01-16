import os
import json

from .config.storage import (
    gradient_similarity_storage_path,
    gradient_similarity_bm25_selected_storage_path,
    gradient_similarity_bm25_selected_model_generated_storage_path,

    dot_product_storage_path,
    dot_product_bm25_selected_storage_path,
    dot_product_bm25_selected_model_generated_storage_path
)

from .config.dataset import SAMPLE_SIZE
from .config.model import MODEL_NAME

def get_gradient_similarity_file_path():
    path = str(os.path.join(gradient_similarity_storage_path, MODEL_NAME))

    if not os.path.exists(path):
        os.makedirs(path)

    return os.path.join(path, f"sample_size_{SAMPLE_SIZE}.csv")

def get_gradient_similarity_bm25_selected_file_path():
    path = str(os.path.join(gradient_similarity_bm25_selected_storage_path, MODEL_NAME))

    if not os.path.exists(path):
        os.makedirs(path)

    return os.path.join(path, f"sample_size_{SAMPLE_SIZE}.json")

def get_gradient_similarity_bm25_selected_model_generated_file_path():
    path = str(os.path.join(gradient_similarity_bm25_selected_model_generated_storage_path, MODEL_NAME))

    if not os.path.exists(path):
        os.makedirs(path)

    return os.path.join(path, f"sample_size_{SAMPLE_SIZE}.json")


def get_dot_product_bm25_selected_file_path(filename, model_name = MODEL_NAME, sample_size = SAMPLE_SIZE):
    path = os.path.join(dot_product_bm25_selected_storage_path, str(model_name), "sample_size", str(sample_size))

    if not os.path.exists(path):
        os.makedirs(path)

    return os.path.join(path, f"{filename}.json")

def get_dot_product_bm25_selected_files(model_name = MODEL_NAME, sample_size = SAMPLE_SIZE) -> tuple[dict[str, dict[str, dict[str, float]]], dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    path = os.path.join(dot_product_bm25_selected_storage_path, str(model_name), "sample_size", str(sample_size))

    # load dot_products.json
    with open(os.path.join(path, "dot_products.json")) as f:
        dot_products = json.load(f)
        f.close()

    # load paraphrased_dot_products.json
    with open(os.path.join(path, "paraphrased_dot_products.json")) as f:
        paraphrased_dot_products = json.load(f)
        f.close()

    # load original_dot_products.json
    with open(os.path.join(path, "original_dot_products.json")) as f:
        original_dot_products = json.load(f)
        f.close()

    return dot_products, paraphrased_dot_products, original_dot_products

def get_dot_product_bm25_selected_model_generated_file_path(filename, model_name = MODEL_NAME, sample_size = SAMPLE_SIZE):
    path = os.path.join(dot_product_bm25_selected_model_generated_storage_path, str(model_name), "sample_size", str(sample_size))

    if not os.path.exists(path):
        os.makedirs(path)

    return os.path.join(path, f"{filename}.json")

def get_dot_product_file_path(filename, model_name = MODEL_NAME, sample_size = SAMPLE_SIZE):
    path = os.path.join(dot_product_storage_path, str(model_name), "sample_size", str(sample_size))

    if not os.path.exists(path):
        os.makedirs(path)

    return os.path.join(path, f"{filename}.json")
