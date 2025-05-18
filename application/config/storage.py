import os
from enum import Enum

data_folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../data")

lima_dataset_path = os.path.join(data_folder_path, "lima")
lima_paraphrased_dataset_path = os.path.join(data_folder_path, "paraphrased")

if not os.path.exists(lima_paraphrased_dataset_path):
    os.makedirs(lima_paraphrased_dataset_path)


class ExperimentType(Enum):
    PARAPHRASED = "paraphrased"
    MODEL_GENERATED = "model_generated"

# gradient similarities

gradient_similarity_storage_path = os.path.join(data_folder_path, "gradient_similarity")
gradient_similarity_paraphrased_storage_path = os.path.join(gradient_similarity_storage_path, ExperimentType.PARAPHRASED.value)
gradient_similarity_model_generated_storage_path = os.path.join(gradient_similarity_storage_path, ExperimentType.MODEL_GENERATED.value)

if not os.path.exists(gradient_similarity_storage_path):
    os.makedirs(gradient_similarity_storage_path)

if not os.path.exists(gradient_similarity_paraphrased_storage_path):
    os.makedirs(gradient_similarity_paraphrased_storage_path)

if not os.path.exists(gradient_similarity_model_generated_storage_path):
    os.makedirs(gradient_similarity_model_generated_storage_path)

# gradient similarities random projection

gradient_similarity_random_projection_storage_path = os.path.join(gradient_similarity_storage_path, "random_projection")
gradient_similarity_random_projection_paraphrased_storage_path = os.path.join(gradient_similarity_random_projection_storage_path, ExperimentType.PARAPHRASED.value)
gradient_similarity_random_projection_model_generated_storage_path = os.path.join(gradient_similarity_random_projection_storage_path, ExperimentType.MODEL_GENERATED.value)

if not os.path.exists(gradient_similarity_random_projection_storage_path):
    os.makedirs(gradient_similarity_random_projection_storage_path)

if not os.path.exists(gradient_similarity_random_projection_paraphrased_storage_path):
    os.makedirs(gradient_similarity_random_projection_paraphrased_storage_path)

if not os.path.exists(gradient_similarity_random_projection_model_generated_storage_path):
    os.makedirs(gradient_similarity_random_projection_model_generated_storage_path)

# dot products

dot_product_storage_path = os.path.join(data_folder_path, "dot_products")
dot_product_paraphrased_storage_path = os.path.join(dot_product_storage_path, ExperimentType.PARAPHRASED.value)
dot_product_model_generated_storage_path = os.path.join(dot_product_storage_path, ExperimentType.MODEL_GENERATED.value)

if not os.path.exists(dot_product_storage_path):
    os.makedirs(dot_product_storage_path)

if not os.path.exists(dot_product_paraphrased_storage_path):
    os.makedirs(dot_product_paraphrased_storage_path)

if not os.path.exists(dot_product_model_generated_storage_path):
    os.makedirs(dot_product_model_generated_storage_path)

# results

results_folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../results")

if not os.path.exists(results_folder_path):
    os.makedirs(results_folder_path)