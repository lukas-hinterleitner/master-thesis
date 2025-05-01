import os

data_folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../data")

lima_dataset_path = os.path.join(data_folder_path, "lima")
lima_paraphrased_dataset_path = os.path.join(data_folder_path, "paraphrased")

if not os.path.exists(lima_paraphrased_dataset_path):
    os.makedirs(lima_paraphrased_dataset_path)

gradient_similarity_storage_path = os.path.join(data_folder_path, "gradient_similarity")
gradient_similarity_bm25_selected_storage_path = os.path.join(data_folder_path, "gradient_similarity_bm25_selected")
gradient_similarity_bm25_selected_model_generated_storage_path = os.path.join(data_folder_path, "gradient_similarity_bm25_selected_model_generated")

if not os.path.exists(gradient_similarity_storage_path):
    os.makedirs(gradient_similarity_storage_path)

if not os.path.exists(gradient_similarity_bm25_selected_storage_path):
    os.makedirs(gradient_similarity_bm25_selected_storage_path)

if not os.path.exists(gradient_similarity_bm25_selected_model_generated_storage_path):
    os.makedirs(gradient_similarity_bm25_selected_model_generated_storage_path)

dot_product_storage_path = os.path.join(data_folder_path, "dot_products")
dot_product_bm25_selected_storage_path = os.path.join(dot_product_storage_path, "bm25_selected")
dot_product_bm25_selected_model_generated_storage_path = os.path.join(dot_product_storage_path, "bm25_selected_model_generated")

if not os.path.exists(dot_product_storage_path):
    os.makedirs(dot_product_storage_path)

if not os.path.exists(dot_product_bm25_selected_storage_path):
    os.makedirs(dot_product_bm25_selected_storage_path)

if not os.path.exists(dot_product_bm25_selected_model_generated_storage_path):
    os.makedirs(dot_product_bm25_selected_model_generated_storage_path)

results_folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../results")

if not os.path.exists(results_folder_path):
    os.makedirs(results_folder_path)