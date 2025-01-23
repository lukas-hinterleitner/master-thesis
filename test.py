dot_products_without_layers = {}

for paraphrased_id in ["lima_0"]:
    dot_products_without_layers[paraphrased_id] = {}
    for original_id in ["lima_0", "lima_451", "lima_266", "lima_947", "lima_110"]:
        print(paraphrased_id, original_id)
        dot_products_without_layers[paraphrased_id][original_id] = 0

import pprint
pprint.pprint(dot_products_without_layers)
