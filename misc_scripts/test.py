# # import google.generativeai as genai
# # import os
# # from dotenv import load_dotenv

# # load_dotenv()
# # genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# # for m in genai.list_models():
# #     print(m.name, "→", m.supported_generation_methods)

# # import sys
# # print("Python path being used:", sys.executable)

# # from datasets import Dataset
# # Dataset.cleanup_cache_files()

# # import torch
# # print(torch.cuda.is_available())

# # model_name = "bert-base-uncased"
# # print(model_name.split('-')[1])

# models = {
#                 "distilbert": "distilbert-base-uncased",
#                 "bert-base": "bert-base-uncased",
#                 "bert-large": "bert-large-uncased",
#                 "bert-base-cased": "bert-base-cased",
#                 "bert-large-cased": "bert-large-cased",
#                 "roberta-base": "roberta-base",
#                 "roberta-large": "roberta-large",
#                 "albert": "albert-base-v2",
#                 "electra-base":	"google/electra-base-discriminator",
#                 "electra-large": "google/electra-large-discriminator",
#                 "deberta-base": "microsoft/deberta-v3-base",
#                 "deberta-large": "microsoft/deberta-v3-large",
#                 "xlnet-base": "xlnet-base-cased",
#                 "xlnet-large": "xlnet-large-cased",
#                 "camembert": "camembert-base",
#                 "xlm-roberta-base": "xlm-roberta-base",
#                 "xlm-roberta-large": "xlm-roberta-large",
#                 "flaubert": "flaubert-base-cased",
#                 "bart-base": "facebook/bart-base",
#                 "bart-large": "facebook/bart-large",
#                 "longformer-base": "allenai/longformer-base-4096"
#             }
# model = "xlma"
# if (model.lower() not in models.keys()) and (model.lower().split('-')[0] not in [i.split('-')[0] for i in models.keys()]):
#     print("Invalid case")
# else:
#     print("Valid case")
#     if model.lower() in models.keys():
#         model_name = models[model.lower()]
#     else:
#         model_name = [i for i in models.keys() if i.startswith(model)][0]
#     print(model_name)

model_name = "google/electra-base-discriminator"
# model_name = "bert-base-uncased"
# model_name = "facebook/bart-large"
# model_name = "roberta-base"
output_dir_model_name = model_name.replace("/", "-").split('-')
if len(output_dir_model_name) > 3:
    output_dir_model_name_1 = "_".join(output_dir_model_name[:2])
elif len(output_dir_model_name) > 2 and (output_dir_model_name[2] == "base" or output_dir_model_name[2] == "large"):
    output_dir_model_name_1 = "_".join(output_dir_model_name[:2])
else:
    output_dir_model_name_1 = "_".join(output_dir_model_name[:1])
output_dir_model_name_2 = "_".join(output_dir_model_name)
# print(output_dir_model_name_1)
# output_dir = f".\\models\\empathetic\\{model_name.split('-')[0]}\\emotion_{model_name.split('-')[0]}_{model_name.split('-')[1]}_finetuned"
output_dir = f".\\models\\empathetic\\{output_dir_model_name_1}\\emotion_{output_dir_model_name_2}_finetuned"
print(output_dir)