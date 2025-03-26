# import google.generativeai as genai
# import os
# from dotenv import load_dotenv

# load_dotenv()
# genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# for m in genai.list_models():
#     print(m.name, "→", m.supported_generation_methods)

# import sys
# print("Python path being used:", sys.executable)

# from datasets import Dataset
# Dataset.cleanup_cache_files()

import torch
print(torch.cuda.is_available())