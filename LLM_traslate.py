# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
# Use a pipeline as a high-level helper
from transformers import pipeline

# Define the input sentence
sentence = "The future of AI is" 

# Tokenize the sentence and convert it into tensor format (PyTorch)
tokenizer = AutoTokenizer.from_pretrained("ModelSpace/GemmaX2-28-2B-v0.1")
model = AutoModelForCausalLM.from_pretrained("ModelSpace/GemmaX2-28-2B-v0.1")


#pipe = pipeline("translation", model="ModelSpace/GemmaX2-28-2B-v0.1")



