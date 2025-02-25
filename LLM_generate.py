# Import the necessary modules from the transformers library
from transformers import AutoTokenizer,AutoModelForCausalLM 

'''
Generacion de texto ¿Qué hace?
--- Usa generate() para crear texto nuevo a partir de input_ids.
--- Autoregresivo: Predice token por token hasta max_length=50 o hasta un token de finalización.
--- Se usa en tareas de completado de texto.
'''

# Load the GPT-2 tokenizer and the GPT-2 language model
tokenizer = AutoTokenizer.from_pretrained("gpt2")  
gpt2 = AutoModelForCausalLM.from_pretrained("gpt2")

# Define the input sentence
sentence = "The future of AI is" 

# Tokenize the sentence and convert it into tensor format (PyTorch)
# The process: words --> tokens --> unique token IDs
# Each word or subword is mapped to a specific token ID
input_ids = tokenizer(sentence, return_tensors='pt').input_ids  
#print('input_id: ', input_ids)  

# Generar texto
# es un tensor de PyTorch con los identificadores de los tokens generados por el modelo.
output2 = gpt2.generate(input_ids, max_length=50)
print('output2-> ',output2)

# DECODIFICAR AMBOS TEXTOS
print("Input text->", tokenizer.decode(input_ids[0]))
#Para convertir el tensor en texto entendible, usamos tokenizer.decode()
generated_text2 = tokenizer.decode(output2[0], skip_special_tokens=True)
print("Generated text2-> ",generated_text2)