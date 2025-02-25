# Import the necessary modules from the transformers library
from transformers import AutoTokenizer,AutoModelForCausalLM, pipeline
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
output_ids = gpt2.generate(input_ids, max_length=20)
print('output_ids-> ',output_ids)

# DECODIFICAR AMBOS TEXTOS
print("Input text->", tokenizer.decode(input_ids[0]))
#Para convertir el tensor en texto entendible, usamos tokenizer.decode()
#generated_text = tokenizer.decode(output_ids[0], repetition_penalty=1.9, skip_special_tokens=True)
generated_text = tokenizer.decode(output_ids[0], repetition_penalty=1.9, do_sample=True,top_k=5, top_p=0.94)
print("Generated text-> ",generated_text)

#help(gpt2)

# GENERATE TEXT USING PIPELINE

pipe = pipeline("text-generation", model="openai-community/gpt2", trust_remote_code=True)
generated_text2=pipe("I went to the happy store today and bought a")
print("Generated text2-> ",generated_text2)

