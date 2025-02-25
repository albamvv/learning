# Import the necessary modules from the transformers library
from transformers import AutoTokenizer,AutoModelForCausalLM 
import torch

'''
Llamar al modelo directamente¿Qué hace?
--- NO genera texto, sino que devuelve las logits del modelo.
--- Usado para análisis del modelo, como calcular probabilidades de los siguientes tokens.
--- Es útil para entrenar modelos o analizar sus predicciones manualmente.
'''

# Load the GPT-2 tokenizer and the GPT-2 language model
tokenizer = AutoTokenizer.from_pretrained("gpt2")  
# Este modelo es autoregresivo, lo que significa que puede predecir el siguiente token basado en el contexto anterior
gpt2 = AutoModelForCausalLM.from_pretrained("gpt2")

# Define the input sentence
sentence = "The future of AI is" 

# Tokenize the sentence and convert it into tensor format (PyTorch)
# The process: words --> tokens --> unique token IDs --> vector embed
# Each word or subword is mapped to a specific token ID
input_ids = tokenizer(sentence, return_tensors='pt').input_ids  
#print('input_id: ', input_ids)  

# Loop through each token ID in the tensor and decode it back to a string
#for token_id in input_ids[0]:  
#    print(tokenizer.decode(token_id))  # Print the corresponding token  


# Pass the tokenized input to the model to obtain output logits, Devuelve logits con las probabilidades de cada token.
output = gpt2(input_ids)  

# Print the shape of the output tensor, which represents model predictions
# Los logits son valores sin procesar que indican la probabilidad de cada token.
# Son un tensor de 3 dimensiones con tamaño (batch, tokens, vocabulario).
# Se convierten en probabilidades con softmax para predecir la siguiente palabra.
print('output tensor shape-> ', output.logits.shape) 
#print('logits-> ',output.logits)
#print('output logic: ',output.logits)
final_logits = gpt2(input_ids).logits[0,-1] 
#print('final logits-> ',final_logits)

print("Input text->", tokenizer.decode(input_ids[0]))

print(final_logits.argmax()) # Token ID <--> Index Location Logits
next_token= tokenizer.decode(final_logits.argmax())
print('next token-> ',next_token)

# TOP TEN PREDICTIONS
top_10_logits = torch.topk(final_logits,10)
for index in top_10_logits.indices:
    print(tokenizer.decode(index))


#next_token = torch.argmax(output.logits[:, -1, :], dim=-1)
#print('next token-> ',tokenizer.decode(next_token))








