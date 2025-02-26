from utils import device,prompt1, prompt2,prompt3,prompt4,prompt5,prompt6
from imports import *

'''
1. Selección de un LLM base pre-entrenado
'''
# Importamos el tokenizador
tokenizer_T5 = T5Tokenizer.from_pretrained("t5-base") # LLM base preentrenado
# Importamos el modelo pre-entrenado
model_T5 = T5ForConditionalGeneration.from_pretrained("t5-base", device_map="auto")

# Tokenizamos el prompt
prompt_tokens = tokenizer_T5(prompt1, return_tensors="pt").input_ids.to(device)

# Generamos los siguientes tokens
outputs = model_T5.generate(prompt_tokens, max_length=100) # Coge el prompt que se le ha pasado y continua haciendo cosas pero no hace funciones

# Transformamos los tokens generados en texto
'''
Presenta el comportamiento de un LLM base, no han sido entrenados para ninguna tarea especifica, ni para conversacion, ni para responder preguntas. 
Sólo presentan el comportamiento primitivo de un LLM, que es generar la siguiente palabra, a partir del prompt que se le indica genera la siguiente palabra,
y continua generando palabras que en este caso coinciden en función del contexto con lo que se habia puesto anteriormente
'''
print(tokenizer_T5.decode(outputs[0]))


'''
2. Selección de un Fine-tuned LLM
'''

'''Lectura del modelo y tokenizador'''
# Importamos el tokenizador
tokenizer_FT5 = T5Tokenizer.from_pretrained("google/flan-t5-base")

# Importamos el modelo pre-entrenado
model_FT5 = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base", device_map="auto")

# Tokenizamos el prompt
prompt_tokens = tokenizer_T5(prompt1, return_tensors="pt").input_ids.to(device)

'''Generación de texto'''
# Generamos los siguientes tokens
outputs = model_FT5.generate(prompt_tokens, max_length=50)

# Transformamos los tokens generados en texto
print(tokenizer_FT5.decode(outputs[0]))


'''
3. Selección de un Fine-tuned LLM de 1.000 millones de parámetros
'''

'''Lectura del modelo y tokenizador'''

# Importamos el tokenizador
tokenizer_FT5 = T5Tokenizer.from_pretrained("google/flan-t5-large")

# Importamos el modelo pre-entrenado
model_FT5 = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large", device_map="auto")

'''Generación de texto'''
# Tokenizamos el prompt
prompt_tokens = tokenizer_FT5(prompt1, return_tensors="pt").input_ids.to(device)

# Generamos los siguientes tokens
outputs = model_FT5.generate(prompt_tokens, max_length=100)

# Transformamos los tokens generados en texto
print(tokenizer_FT5.decode(outputs[0]))