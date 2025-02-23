from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from prompt import prompt1, prompt2,prompt3,prompt4,prompt5,


'''
1. Selección de un LLM base pre-entrenado
'''


'''
T5Tokenizer: Para convertir texto en tokens que el modelo puede entender.
Se descarga y carga el tokenizador del modelo "t5-base" desde Hugging Face.
Permite tokenizar (convertir texto en números) y detokenizar (convertir números en texto).
Este LLM esta compuesto por 220 millones de parámetros y ha sido pre-entrenado en número elevado de conjuntos de datos:
'''
# Importamos el tokenizador
tokenizer_T5 = T5Tokenizer.from_pretrained("t5-base") # LLM base preentrenado

'''
T5ForConditionalGeneration: Modelo preentrenado de T5, diseñado para generar texto en base a una entrada.
Se carga el modelo T5-Base, con 220M de parámetros.
device_map="auto" permite asignar automáticamente el modelo a GPU o CPU, dependiendo de la disponibilidad.
'''
# Importamos el modelo pre-entrenado
model_T5 = T5ForConditionalGeneration.from_pretrained("t5-base", device_map="auto")

# Este código funcionará en CPU sin problemas. Si en el futuro activas CUDA, también lo usará automáticamente.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(f"Usando: {device}")


# Tokenizamos el prompt
#prompt_tokens = tokenizer_T5(prompt, return_tensors="pt").input_ids.to("cuda")
prompt_tokens = tokenizer_T5(prompt6, return_tensors="pt").input_ids.to(device)

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
prompt_tokens = tokenizer_FT5(prompt, return_tensors="pt").input_ids.to("cuda")

'''Generación de texto'''
# Generamos los siguientes tokens
outputs = model_FT5.generate(prompt_tokens, max_length=50)

# Transformamos los tokens generados en texto
print(tokenizer_FT5.decode(outputs[0]))