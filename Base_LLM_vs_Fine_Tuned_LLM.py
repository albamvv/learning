from transformers import T5Tokenizer, T5ForConditionalGeneration

'''
T5Tokenizer: Para convertir texto en tokens que el modelo puede entender.
Se descarga y carga el tokenizador del modelo "t5-base" desde Hugging Face.
Permite tokenizar (convertir texto en números) y detokenizar (convertir números en texto).
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