from transformers import T5Tokenizer, T5ForConditionalGeneration
from utils import device,text1, text2,text3
from datasets import load_dataset


'''1. Behavior of Flan-T5-small without Fine-tuning''' 

### Lectura del modelo y tokenizador
# Importamos el tokenizador
tokenizer_FT5 = T5Tokenizer.from_pretrained("google/flan-t5-small")
# Importamos el modelo pre-entrenado
model_FT5 = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small", device_map="auto")

#Generación de texto
prompt_template = f"Resume el siguiente articulo:\n\n{text3}"
#prompt_template = "translate English to German: How old are you?"

# Tokenizamos el prompt
prompt_tokens = tokenizer_FT5(prompt_template, return_tensors="pt").input_ids.to(device)

# Generamos los siguientes tokens
outputs = model_FT5.generate(prompt_tokens, max_length=200)

# Transformamos los tokens generados en texto
#print(tokenizer_FT5.decode(outputs[0]))

''' 2. Selection and preparation of the data set'''

### 1. Reading the data set
ds = load_dataset("mlsum", 'es',trust_remote_code=True)
# Mostramos un ejemplo del subconjunto de datos de entrenamiento
#print(ds["train"]["text"][10]) #Muestra el décimo artículo de la sección de entrenamiento.
# Mostramos el resumen correspondiente al ejemplo anterior
#print(ds["train"]["summary"][10]) # Muestra el resumen correspondiente al décimo artículo.

### 2. Formatting the data set

# Reducimos el conjunto de datos
NUM_EJ_TRAIN = 1500
NUM_EJ_VAL = 500
NUM_EJ_TEST = 200

# Subconjunto de entrenamiento
ds['train'] = ds['train'].select(range(NUM_EJ_TRAIN))

# Subconjunto de validación
ds['validation'] = ds['validation'].select(range(NUM_EJ_VAL))

# Subconjunto de pruebas
ds['test'] = ds['test'].select(range(NUM_EJ_TEST))

### 3. Tokenization of the data set
''' 3. Model Fine tuning'''
''' 4. Flan-T5 Fine-tuned text generation and evaluation'''

