from transformers import T5Tokenizer, T5ForConditionalGeneration
from utils import device,text1, text2,text3



'''1. Behavior of Flan-T5-small without Fine-tuning''' 

### Lectura del modelo y tokenizador
# Importamos el tokenizador
tokenizer_FT5 = T5Tokenizer.from_pretrained("google/flan-t5-small")
# Importamos el modelo pre-entrenado
model_FT5 = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small", device_map="auto")

#Generación de texto
prompt_template = f"Resume el siguiente articulo:\n\n{text1}"
prompt_template = "translate English to German: How old are you?"

# Tokenizamos el prompt
prompt_tokens = tokenizer_FT5(prompt_template, return_tensors="pt").input_ids.to(device)

# Generamos los siguientes tokens
outputs = model_FT5.generate(prompt_tokens, max_length=200)

# Transformamos los tokens generados en texto
print(tokenizer_FT5.decode(outputs[0]))

''' 2. Selection and preparation of the data set'''
''' 3. Model Fine tuning'''
''' 4. Flan-T5 Fine-tuned text generation and evaluation'''

