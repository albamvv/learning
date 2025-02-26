# Differents proyects using Hugging Face

## 📌 Projects Overview  

### 1️⃣ LLM Traslate
The script loads a pretrained language model, tokenizes an input text prompt for translation, 
and generates a response using the model. It then decodes and prints the generated output.

https://huggingface.co/ModelSpace/GemmaX2-28-2B-v0.1

### 2️⃣ LLM Text tokenization
The script loads the GPT-2 model and tokenizer, tokenizes a given input sentence, and processes it through the model to obtain raw logits,
which represent the probabilities of possible next tokens. Instead of generating full text, it analyzes these logits to determine the most likely next token. 
It also extracts and displays the top ten predicted tokens along with their probabilities, 
making it useful for understanding how the model ranks different token predictions 
and how it assigns likelihoods to various continuations of a given text.

### 3️⃣ LLM Generate Text 
This Python script generates text using a GPT-2 language model. It first tokenizes an input sentence, converts it into tensor format, and then generates new text using the model’s generate() function. The generated text is then decoded back into a readable format. Additionally, the script uses the Hugging Face pipeline for text generation with the openai-community/gpt2 model.

### 4️⃣ Base_LLM_vs_Fine_Tuned_LLM   
The script compares three LLMs: a base model (t5-base), a fine-tuned model (google/flan-t5-base), and a larger fine-tuned model (google/flan-t5-large). It loads each model and its tokenizer, tokenizes a given prompt, and generates text. The base model produces general text, while the fine-tuned models specialize in tasks like question answering. The larger fine-tuned model has more parameters, improving performance. Outputs from each model are printed for comparison. The script highlights differences in text generation between pre-trained and fine-tuned models, showing the impact of fine-tuning and model size.
### 5️⃣ Instruction_Fine_tuning_LLM_Base
In this case study, the student is proposed to perform a fine-tuning instruction on the LLM Flan-T5-small in order to be able to summarize newspaper articles in Spanish.

### 6️⃣  xxxxxxx  
### 7️⃣   xxxxxxxx




## 🛠️ Installation  

### 1️⃣ Clone the repository  
```bash 
git clone https://github.com/albamvv/learning.git
```
```bash 
cd chatbot
```

### 2️⃣ Create and activate a virtual environment
```bash  
python -m venv env 
```
```bash
python3 -m venv env
```

```bash 
source env/bin/activate  # En Linux/macOS
```

```bash
env\Scripts\activate  # En Windows
```
### 3️⃣ Install dependencies 
```bash  
pip install -r requirements.txt 
```



## 📝 Structure

### 4️⃣ Base_LLM_vs_Fine_Tuned_LLM    
In this case study, the student is asked to implement a pre-trained base model (T5 is recommended) and compare it with the same model after applying fine-tuning (Flan-T5 is recommended).

#### 1. Base Pre-trained LLM (t5-base)

In this case study, we are going to use the base model T5 (https://huggingface.co/t5-base). This LLM is composed of 220 million parameters and has been pre-trained on a large number of datasets: https://huggingface.co/t5-base#training-details.

In this example we present the behavior of a base LLM, they have not been trained for any specific task, neither for conversation, nor for answering questions. They only present the primitive behavior of a LLM, which is to generate the next word, from the prompt that is indicated to it, it generates the next word, and continues generating words that in this case coincide in function of the context with what had been previously put.


---Loads a tokenizer and model from the t5-base checkpoint.
---Tokenizes a given prompt.
---Generates text without any fine-tuning (general-purpose text generation).
---Prints the generated output.

#### 2. Selección de un Fine-tuned LLM
In this case study, we will make the base model Flan-T5 (google/flan-t5-base).

These models are based on pre-trained T5 (Raffel et al., 2020) and have been fine-tuned to improve performance on over 1,000 additional tasks and to support multiple languages: https://huggingface.co/google/flan-t5-base#training-details

-Loads a tokenizer and model from the google/flan-t5-base checkpoint.
-Tokenizes the same prompt.
-Generates text based on fine-tuned knowledge (e.g., question answering).
-Prints the generated output.

#### 3. Selection of a Fine-tuned LLM of 1 billion parameters

In this last section we will make use of Flan-T5-Large, which has a total of 1.2 billion parameters: https://huggingface.co/google/flan-t5-large.

---Loads a tokenizer and model from the google/flan-t5-large checkpoint.
---Tokenizes the same prompt.
---Generates text using the larger fine-tuned model with more parameters.
---Prints the generated output.

### 5️⃣ Instruction_Fine_tuning_LLM_Base

#### 1. Behavior of Flan-T5-small without Fine-tuning

https://huggingface.co/google/flan-t5-small

#### 2. Selection and preparation of the data set

El conjunto de datos que vamos a utilizar para la realización del fine-tuning se corresponde con MLSUM.

MLSUM es el primer conjunto de datos de resumen multilingüe a gran escala. Obtenido de periódicos en línea, contiene más de 1.5 millones de pares de artículo/resumen en cinco idiomas diferentes: francés, alemán, español, ruso y turco.

Para más información: https://huggingface.co/datasets/mlsum

### 1. Reading the data set
### 2. Formatting the data set

Tal y como hemos comentado en secciones anteriores, el conjunto de datos utilizado para aplicar instruction fine-tuning debe estar formado por ejemplos de entrenamiento de la siguiente forma:

```bash 
(prompt, completion)
```
Para formar el prompt debemos tener en cuenta los siguientes puntos:

Debe indicarse una instrucción para que realice el modelo. Es habitual utilizar plantillas que proponen los desarrolladores de los LLM para diseñar nuestros ejemplos de entrenamiento: https://github.com/google-research/FLAN/blob/main/flan/v2/flan_templates_branched.py
La plantilla que vamos a seleccionar es la siguiente:

```bash 
("Resume el siguiente articulo:\n\n{text}", "{summary}")
```
Para afinar modelos como FLAN-T5 en tareas conversacionales donde queramos preservar el contexto de la conversación, se adopta un enfoque basado en secuencias, donde la interacción entre los participantes de la conversación se estructura en una sola cadena. La pregunta y la respuesta suelen estar separadas por un token especial, como , , o simplemente utilizando un delimitador claro (ej: \n)
```bash 
"Conversación:\n[Usuario] ¿Cuál es la capital de Francia?\n[Asistente] La capital de Francia es París.\n[Usuario] ¿Y cuál es su río principal?\n
```
### 3. Tokenization of the data set

Una de las cosas que comentabamos cuándo comenzamos a hablar de modelos generativos como la Redes Neuronales Recurrentes, es que este tipo de algoritmos, al igual que los LLMs, reciben secuencias del mismo tamaño.

Con lo cual, al igual que hicimos en ese caso práctico al comienzo del curso, debemos obtener la secuencia más larga de nuestro conjunto de datos y realizar padding al resto de secuencias para que todas tengan el mismo tamaño.

#### 3. Model Fine tuning

1. Lectura del modelo
2. Evaluación durante el entrenamiento
3. Lectura y adaptación de los datos para el entrenamiento
4. Preparación y ejecución del fine-tuning (entrenamiento)

#### 4. Flan-T5 Fine-tuned text generation and evaluation
