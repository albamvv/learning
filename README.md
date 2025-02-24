# Base LLM vs Fine TUned LLM

## 📌 Project Overview  

### 1️⃣ Base_LLM_vs_Fine_Tuned_LLM  
In this case study, the student is asked to implement a pre-trained base model (T5 is recommended) and compare it with the same model after applying fine-tuning (Flan-T5 is recommended).
### 2️⃣ Instruction_Fine_tuning_LLM_Base
In this case study, the student is proposed to perform a fine-tuning instruction on the LLM Flan-T5-small in order to be able to summarize newspaper articles in Spanish.


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

### 1️⃣ Base_LLM_vs_Fine_Tuned_LLM  
In this case study, the student is asked to implement a pre-trained base model (T5 is recommended) and compare it with the same model after applying fine-tuning (Flan-T5 is recommended).

#### 1. Selección de un LLM base pre-entrenado

Tal y como hemos visto en secciones anteriores, existe una gran variedad de LLMs base que podemos utilizar: https://huggingface.co/models

En este caso práctico, vamos a hacer del modelo base T5 (https://huggingface.co/t5-base).

Este LLM esta compuesto por 220 millones de parámetros y ha sido pre-entrenado en número elevado de conjuntos de datos: https://huggingface.co/t5-base#training-details

#### 2. Selección de un Fine-tuned LLM

En este caso práctico, vamos a hacer del modelo base Flan-T5 (google/flan-t5-base).

Estos modelos se basan en T5 preentrenados (Raffel et al., 2020) y se les ha realizado fine-tuning para mejorar el rendimiento en más de 1.000 tareas adicionales y para soportar varios idiomas: https://huggingface.co/google/flan-t5-base#training-details

#### 3. Selection of a Fine-tuned LLM of 1 billion parameters

En este último apartado vamos a hacer uso de Flan-T5-Large que tiene un total de 1.200 millones de parámetros: https://huggingface.co/google/flan-t5-large

### 2️⃣ Instruction_Fine_tuning_LLM_Base

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
#### 4. Flan-T5 Fine-tuned text generation and evaluation
