# Base LLM vs Fine TUned LLM

## 📌 Project Overview  
In this case study, the student is asked to implement a pre-trained base model (T5 is recommended) and compare it with the same model after applying fine-tuning (Flan-T5 is recommended).



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

### 1️⃣ Selección de un LLM base pre-entrenado

Tal y como hemos visto en secciones anteriores, existe una gran variedad de LLMs base que podemos utilizar: https://huggingface.co/models

En este caso práctico, vamos a hacer del modelo base T5 (https://huggingface.co/t5-base).

Este LLM esta compuesto por 220 millones de parámetros y ha sido pre-entrenado en número elevado de conjuntos de datos: https://huggingface.co/t5-base#training-details

### 2️⃣ Selección de un Fine-tuned LLM

En este caso práctico, vamos a hacer del modelo base Flan-T5 (google/flan-t5-base).

Estos modelos se basan en T5 preentrenados (Raffel et al., 2020) y se les ha realizado fine-tuning para mejorar el rendimiento en más de 1.000 tareas adicionales y para soportar varios idiomas: https://huggingface.co/google/flan-t5-base#training-details

### 3️⃣ Selection of a Fine-tuned LLM of 1 billion parameters

En este último apartado vamos a hacer uso de Flan-T5-Large que tiene un total de 1.200 millones de parámetros: https://huggingface.co/google/flan-t5-large