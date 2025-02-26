import langraph.graph as lg
from transformers import pipeline

# Cargar modelo gratuito de Hugging Face (Mistral 7B Instruct)
modelo_llm = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct", device="cpu")

# Crear un gráfico Langraph
graph = lg.Graph()

# Nodo de inicio
@graph.node()
def inicio():
    return "Hola, ¿cómo puedo ayudarte hoy?"

# Nodo de respuesta
@graph.node()
def responder_pregunta(user_input: str):
    respuesta = modelo_llm(user_input, max_length=100, do_sample=True)[0]["generated_text"]
    return respuesta

# Conectar nodos
@graph.edge(inicio, responder_pregunta)
def conectar_nodos():
    user_input = input("Tu pregunta: ")
    return {"user_input": user_input}

# Ejecutar flujo
graph.set_entry_point(inicio)
graph.run()
