from imports import * # Importing all necessary libraries (the actual imports are not shown here).

'''
The script loads a pretrained language model, tokenizes an input text prompt for translation, 
and generates a response using the model. It then decodes and prints the generated output.
'''
# -------------  Load model directly ------------------------ #

model_id = "ModelSpace/GemmaX2-28-2B-v0.1"  
tokenizer = AutoTokenizer.from_pretrained(model_id)  # Load the tokenizer for the specified model.
model = AutoModelForCausalLM.from_pretrained(model_id)  # Load the pretrained language model.

# The input text instructs the model to translate the given Chinese sentence into English.
text = "Translate this from Chinese to English:\nChinese: 我爱机器翻译\nEnglish:"  

# Tokenize the input text and return it as PyTorch tensors.
inputs = tokenizer(text, return_tensors="pt")  

# Generate text using the model with a maximum of 50 new tokens.
outputs = model.generate(**inputs, max_new_tokens=50)  

# Decode the generated tokens back into a human-readable string and print the output.
#print(tokenizer.decode(outputs[0], skip_special_tokens=True))  


# ------------- Use a pipeline as a high-level helper ------------------------ #

# Create a translation pipeline using the specified model
pipeline_translator = pipeline("translation", model="ModelSpace/GemmaX2-28-2B-v0.1")

# Input text in Chinese to be translated
text = "我爱机器翻译"

# Perform the translation
translated_text = pipeline_translator(text)

# Print the translated output
#print(translated_text)


# ------------- Use a pipeline as a high-level helper with another model ------------------------ #

pipeline_translator2 = pipeline("translation", model="Helsinki-NLP/opus-mt-zh-en")
translated_text2 = pipeline_translator2(text)
print(translated_text2[0]['translation_text'])






