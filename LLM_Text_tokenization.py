from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2") 

sentence=" hello darkness, my old friend"

input_ids=tokenizer(sentence, return_tensors='pt').input_ids
print('input_id: ',input_ids)
# input_id:  tensor([[23748, 11854,    11,   616,  1468,  1545]])

# words --> token --> Unique ID (cada token tiene su identificador)
#print(hasattr(tokenizer, "decode"))  # Should print True

for token_id in input_ids[0]:
    print(tokenizer.decode(token_id))

