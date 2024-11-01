from transformers import RobertaModel, RobertaTokenizer
import torch

def roberta_encode(input_text):
    embedding = []
    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    # Load pre-trained model (weights)
    model = RobertaModel.from_pretrained('roberta-base')
    # Predict hidden states features for each layer
    
    for text in input_text:
        with torch.no_grad():  # No gradient is needed (inference mode)
            encoded_input = tokenizer(text, return_tensors='pt')  # "pt" for PyTorch tensors
                # Forward pass, get hidden states output
            outputs = model(**encoded_input)
            # The last_hidden_state is the last layer hidden states, which are the contextualized word embeddings
            last_hidden_state = outputs.last_hidden_state
    
            # To get the embeddings for the `[CLS]` token, which can be used as a sentence representation
            cls_embedding = last_hidden_state[:, 0, :]
    
            # Now `cls_embedding` holds the sentence-level representation for the input text
            
            embedding.append(cls_embedding)
        
    embedding = torch.stack(embedding)
    
    assert embedding.shape == (len(input_text), 1, 768)
    
    return embedding