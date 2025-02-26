# next_word_predictor.py - Next Word Prediction Module

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import os

class NextWordPredictor:
    def __init__(self, model_name="gpt2", device=None, cache_dir=None):
        """
        Initialize the next word prediction model.
        
        Args:
            model_name (str): The pre-trained model to use (default: 'gpt2')
            device (str): Device to run model on ('cpu' or 'cuda'). If None, uses CUDA if available.
            cache_dir (str): Directory to cache downloaded models
        """
        print(f"Loading model: {model_name}")
        start_time = time.time()
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode
        
        print(f"Model loaded in {time.time() - start_time:.2f} seconds")
    
    def predict_next_words(self, text, num_words=1, num_predictions=5, temperature=1.0):
        """
        Predict the next word(s) given an input text.
        
        Args:
            text (str): The input text to complete
            num_words (int): Number of words to predict
            num_predictions (int): Number of alternative predictions to return
            temperature (float): Sampling temperature (higher = more random)
            
        Returns:
            list: List of tuples (predicted_text, probability)
        """
        # Tokenize input text
        input_ids = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
        
        # For predicting multiple words at once
        if num_words > 1:
            return self._predict_multiple_words(input_ids, num_words, num_predictions, temperature)
        
        # For single word prediction
        return self._predict_single_word(input_ids, num_predictions, temperature)
        
    def _predict_single_word(self, input_ids, num_predictions, temperature):
        """
        Helper method to predict a single next word with proper tokenization.
        """
        with torch.no_grad():
            # Get model predictions
            outputs = self.model(input_ids)
            logits = outputs.logits[:, -1, :]
            
            # Apply temperature
            scaled_logits = logits / temperature
            
            # Get probabilities
            probs = torch.nn.functional.softmax(scaled_logits, dim=-1)
            
            # Get top-k token indices and probabilities
            topk_probs, topk_indices = torch.topk(probs, num_predictions * 5, dim=-1)  # Get more candidates than needed
        
        # Process predictions
        predictions = []
        seen_words = set()  # To avoid duplicate words
        
        for i in range(topk_indices.shape[1]):
            if len(predictions) >= num_predictions:
                break
                
            # Get token and decode
            token_id = topk_indices[0][i].item()
            token_text = self.tokenizer.decode([token_id]).strip()
            
            # Skip special tokens and empty strings
            if (token_text.startswith('<') and token_text.endswith('>')) or not token_text:
                continue
                
            # Skip if it's just whitespace
            if token_text.isspace() or not token_text.strip():
                continue
                
            # Skip duplicates at the word level
            word = token_text.strip().lower()
            if word in seen_words:
                continue
                
            seen_words.add(word)
            predictions.append((token_text, topk_probs[0][i].item()))
            
        return predictions
        
    def _predict_multiple_words(self, input_ids, num_words, num_predictions, temperature):
        """
        Helper method to predict multiple words with greedy decoding for each candidate.
        """
        # First, get the next word candidates
        with torch.no_grad():
            outputs = self.model(input_ids)
            next_token_logits = outputs.logits[:, -1, :] / temperature
            next_token_probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            topk_probs, topk_indices = torch.topk(next_token_probs, num_predictions * 3, dim=-1)
        
        # Filter first tokens
        filtered_indices = []
        filtered_probs = []
        seen_words = set()
        
        for i in range(topk_indices.shape[1]):
            if len(filtered_indices) >= num_predictions:
                break
                
            token_id = topk_indices[0][i].item()
            token_text = self.tokenizer.decode([token_id]).strip()
            
            # Skip special tokens and duplicates
            if ((token_text.startswith('<') and token_text.endswith('>')) or 
                not token_text or token_text.isspace() or 
                token_text.lower() in seen_words):
                continue
                
            seen_words.add(token_text.lower())
            filtered_indices.append(token_id)
            filtered_probs.append(topk_probs[0][i].item())
            
        # For each candidate first word, generate the full sequence
        predictions = []
        
        for idx, (first_token_id, first_prob) in enumerate(zip(filtered_indices, filtered_probs)):
            if idx >= num_predictions:
                break
                
            # Start with the input plus the first predicted token
            current_ids = torch.cat([
                input_ids, 
                torch.tensor([[first_token_id]]).to(self.device)
            ], dim=1)
            
            # Initial token
            generated_text = self.tokenizer.decode([first_token_id])
            cumulative_prob = first_prob
            
            # Generate remaining tokens one by one
            for _ in range(num_words - 1):
                with torch.no_grad():
                    outputs = self.model(current_ids)
                    next_logits = outputs.logits[:, -1, :] / temperature
                    next_probs = torch.nn.functional.softmax(next_logits, dim=-1)
                    
                    # Get the highest probability token
                    prob, next_id = torch.max(next_probs, dim=-1)
                    prob = prob.item()
                    next_id = next_id.item()
                    
                    # Get the token text
                    next_text = self.tokenizer.decode([next_id])
                    
                    # Update tracking variables
                    current_ids = torch.cat([
                        current_ids, 
                        torch.tensor([[next_id]]).to(self.device)
                    ], dim=1)
                    generated_text += next_text
                    cumulative_prob *= prob
            
            # Clean up predicted text
            clean_text = self._clean_prediction(generated_text)
            predictions.append((clean_text, cumulative_prob))
        
        return predictions
    
    def _clean_prediction(self, text):
        """Clean up predicted text by removing extra spaces and special tokens."""
        # Replace multiple spaces with a single space
        text = ' '.join(text.split())
        # Remove special tokens if any slipped through
        for special in ['<s>', '</s>', '<pad>', '<|endoftext|>']:
            text = text.replace(special, '')
        return text.strip()