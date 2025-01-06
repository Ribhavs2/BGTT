import torch
import torch.nn as nn
from transformers import BertModel

class TripleListEncoder(nn.Module):
    def __init__(self, bert_model_name="bert-base-uncased", embedding_dim=128, tokenizer=None):
        """
        Initialize the Triple List Encoder.
        Args:
            bert_model_name (str): Pre-trained BERT model to use.
            embedding_dim (int): Desired dimensionality of the output embeddings.
        """
        super(TripleListEncoder, self).__init__()
        # Load pre-trained BERT model
        self.bert = BertModel.from_pretrained(bert_model_name)

        if tokenizer is not None:
            self.bert.resize_token_embeddings(len(tokenizer))

        # Dimensionality of BERT's output embeddings
        self.bert_output_dim = self.bert.config.hidden_size  # Typically 768 for BERT base            
        
        # Linear projection layer (optional)
        self.projection_layer = nn.Linear(self.bert_output_dim, embedding_dim)
        self.use_projection = embedding_dim != self.bert_output_dim  # Use projection only if dimensions differ

    def forward(self, triple_input_ids, triple_attention_mask):
        """
        Forward pass to encode triples into embeddings.
        Args:
            triple_input_ids (torch.Tensor): Tokenized triple input IDs [batch_size, max_seq_length].
            triple_attention_mask (torch.Tensor): Attention masks for the triple inputs [batch_size, max_seq_length].
        Returns:
            torch.Tensor: Triple-level embeddings [batch_size, embedding_dim].
        """
        # Pass through BERT to get token-level embeddings
        bert_outputs = self.bert(
            input_ids=triple_input_ids,
            attention_mask=triple_attention_mask
        )
        
        # Extract [CLS] token embeddings (first token in each sequence)
        cls_embeddings = bert_outputs.last_hidden_state[:, 0, :]  # [batch_size, bert_output_dim]
        
        # Optional linear projection
        if self.use_projection:
            cls_embeddings = self.projection_layer(cls_embeddings)  # [batch_size, embedding_dim]
        
        return cls_embeddings
