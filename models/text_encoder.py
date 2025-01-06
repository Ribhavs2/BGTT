import torch
import torch.nn as nn
from transformers import BertModel

class TextUnicoder(nn.Module):
    def __init__(self, bert_model_name="bert-base-uncased", embedding_dim=128, tokenizer=None):
        """
        Initialize the Text Unicoder.
        Args:
            bert_model_name (str): Pre-trained BERT model to use.
            embedding_dim (int): Desired dimensionality of the output embeddings.
            tokenizer: Tokenizer with added special tokens (optional).
        """
        super(TextUnicoder, self).__init__()
        
        # Load the pre-trained BERT model
        self.bert = BertModel.from_pretrained(bert_model_name)

        # Resize embeddings if custom tokens are added
        if tokenizer is not None:
            self.bert.resize_token_embeddings(len(tokenizer))

        # Dimensionality of BERT's output embeddings
        self.bert_output_dim = self.bert.config.hidden_size  # Typically 768 for BERT base

        # Linear projection layer (optional)
        self.projection_layer = nn.Linear(self.bert_output_dim, embedding_dim)
        self.use_projection = embedding_dim != self.bert_output_dim  # Apply projection only if dimensions differ

    def forward(self, text_input_ids, text_attention_mask):
        """
        Forward pass for Text Unicoder.
        Args:
            text_input_ids (torch.Tensor): Tokenized text input IDs [batch_size, max_seq_length].
            text_attention_mask (torch.Tensor): Attention masks for the text inputs [batch_size, max_seq_length].
        Returns:
            torch.Tensor: Text embeddings [batch_size, embedding_dim].
        """
        # Pass inputs through BERT to get token-level embeddings
        bert_outputs = self.bert(
            input_ids=text_input_ids,
            attention_mask=text_attention_mask
        )

        # Extract [CLS] token embeddings
        cls_embeddings = bert_outputs.last_hidden_state[:, 0, :]  # [batch_size, bert_output_dim]

        # Optional linear projection
        if self.use_projection:
            cls_embeddings = self.projection_layer(cls_embeddings)  # [batch_size, embedding_dim]

        return cls_embeddings
