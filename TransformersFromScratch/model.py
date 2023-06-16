"""
1.We are building the postional emb3edding part below.
2. Then we will build the Layer normalisation.
3. Then Multi-head attention
"""

import torch
import torch.nn as nn
import math
###3.4EmbeddingsandSoftmax(pg 5)
class InputEmbeddings(nn.module):

    def __init__(self,d_model : int, vocab_size: int):
        super().__init__()

        self.d_model=d_model
        self.vocab_size=vocab_size
        self.embeddings = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embeddings(x) * math.sqrt(self.d_model)
    
"""
This multiplication is performed to normalize the embeddings and prevent the values from becoming too large. 
Normalization helps in stabilizing the learning process and ensuring that the model can effectively process different inputs.
The model dimension (dmodel) refers to the dimensionality of the model's hidden layers. 
By multiplying the weights by sqrt(dmodel), the scaling effect ensures that the magnitude of the embeddings remains balanced 
and avoids potential issues that can arise from very large or very small values.
"""

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int,dropout: float) -> None:
        super().__init__()
        self.d_model=d_model
        self.seq_len=seq_len
        self.dropout=nn.Dropout(dropout)

        #Creating a matrix of shape (seq length ,d_model)
        pe = torch.zeros(seq_len,d_model)

        #crating a vector that will represent the position of the words inside the sentence
        #create a vector of shape (sequience length)
        position = torch.arange(0,seq_len,dtype=torch.float()).unsqueeze(1)
        div_term = torch.exp(torch.arange(0,d_model,2).float() * (-math.log(10000.0) / d_model))
        # apply the sin to even postion and cosine to odd positions
        pe[:,0::2]=torch.sin(position * div_term)
        pe[:,1::2]=torch.cos(position * div_term)

        pe =pe.unsqueeze(1) #this will become tnesor of 1,seq_len,d_model

        self.register_buffer('pe', pe) #### when we want to save atnesor along with them mdoel we use this method

    
    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1,:]]).requires_grad(False) ## we need to add positional encoding to every words inside the sentence require grad = flase make the model not learn the tensor
        return self.dropout(x)
    
"""
Layer Normalisation: It normalizes the inputs to each sub-layer within the encoder and decoder stacks.
The purpose is to ensure consistent input distribution and stabilize the training process.
It calculates mean and standard deviation along the feature dimension and centers and scales the inputs.
Layer normalization mitigates issues like vanishing/exploding gradients and helps capture long-range dependencies.
>Gamma and beta are two parameters introduced in layer normalization.They allow for fluctuations in the normalized data.
Gamma is a multiplicative parameter, and beta is an additive parameter.These parameters provide flexibility beyond the strict range of values between 0 and 1.
The network learns to tune gamma and beta during training.Gamma and beta enable the network to introduce scaling and shifting of the normalized values when necessary.
This adaptability enhances the network's ability to capture patterns and handle different input distributions. Epsilion keeps the number small and shorter for cpu/gpu to handle
"""

class LayerNormalisation(nn.Module):

    def __init__(self, eps: float = 10 **-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) #multiplied
        self.bias = nn.Parameter(torch.zeros(1)) #added
    
    def forward(self,x):
        mean = x.mean(dim = -1,keepdim =True)
        std = x.std(dim = -1, keepdim=True)
        return self.alpha * (x - mean) / (std +self.eps) + self.bias
    
class FeedForwardBlock(nn.Module):

    def __init__(self, d_model:int, d_ff:int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model,d_ff) #Matrix W1 and Bias B1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_model,d_ff) #Matrix W2 and Bias B2


    def forward(self, x):
        #input sentence is a tensor here with dimensions (Batch,Seq_len,d_model) -> convert to another tnesor (Batch,seq_len, d_ff) --> Linear conver (batch,seqlen,d_ff)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    
class MultiHeadAttentionBlock(nn.Module):

    def __init__(self,d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h 
        assert d_model % h == 0, "d_model is not divisble by h"
