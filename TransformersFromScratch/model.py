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
    """
    This class represents the input embeddings layer. It takes two arguments: d_model for the model dimension and vocab_size for the size of the vocabulary. 
    It initializes an embedding layer (self.embeddings) using nn.Embedding with the specified vocab_size and d_model.
    """

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

    """
    PositionalEncoding: This class represents the positional encoding layer. It takes three arguments: d_model for the model dimension, seq_len for the length of the input sequence, 
    and dropout for the dropout rate. 
    It generates positional encodings based on the sine and cosine functions. The positional encodings are added to the input tensor (x) and passed through a dropout layer.
    """

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

    """
    LayerNormalisation: This class implements layer normalization. It takes an optional argument eps for a small value to avoid division by zero. 
    It normalizes the input tensor (x) along the last dimension by subtracting the mean and dividing by the standard deviation. 
    It uses learnable parameters (self.alpha and self.bias) for scaling and shifting.
    """


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

    """
    FeedForwardBlock: This class represents the feed-forward block. It takes three arguments: d_model for the model dimension, d_ff for the feed-forward dimension, and dropout for the dropout rate. 
    It consists of two linear layers (self.linear_1 and self.linear_2) with ReLU activation in between. 
    The input tensor (x) is passed through the first linear layer, then through the ReLU activation function, dropout is applied, and finally, the output is passed through the second linear layer.
    """


    def __init__(self, d_model:int, d_ff:int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model,d_ff) #Matrix W1 and Bias B1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_model,d_ff) #Matrix W2 and Bias B2


    def forward(self, x):
        #input sentence is a tensor here with dimensions (Batch,Seq_len,d_model) -> convert to another tnesor (Batch,seq_len, d_ff) --> Linear conver (batch,seqlen,d_ff)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    
class MultiHeadAttentionBlock(nn.Module):

    """
    MultiHeadAttentionBlock: This class represents the multi-head attention block. It takes three arguments: d_model for the model dimension, h for the number of attention heads, and 
    dropout for the dropout rate. 
    It initializes linear layers (self.w_q, self.w_k, self.w_v, and self.w_o) for query, key, value, and output transformations. 
    It also defines the attention static method to perform attention calculation. The forward pass takes query, key, value tensors, and an optional mask. 
    It applies linear transformations to query, key, and value, splits them into multiple heads, applies attention calculation, and returns the output.
    
    """


    def __init__(self,d_model: int, h: int, dropout: float) -> None: #d_model =572, h= Number of heads = 2048 in paper
        super().__init__() 
        self.d_model = d_model
        self.h = h 
        assert d_model % h == 0, "d_model is not divisble by h"

        self.d_k = d_model//h #d_model is divided by h wem get a new number its called d_k
        self.w_q = nn.Linear(d_model,d_model) #wq
        self.w_k = nn.Linear(d_model,d_model) #wk
        self.w_v = nn.Linear(d_model,d_model) #wv

        self.w_o = nn.Linear(d_model,d_model) #w0
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query,key,value,mask,dropout: nn.Dropout):
        d_k = query.shape[-1]

        attention_scores = (query @ key.transpose(-2,-1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores = attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores= attention_scores.softmax(dim =-1) #(batch, h ,seqlen, seqlen)

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value), attention_scores



# We dont want the model to see some few words
    def forward(self, q, k, v, mask):
        query = self.w_q(q) #Batch,seqlen,d_model --->  changes to (batch,seqLen,)
        key = self.w_k(k) #Batch,seqlen,d_model --->  changes to (batch,seqLen,)
        value = self.w_v(v) #Batch,seqlen,d_model --->  changes to (batch,seqLen,)

        # (Batch,seqlen,d_model)--->(batch, seqlen,h,d_model)-->(batch,seqlen,d_k)

        query = query.view(query.shape[0],query.shape[1],self.h,self.d_k).transpose(1,2)
        key = key.view(key.shape[0],key.shape[1],self.h,self.d_k).transpose(1,2)
        value = value.view(value.shape[0],value.shape[1],self.h,self.d_k).transpose(1,2)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query,key,value,mask,self.dropout)
        # (batch,h,seqlen,d_k) ---> (batch,seqlen,h,d_k) --> (batch,seqlen,d_model)
        x = x.transpose(1,2).contiguous().view(x.shape[0],-1,self.h * self.d_k)

        # (batch,seqlen,d_model)-->(batch,seqlen,d_model)
        return self.w_o(x)
    
class ResidualConnetion(nn.Module):

    """
    ResidualConnetion: This class represents the residual connection with layer normalization. It takes one argument: dropout for the dropout rate. 
    The input tensor (x) is passed through the layer normalization, followed by the sublayer function (e.g., self-attention or feed-forward block), 
    and finally, the output is added to the input tensor with dropout applied.
    """

    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalisation()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x))) 
    
class EncoderBlock(nn.Module):

    """
    EncoderBlock: This class represents the encoder block. It takes three arguments: self_attention_block (an instance of MultiHeadAttentionBlock), feed_forward_block (an instance of FeedForwardBlock), 
    and dropout for the dropout rate. 
    It initializes a list of residual connections with layer normalization (self.residual_connections). 
    In the forward pass, the input tensor (x) is passed through the self-attention block, followed by the feed-forward block, and then through the residual connections.
    """

    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block:FeedForwardBlock, dropout:float) -> None:
        super().__init__(*args, **kwargs)
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnetion(dropout) for _ in range(2)])
        
    def forward(self,x, src_mask):

        x = self.residual_connections[0](x, lambda x : self.self_attention_block(x ,x, x , src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)

        return x
        
    
class Encoder(nn.Module):

    """
    Encoder: This class represents the encoder, which is a sequence of encoder blocks. 
    It takes one argument: layers, a list of encoder blocks. 
    In the forward pass, the input tensor (x) and a mask are passed through each encoder block, and the output is returned.
    
    """


    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x,mask)

        return self.norm(x)
    
class DecoderBlock(nn.Module):

    """
    DecoderBlock:
        This class represents a single block in the decoder of the Transformer model.
        It takes the following parameters:
            self_attention_block: An instance of the MultiHeadAttentionBlock class representing self-attention within the decoder.
            cross_attention_block: An instance of the MultiHeadAttentionBlock class representing cross-attention between the decoder and encoder.
            feed_forward_block: An instance of the FeedForwardBlock class representing the feed-forward neural network within the decoder.
            dropout: A float representing the dropout rate.
        The class initializes a list of residual connections with layer normalization (self.residual_connections).The forward method performs the forward pass through the decoder block. 
        It takes the following inputs:
            x: The input tensor to the decoder block.
            encoder_output: The output tensor from the encoder.
            src_mask: The mask for the source sequence.
            tgt_mask: The mask for the target sequence.
        Within the forward method, the input tensor x undergoes several operations:
        The input tensor is passed through the self-attention block using the self_attention_block instance, which performs self-attention on the input tensor.
        The resulting tensor is passed through the cross-attention block using the cross_attention_block instance, which performs cross-attention between the input tensor and the encoder output.
        The output tensor from cross-attention is passed through the feed-forward block using the feed_forward_block instance, which applies a feed-forward neural network to the tensor.
        The output tensor from the feed-forward block is returned as the result of the decoder block's forward pass.
    
    """

    def __init__(self,self_attention_block:MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnetion(dropout) for _ in range(3)])

    def forward(self,x, encoder_output, src_mask, tgt_mask):

        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x,x,x, tgt_mask))
        x = self.residual_connections[1](x ,lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, lambda x: self.feed_forward_block)
        return x

class Decoder(nn.Module):

    """
    Decoder:

        This class represents the decoder of the Transformer model.It takes the following parameter:
            layers: An nn.ModuleList containing the decoder blocks.
        The forward method performs the forward pass through the decoder. It takes the following inputs:
            x: The input tensor to the decoder.
            encoder_output: The output tensor from the encoder.
            src_mask: The mask for the source sequence.
            tgt_mask: The mask for the target sequence.
        The method iterates over the decoder blocks in self.layers and applies each block to the input tensor, encoder output, and masks.
        The output tensor from the last decoder block is returned as the result of the decoder's forward pass.

    """

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalisation()

    def forward(self,x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x,encoder_output,src_mask,tgt_mask)

        return self.norm(x)
    
class ProjectionLayer(nn.Module):

    """
    ProjectionLayer:

        This class represents the projection layer of the Transformer model.
        It takes the following parameters:
            d_model: An integer representing the dimensionality of the model.
            vocab_size: An integer representing the size of the vocabulary.
        The class initializes a linear projection layer (self.proj) to map the model's output to the vocabulary size.
        The forward method performs the forward pass through the projection layer. It takes the following input:
            x: The input tensor to the projection layer.
        The input tensor is passed through the linear projection layer, and the log softmax activation function is applied to obtain the output tensor.
    
    """


    def __init__(self, d_model:int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model,vocab_size)

    def forward(self,x):
        #(batch,seqlen,d_model)--->(batch,seqlen,vocabsize)
        return torch.log_softmax(self.proj(x),dim=1)
    
class Tansformer(nn.Module):

    """
    Transformer:

        This class represents the Transformer model.
        It takes the following parameters:
            encoder: An instance of the Encoder class representing the encoder.
            decoder: An instance of the Decoder class representing the decoder.
            src_embed: An instance of the InputEmbeddings class representing the source input embeddings.
            tgt_embed: An instance of the InputEmbeddings class representing the target input embeddings.
            src_pos: An instance of the PositionalEncoding class representing the positional encoding for the source sequence.
            tgt_pos: An instance of the PositionalEncoding class representing the positional encoding for the target sequence.
            projection_layer: An instance of the ProjectionLayer class representing the projection layer.
        The encode method performs the encoding step of the Transformer. It takes the following inputs:
            src: The source sequence tensor.
            src_mask: The mask for the source sequence.
        The method applies the source input embeddings, positional encoding, and encoder to the source sequence and returns the encoder output.
        The decode method performs the decoding step of the Transformer. It takes the following inputs:
            encoder_output: The output tensor from the encoder.
            src_mask: The mask for the source sequence.
            tgt: The target sequence tensor.
            tgt_mask: The mask for the target sequence.
        The method applies the target input embeddings, positional encoding, and decoder to the target sequence and returns the decoder output.
        The project method performs the projection step of the Transformer. It takes the following input:
            x: The input tensor to the projection layer.
        The method passes the input tensor through the projection layer and applies log softmax activation to obtain the projected output.
    
    """


    def __init__(self, encoder: Encoder, decoder:Decoder, src_embed: InputEmbeddings, tgt_embed:InputEmbeddings, src_pos: PositionalEncoding, tgt_pos:PositionalEncoding,projection_layer:ProjectionLayer) -> None:
        super().__init__()
        self.encoder= encoder
        self.decoder = decoder
        self.src_embed= src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):

        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src,src_mask)
    
    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt,encoder_output, src_mask,tgt_mask)
    
    def project(self,x):
        return self.projection_layer(x)



