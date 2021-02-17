import math

import torch.nn.functional as F
import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    """
    Classe permettant d'effectuer une self attention
    """
    def __init__(self, emb_dim=50, value_model=True, key_model=True, query_model=True, num_sample=False):
        """
        Desc

        (Args)
        
        """
        super().__init__()
        self.value_model = value_model
        self.key_model = key_model
        self.query_model = query_model
        self.is_sm_approx = False

        if(num_sample==True):
            self.sm = SMapprox(10, emb_dim)
            self.is_sm_approx = True
        elif(num_sample!=False):
            self.sm = SMapprox(num_sample, emb_dim)
            self.is_sm_approx = True
            
        if(value_model==True):
            self.value_model = nn.Linear(emb_dim, emb_dim, bias=True)
        elif(value_model==False):
            self.value_model = nn.Identity()

        if(key_model==True):
            self.key_model = nn.Linear(emb_dim, emb_dim, bias=True)
        elif(key_model==False):
            self.key_model = nn.Identity()

        if(query_model==True):
            self.query_model = nn.Linear(emb_dim, emb_dim, bias=True)
        elif(query_model==False):
            self.query_model = nn.Identity()


    def forward(self, x_mask):
        
        x, mask = x_mask
        
        #values representation
        values = self.value_model(x)
        
        #keys representation
        keys = self.key_model(x)
        
        #query representation
        queries = self.query_model(x)
        
        #Approximation of SoftMax with Favor+
        if(self.is_sm_approx):
            y = self.sm(queries, keys, values)
        #Normal softMax
        else :
            #compute probas
            attention_logits = torch.matmul(queries, torch.transpose(keys, 1, 2))/torch.sqrt(torch.tensor(x.shape[2], dtype=torch.float))
            
            #import ipdb; ipdb.set_trace()
            attention_logits_masked = attention_logits+mask
            
            probas = F.softmax(attention_logits_masked, dim=2)
            
            #compute sequence representation
            y = torch.matmul(probas, values)

        return y



class AttentionBlock(nn.Module):
    
    def __init__(self, emb_dim=50, value_model=False, key_model=False, query_model=False, mlp=False, \
        norm1=False, norm2=False, num_sample=False):
        
        super().__init__()
        self.mlp = mlp
        self.norm1 = norm1
        self.norm2 = norm2
        self.self_attention = SelfAttention(emb_dim, value_model, key_model, query_model, num_sample=num_sample)
            
        if(mlp==False):
            self.mlp = nn.Identity()
            
        if(norm1==True):
            self.norm1 = nn.LayerNorm(emb_dim) 
            
        if(norm2==True):
            self.norm2 = nn.LayerNorm(emb_dim) 
        elif(norm2==False):
            self.norm2 = nn.Identity()
        
    
    def forward(self, x_mask):
        
        x, mask = x_mask
        
        #compute sequence representation
        out1 =  self.self_attention(x_mask)
        
        if(self.norm1!=False):
            #sum input & output of attention
            out1 = x + out1 
            
            #Normalisation
            out1 = self.norm1(out1)
        
        #feed mlp
        out2 = self.mlp(out1)
        
        if(self.norm1!=False):
            out2 = out1+out2

        #Normalisation
        outputs = self.norm2(out2)
        
        return outputs, mask



class TransPerformer(nn.Module):
    
    def __init__(self, embeddings, word2id, emb_dim=50, nb_classe=2, L=3, \
                max_len=False, num_sample=False, norm1=False, norm2=True, \
                context_model=False, mlp=False, classifier=None, \
                value_model=False, key_model=False, query_model=False):
        
        super().__init__()
        self.word2id = word2id
        self.embeddings = torch.tensor(embeddings, dtype=torch.float)
        self.classifier = classifier
        self.context_model = context_model
        self.main = nn.Sequential(
                    *[AttentionBlock(emb_dim, num_sample=num_sample, norm1=norm1, norm2=norm2, \
                                     value_model=None if value_model is None else value_model[i], \
                                     key_model=None if key_model is None else key_model[i],\
                                     query_model=None if query_model is None else query_model[i],\
                                     mlp=None if mlp is None else mlp[i]) \
                      for i in range(L)]
                    )

        if(classifier is None):
            self.classifier = nn.Sequential(
                                nn.Linear(emb_dim, 10, bias=True),
                                nn.ReLU(),
                                nn.Linear(10, nb_classe, bias=True)
                                )
            
        if(context_model==False):
            self.context_model = nn.Identity()
            
        self.pe = PositionalEncoding(d_model=emb_dim, max_len=max_len) if max_len!=False else nn.Identity()

    
    def forward(self, x):
        
        #embedding
        emb_x = self.embeddings[x]
        
        #positional encoding
        emb_x = self.pe(emb_x) 
        
        #contextualising embedding
        context_emb = self.context_model(emb_x)
        
        #compute mask        
        mask = torch.where(x==self.word2id["__PAD__"], -float("Inf"), 0.)\
            .unsqueeze(2).repeat(1,1,emb_x.shape[1]).transpose(1,2)
        
        #compute self-attention layers
        outputs, _ = self.main((context_emb, mask))
        
        #average representation and classifier
        average = torch.mean(outputs, dim=1)
        
        y_hat = self.classifier(average)
        
        return y_hat



class SMapprox(nn.Module):
    def __init__(self,rd, hd, ort=True, pos=True):
        super().__init__()
        self.random_dim = rd
        self.hidden_dim = hd
        self.ortho = ort
        self.pos = pos
        self.vects = self.random_vect()
        
    def redrawn(self):
        self.vects = self.random_vect()

    # Vanila Transformer attention implementation
    def att(self, q, k, v, normalize=True):
        l, d = q.shape
        normalizer = 1 / (d ** 0.5) if normalize else 1
        a = torch.exp(q @ k.T * normalizer)
        d_inv = torch.diag(1 / (a @ torch.ones(l)))
        return d_inv @ a @ v


    # Perfomer attention implementation using some random feature map phi
    def att_hat(self,q, k, v, phi, normalize=True):

        if len(q.shape) == 2:
            l, d = q.shape
            normalizer = 1 / (d ** 0.25)
            q_prime = phi(q * normalizer)
            k_prime = phi(k * normalizer)

            d_inv = torch.diag(1 / (q_prime @ (k_prime.T @ torch.ones(l))))
            return d_inv @ (q_prime @ (k_prime.T @ v))


        # Batch mode
        elif len(q.shape) == 3:
            _, l, d = q.shape
            normalizer = 1 / (d ** 0.25)
        
            q_prime = phi(q * normalizer)
            k_prime = phi(k * normalizer)

            diag_coeff = 1 / torch.bmm(q_prime, k_prime.transpose(1, 2).sum(2).unsqueeze(2)).squeeze(2)
            d_inv = torch.diag_embed(diag_coeff)

            c1 = torch.bmm(k_prime.transpose(1,2),v)
            c2 = torch.bmm(q_prime,c1)
            out = torch.bmm(d_inv,c2)

            return out



    # random feature map
    def phi(self,h, fs):
        return lambda x: (
            h(x)
            / (self.random_dim**1/2)
            * torch.cat(
                [f(torch.einsum("...d,md->...m", x, self.vects)) for f in fs],
                axis=-1,
            )
        )


    # Performer "sin/cos" attention
    def sincos_att_hat(self,q, k, v, normalize=True):
        def h(x):
            return torch.exp(torch.square(x).sum(axis=-1, keepdims=True) / 2)

        sin = lambda x: torch.sin(2 * np.pi * x)
        cos = lambda x: torch.cos(2 * np.pi * x)

        kernel = self.phi(h, [sin, cos])
        return self.att_hat(q, k, v, kernel, normalize)


    # Performer "positive" attention
    def positive_att_hat(self,q, k, v, normalize=True):
        def h(x):
            return torch.exp(-torch.square(x).sum(axis=-1, keepdims=True) / 2)

        kernel = self.phi(h, [torch.exp])
        return self.att_hat(q, k, v, kernel, normalize)


    # generate IID Gaussian random features
    def iid_gaussian(self,m, d):
        return torch.randn(size=(m, d))


    # generate orthogonal Gaussian random features
    def orthogonal_gaussian(self,m, d):
        def orthogonal_square():
            # create orthogonal square matrix using Gram-Schmidt
            q, _ = torch.qr(self.iid_gaussian(d, d))
            return q.T

        num_squares = int(m / d)
        blocks = [orthogonal_square() for _ in range(num_squares)]

        remainder = m - d * num_squares
        if remainder:
            blocks.append(orthogonal_square()[:remainder])

        matrix = torch.cat(blocks)
        matrix /= (num_squares + remainder / d)**(1/2)

        return matrix

    def random_vect(self):
        if self.ortho:
            return self.orthogonal_gaussian(self.random_dim, self.hidden_dim)
        else :
            return self.iid_gaussian(self.random_dim, self.hidden_dim)


    # mean squared error
    def mse(self,a, b):
        return torch.square(a - b).mean()


    def forward(self,q,k,v,pos=True):
        
        if pos:
            return self.positive_att_hat(q,k,v,self.random_vect)
        else:
            return self.sincos_att_hat(q,k,v,self.random_vect)

class Contextualiser(nn.Module):
    def __init__(self, input_size=50, hidden_size=50, \
                    num_layers=1, batch_first=True, dropout=0., bidirectional=False):
        super(Contextualiser, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, \
                                        num_layers=1, batch_first=True, dropout=0., bidirectional=False)
    
    def forward(self, x):
        return self.lstm(x)[0]



class PositionalEncoding(nn.Module):
    """Position embeddings"""

    def __init__(self, d_model: int, max_len: int = 5000):
        """Génère des embeddings de position

        Args:
            d_model (int): Dimension des embeddings à générer
            max_len (int, optional): Longueur maximale des textes.
                Attention, plus cette valeur est haute, moins bons seront les embeddings de position.
        """
        super().__init__()

        pe = torch.zeros(max_len, d_model, dtype=torch.float)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        pe.requires_grad = False
        self.register_buffer('pe', pe)

    def forward(self, x):
        """Ajoute les embeddings de position"""
        x = x + self.pe[:, :x.size(1)]
        return x