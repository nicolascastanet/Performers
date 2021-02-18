"""
Module regroupant les classes liées au modèle Transformer et Performer
"""
import math

import torch.nn.functional as F
import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    """
    Permet d'effectuer une self attention
    """
    def __init__(self, emb_dim=50, value_model=True, key_model=True, query_model=True, num_sample=False):
        """
        
        Args
          emb_dim (int): Dimension d'embeddings des mots
          value_model (list[nn.Module]): List de longueur L (nombre de layer) contenant à l'indice i le model permettant d'obtenir les values à la layer i
          key_model (list[nn.Module]): List de longueur L (nombre de layer) contenant à l'indice i le model permettant d'obtenir les keys à la layer i
          query_model (list[nn.Module]): List de longueur L (nombre de layer) contenant à l'indice i le model permettant d'obtenir les queries à la layer i
          num_sample (int/False): Nombre de tirage de vecteur (w) que l'on doit faire pour l'approximation du softmax. False permet de désactiver le mécanisme.
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
            
            attention_logits_masked = attention_logits+mask
            
            probas = F.softmax(attention_logits_masked, dim=2)
            
            #compute sequence representation
            y = torch.matmul(probas, values)

        return y



class AttentionBlock(nn.Module):
    """
    Applique une self-attention avec les normalisations et les changements de représentations qui vont (correspond à un layer).
    """
    def __init__(self, emb_dim=50, value_model=False, key_model=False, query_model=False, mlp=False, \
        norm1=False, norm2=False, num_sample=False):
        """
        Args
          emb_dim (int): Dimension d'embeddings des mots
          value_model (list[nn.Module]): List de longueur L (nombre de layer) contenant à l'indice i le model permettant d'obtenir les values à la layer i
          key_model (list[nn.Module]): List de longueur L (nombre de layer) contenant à l'indice i le model permettant d'obtenir les keys à la layer i
          query_model (list[nn.Module]): List de longueur L (nombre de layer) contenant à l'indice i le model permettant d'obtenir les queries à la layer i
          mlp (list[nn.Module]): List de longueur L (nombre de layer) contenant à l'indice i le model changeant la représentation à la layer i
          norm1 (bool): Permet d'activer ou non la normalisation suivant la sel-attention 
          norm2 (bool): Permet d'activer ou non la normalisation avant la couche suivante
          num_sample (int/False): Nombre de tirage de vecteur (w) que l'on doit faire pour l'approximation du softmax. False permet de désactiver le mécanisme.
        """
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
    """
    Applique une succession de layer d'attention avec un modèle final de classification ou de regression.
    """
    def __init__(self, embeddings, word2id, emb_dim=50, nb_classe=2, L=3, \
                max_len=False, num_sample=False, norm1=False, norm2=True, \
                context_model=False, mlp=False, classifier=None, \
                value_model=False, key_model=False, query_model=False):
        """
        Args
          embeddings (Tensor/Matrix): Embeddings des mots voulues sous forme de tensor
          word2id (dict): Dictionnaire des mots du vocabulaire choisi
          emb_dim (int): Dimension d'embeddings des mots
          nb_classe (int): nombre de classe du dataset considéré (utilisé pour construire un classifieur par défaut lorsque non fourni)
          L (int): nombre de couches d'attention
          max_len (int/False): Longueur maximal utile pour le positionnal encodding. False désactive cette option.
          num_sample (int/False): Nombre de tirage de vecteur (w) que l'on doit faire pour l'approximation du softmax. False permet de désactiver le mécanisme.
          norm1 (bool): Permet d'activer ou non la normalisation suivant la self-attention 
          norm2 (bool): Permet d'activer ou non la normalisation avant la couche suivante
          context_model (nn.Module): Modèle utiliser avant les couches d'attentions pour modifier l'embeddings utilisé
          mlp (list[nn.Module]): List de longueur L (nombre de layer) contenant à l'indice i le model changeant la représentation à la layer i
          classifier (nn.Module): Modèle suivant les couches d'attentions permettant la classification (ou la régression)
          value_model (list[nn.Module]): List de longueur L (nombre de layer) contenant à l'indice i le model permettant d'obtenir les values à la layer i
          key_model (list[nn.Module]): List de longueur L (nombre de layer) contenant à l'indice i le model permettant d'obtenir les keys à la layer i
          query_model (list[nn.Module]): List de longueur L (nombre de layer) contenant à l'indice i le model permettant d'obtenir les queries à la layer i
        """
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
        mask = torch.zeros_like(x)
        mask = mask.float().masked_fill(x==self.word2id["__PAD__"], float('-inf'))\
            [:,None,:].expand(x.shape[0],x.shape[1],x.shape[1])
        
        #compute self-attention layers
        outputs, _ = self.main((context_emb, mask))
        
        #average representation and classifier
        average = torch.mean(outputs, dim=1)
        
        y_hat = self.classifier(average)
        
        return y_hat


#TODO :
#1) vérifier que toutes les fonctions et arguments sont utiles :
#  - argument "normalize" de "att_hat" qui ne semble pas être utilisé
#2) Vérifier DocString des méthodes (dont types et descriptions d'arguments)

class SMapprox(nn.Module):
    def __init__(self, rd, hd, ort=True, pos=True):
        """
        Args
          rd (int): Nombre de vecteur (w) à tirer pour l'estimation du softmax 
          hd (int): Dimension de l'embedding
          ort (bool): Permet de choisir si les vecteurs aléatoires sont orthogonaux ou non
          pos (bool): Permet de choisir si on active l'option positive ou non
        """
        super().__init__()
        self.random_dim = rd
        self.hidden_dim = hd
        self.ortho = ort
        self.pos = pos
        self.vects = self.random_vect()
        
    def redrawn(self):
        """Permet de retirer les vecteurs aléatoires"""
        self.vects = self.random_vect()


   
    def att_hat(self,q, k, v, phi, normalize=True):
        """
        Implémentation du Performer utilisant Favor+ (positive random feature)

        Args:
          q (Tensor): Tenseur contenant les queries
          k (Tensor): Tenseur contenant les keys
          v (Tensor): Tenseur contenant les values
          phi (callable): Fonction ???
          normalize (bool): ???

        Return:
          La matrice Y résultant de Q'.K'_transpose.V où Q'.K'_transpose = SoftMax(Q.K_t)
        """
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



    def phi(self,h, fs):
        """
        Calcul des features maps aléatoires

        Args:
          h (callable): Fonction ??? (reprendre terme du papier peut-être?)
          fs (list[callable]): Liste de fonction ??? (reprendre terme du papier peut-être?)

        Return:
          La fonction ??? (reprendre terme du papier peut-être?)
        """
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
        """
        Implémentation du Performer utilisant "sin/cos" attention

        Args:
          q (Tensor): Tenseur contenant les queries
          k (Tensor): Tenseur contenant les keys
          v (Tensor): Tenseur contenant les values
          normalize (bool): ???

        Return:
          La matrice Y résultant de Q'.K'_transpose.V où Q'.K'_transpose = SoftMax(Q.K_t)
        """
        def h(x):
            return torch.exp(torch.square(x).sum(axis=-1, keepdims=True) / 2)

        sin = lambda x: torch.sin(2 * np.pi * x)
        cos = lambda x: torch.cos(2 * np.pi * x)

        kernel = self.phi(h, [sin, cos])
        return self.att_hat(q, k, v, kernel, normalize)


    # Performer "positive" attention
    def positive_att_hat(self,q, k, v, normalize=True):
        """
        Implémentation du Performer utilisant la "positive" attention

        Args:
          q (Tensor): Tenseur contenant les queries
          k (Tensor): Tenseur contenant les keys
          v (Tensor): Tenseur contenant les values
          normalize (bool): ???

        Return:
          La matrice Y résultant de Q'.K'_transpose.V où Q'.K'_transpose = SoftMax(Q.K_t)
        """
        def h(x):
            return torch.exp(-torch.square(x).sum(axis=-1, keepdims=True) / 2)

        kernel = self.phi(h, [torch.exp])
        return self.att_hat(q, k, v, kernel, normalize)


    def iid_gaussian(self,m, d):
        """
        Génère des features Gaussiennes aléatoire IID

        Args:
          m (int): Dimension ???
          d (int): Dimension ???

        Return:
          Retourne les features gaussiennes aléatoires de dimension m*d
        """
        return torch.randn(size=(m, d))


    def orthogonal_gaussian(self,m, d):
        """
        Génère des features othogonales Gaussiennes aléatoires

        Args:
          m (int): Dimension ???
          d (int): Dimension ???

        Return:
          Retourne les features orthogonales gaussiennes aléatoires de dimension m*d
        """
        def orthogonal_square():
            """
            Create orthogonal square matrix using Gram-Schmidt

            Return:
              ???
            """
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
        """
        Génère des features aléatoires selon les paramètres initiaux

        Return:
          Retourne les features aléatoires
        """
        if self.ortho:
            return self.orthogonal_gaussian(self.random_dim, self.hidden_dim)
        else :
            return self.iid_gaussian(self.random_dim, self.hidden_dim)


    def forward(self,q,k,v,pos=True):
        
        if pos:
            return self.positive_att_hat(q,k,v,self.random_vect)
        else:
            return self.sincos_att_hat(q,k,v,self.random_vect)

class Contextualiser(nn.Module):
    """
    Permet d'effectuer un plongement contextualisé d'une séquence
    """
    def __init__(self, input_size=50, hidden_size=50, \
                    num_layers=1, batch_first=True, dropout=0., bidirectional=False):
        """
        Args:
          input_size (int): Dimension d'entrée
          hidden_size (int): Dimension désirée de sortie
          num_layers (int): Nombre de layer de LSTM empilé
          batch_first (bool): Indique si la dimension de batch est en première position (True) ou non (False)
          dropout (float): Probabilité d'appliquer un dropout (seulement entre les couches intermédiaires)
          bidirectional (bool): Permet de rendre les LSTMs bidirectionnaux (True) ou non (False)

        Return:
          Retourne les features orthogonales gaussiennes aléatoires de dimension m*d
        """
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