import re
import logging
import os.path
from pathlib import Path
import math

import numpy as np
from tqdm.auto import tqdm
from datamaestro import prepare_dataset
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.distributions import Categorical

from earlyStopping import EarlyStopping
from SMkernel import SMapprox

##############################
########### Models ###########
##############################

class SelfAttention(nn.Module):
    def __init__(self, emb_dim=50, values_model=None, keys_model=None, query_model=None, num_sample=10):
        super(SelfAttention, self).__init__()
        self.values_model = values_model
        self.keys_model = keys_model
        self.query_model = query_model
        self.is_sm_approx = False

        if(num_sample>0):
            self.sm = SMapprox(num_sample, emb_dim)
            self.is_sm_approx = True
            
        if(values_model is None):
            self.values_model = nn.Linear(emb_dim, emb_dim, bias=True)
            
        if(keys_model is None):
            self.keys_model = nn.Linear(emb_dim, emb_dim, bias=True)
            
        if(query_model is None):
            self.query_model = nn.Linear(emb_dim, emb_dim, bias=True)

    def forward(self, x):
        
        context_emb, mask = x
        
        #values representation
        values = self.values_model(context_emb)
        
        #keys representation
        keys = self.keys_model(context_emb)
        
        #query representation
        queries = self.query_model(context_emb)
        
        #Approximation of SoftMax with Favor+
        if(self.is_sm_approx):
            y = self.sm(queries, keys, values)
        #Normal softMax
        else :
            #compute probas
            attention_logits = torch.matmul(queries, torch.transpose(keys, 1, 2))/torch.sqrt(torch.tensor(context_emb.shape[2], dtype=torch.float))
            
            #import ipdb; ipdb.set_trace()
            attention_logits_masked = attention_logits+mask
            
            probas = F.softmax(attention_logits_masked, dim=2)
            
            #compute sequence representation
            y = torch.matmul(probas, values)

        return y


class AttentionBlockTP10(nn.Module):
    
    def __init__(self, emb_dim=50, values_model=None, keys_model=None, query_model=None, mlp=None, norm=None, num_sample=0):
        
        super(AttentionBlockTP10, self).__init__()
        self.mlp = mlp
        self.self_attention = SelfAttention(emb_dim, values_model, keys_model, query_model, num_sample=num_sample)
            
        if(mlp is None):
            self.mlp = nn.Linear(emb_dim, emb_dim, bias=True)
            
        self.norm = norm
        if(norm is None):
            self.norm = nn.LayerNorm(emb_dim) 
        elif(norm==False):
            self.norm = nn.Identity()
            

    
    def forward(self, x):
        
        context_emb, mask = x
        
        #compute sequence representation
        y =  self.self_attention(x)
        
        #sum input & output
        tmp_outputs = context_emb + y 
        
        #feed mlp
        outputs = self.mlp(tmp_outputs)
        
        #Normalisation
        outputs = self.norm(outputs)
        
        return outputs, mask


class AttentionBlock(nn.Module):
    
    def __init__(self, emb_dim=50, values_model=None, keys_model=None, query_model=None, mlp=None, norm1=None, norm2=None):
        
        super(AttentionBlock, self).__init__()
        self.mlp = mlp
        self.self_attention = SelfAttention(emb_dim, values_model, keys_model, query_model)
            
        if(mlp is None):
            self.mlp = nn.Linear(emb_dim, emb_dim, bias=True)
            
        self.norm1 = norm1
        if(norm1 is None):
            self.norm1 = nn.LayerNorm(emb_dim) 
        elif(norm1==False):
            self.norm1 = nn.Identity()
            
        self.norm2 = norm2
        if(norm2 is None):
            self.norm2 = nn.LayerNorm(emb_dim) 
        elif(norm2==False):
            self.norm2 = nn.Identity()
        
    
    def forward(self, x):
        
        context_emb, mask = x
        
        #compute sequence representation
        y =  self.self_attention(x)
        
        #sum input & output
        out1 = context_emb + y 
        
        #Normalisation
        out1 = self.norm1(out1)
        
        #feed mlp
        out2 = self.mlp(out1)
        
        #Normalisation
        outputs = self.norm2(out1+out2)
        
        return outputs, mask


class TransformerQ1Q2(nn.Module):
    
    def __init__(self, embeddings, word2id, emb_dim=50, nb_classe= 2, L=3, context_model=None, \
                 values_model=None, keys_model=None, query_model=None, mlp=None, classifier=None, max_len=0, num_sample=10):
        
        super(TransformerQ1Q2, self).__init__()
        self.word2id = word2id
        self.embeddings = torch.tensor(embeddings, dtype=torch.float)
        self.classifier = classifier
        self.context_model = context_model
        self.main = nn.Sequential(
                    *[AttentionBlockTP10(emb_dim, num_sample=num_sample,\
                                     values_model=None if values_model is None else values_model[i], \
                                     keys_model=None if keys_model is None else keys_model[i],\
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
            
        if(context_model is None):
            self.context_model = Contextualiser()
            
        self.pe = PositionalEncoding(d_model=emb_dim, max_len=max_len) if max_len>0 else nn.Identity()

    
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


class TransformerQ3(nn.Module):
    
    def __init__(self, embeddings, word2id, emb_dim=50, nb_classe= 2, L=3, context_model=None, \
                 values_model=None, keys_model=None, query_model=None, mlp=None, classifier=None, \
                 embedding_cls_model=None, max_len=0):
        
        super(TransformerQ3, self).__init__()
        self.word2id = word2id
        self.embeddings = torch.tensor(embeddings, dtype=torch.float)
        self.classifier = classifier
        self.context_model = context_model
        self.main = nn.Sequential(
                    *[AttentionBlockTP10(emb_dim, \
                                     values_model=None if values_model is None else values_model[i], \
                                     keys_model=None if keys_model is None else keys_model[i],\
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
            
        if(context_model is None):
            self.context_model = Contextualiser()
            
        if(embedding_cls_model is None):
            self.embedding_cls_model = nn.Linear(1, emb_dim, bias=True)
            
        self.pe = PositionalEncoding(d_model=emb_dim, max_len=max_len) if max_len>0 else nn.Identity()

    
    def forward(self, x):
        
        #embedding
        emb_x = self.embeddings[x]
        
        #positional encoding
        emb_x = self.pe(emb_x) 
        
        #add cls token embedding
        emb_cls = self.embedding_cls_model(torch.tensor([1.])).unsqueeze(0).unsqueeze(0).repeat(emb_x.shape[0],1,1)
        emb_x = torch.cat((emb_cls, emb_x), dim=1)
        
        #contextualising embedding
        context_emb = self.context_model(emb_x)
        
        #compute mask
        cls_mask = torch.zeros((emb_x.shape[0], 1), dtype=torch.float)
        mask = torch.where(x==self.word2id["__PAD__"], -float("Inf"), 0.)
        mask = torch.cat((cls_mask, mask), dim=1)
        mask = mask.unsqueeze(2).repeat(1,1,emb_x.shape[1])
        
        #compute self-attention layers
        outputs, _ = self.main((context_emb, mask))
        
        #get cls token
        cls_token = outputs[:, 0,:]
        
        y_hat = self.classifier(cls_token)
        
        return y_hat


class Transformer(nn.Module):
    
    def __init__(self, embeddings, word2id, emb_dim=50, nb_classe= 2, L=3, context_model=None, \
                 values_model=None, keys_model=None, query_model=None, mlp=None, classifier=None, \
                 embedding_cls_model=None, max_len=0, norm_layers=None):
        
        super(Transformer, self).__init__()
        self.word2id = word2id
        self.embeddings = torch.tensor(embeddings, dtype=torch.float)
        self.classifier = classifier
        self.context_model = context_model
        self.main = nn.Sequential(
                    *[AttentionBlock(emb_dim, \
                                     values_model=None if values_model is None else values_model[i], \
                                     keys_model=None if keys_model is None else keys_model[i],\
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
            
        if(context_model is None):
            self.context_model = Contextualiser()
            
        if(embedding_cls_model is None):
            self.embedding_cls_model = nn.Linear(1, emb_dim, bias=True)
            
        if(norm_layers is None):
            self.norm_layers = nn.Linear(1, emb_dim, bias=True)
            
        self.pe = PositionalEncoding(d_model=emb_dim, max_len=max_len) if max_len>0 else nn.Identity()

    
    def forward(self, x):
        
        #embedding
        emb_x = self.embeddings[x]
        
        #positional encoding
        emb_x = self.pe(emb_x) 
        
        #add cls token embedding
        emb_cls = self.embedding_cls_model(torch.tensor([1.])).unsqueeze(0).unsqueeze(0).repeat(emb_x.shape[0],1,1)
        emb_x = torch.cat((emb_cls, emb_x), dim=1)
        
        #contextualising embedding
        context_emb = self.context_model(emb_x)
        
        #compute mask
        cls_mask = torch.zeros((emb_x.shape[0], 1), dtype=torch.float)
        mask = torch.where(x==self.word2id["__PAD__"], -float("Inf"), 0.)
        mask = torch.cat((cls_mask, mask), dim=1)
        mask = mask.unsqueeze(2).repeat(1,1,emb_x.shape[1])
        
        #compute self-attention layers
        outputs, _ = self.main((context_emb, mask))
        
        #get cls token
        cls_token = outputs[:, 0,:]
        
        y_hat = self.classifier(cls_token)
        
        return y_hat



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
        

#################################
########### Trainning ###########
#################################

class InfiniteLoader():
    """
    To iterate indefinitely on a loader
    """
    def __init__(self, loader):
        self.loader = loader
        self.iterator = iter(loader)
        
    def get_batch(self):
        try:
            x = next(self.iterator)
        except:
            self.iterator = iter(self.loader)
            x = next(self.iterator)
        return x


def train(train_loader, test_loader, model, optimizer, criterion, nb_step, nb_step_val, interval_step_val, \
            path=None, path_early_stopping=None, patience=20, verbose=False):
    
    ############################
    # Prepare verbose settings #
    ############################
    
    progress_bar = tqdm if verbose else lambda first_arg, **kwargs: first_arg
    
    ###################
    # Infinite Loader #
    ###################
    train_batchs = InfiniteLoader(train_loader)
    test_batchs = InfiniteLoader(test_loader)
    
    ##################
    # Initialisation #
    ##################
    if(path!=None and os.path.isfile(path)):
        
        checkpoint = torch.load(path)
        
        model.load_state_dict(checkpoint["model_params"])
        
        if("early_stopping" in checkpoint):
            early_stopping = checkpoint["early_stopping"]
            path_early_stopping = checkpoint["path_early_stopping"]
        
        current_step = checkpoint["current_step"]
        
        
        optimizer.load_state_dict(checkpoint["optimizer_params"])
        
        train_loss = checkpoint["train_loss"]
        eval_loss = checkpoint["eval_loss"]
    else:

        current_step = 0

        train_loss = []
        eval_loss = []
        
        if(path_early_stopping is not None):
            early_stopping = EarlyStopping(patience=patience, verbose=verbose, delta=1e-10, path=path_early_stopping)
        
    
    ############
    # Training #
    ############
    tmp_train_loss = []
    model.train()
    for step in progress_bar(range(current_step, nb_step), initial=current_step, total=nb_step):
        
        data, labels = train_batchs.get_batch()

        optimizer.zero_grad()

        outputs = model(data)


        loss = criterion(outputs, labels)
        tmp_train_loss.append(loss.item())
        loss.backward()
        optimizer.step()
        
        ##############
        # Evaluation #
        ##############
        if(step%interval_step_val==0 or step==nb_step-1):
            
            with torch.no_grad():
                
                tmp_eval_loss = []
                model.eval()
                for step_val in range(nb_step_val):
                    
                    data, labels = test_batchs.get_batch()

                    outputs = model(data)

                    loss = criterion(outputs, labels)
                    tmp_eval_loss.append(loss.item())

                eval_loss.append(np.mean(tmp_eval_loss)) 
            model.train()
            
            ##################
            # Early stopping #
            ##################
            if(path_early_stopping is not None):
                early_stopping(eval_loss[-1], model)
            
            ##############################
            # Saving Parameters and loss #
            ##############################
            train_loss.append(np.mean(tmp_train_loss))
            
            tmp_train_loss = []
            
            if(path!=None):
                checkpoint = {
                    "model": model.__class__,
                    "model_params": model.state_dict(),

                    "current_step": step+1,

                    "optimizer": optimizer.__class__,
                    "optimizer_params": optimizer.state_dict(),

                    "train_loss": train_loss,
                    "eval_loss": eval_loss,
                }
                if(path_early_stopping is not None):
                    checkpoint["early_stopping"] = early_stopping
                    checkpoint["path_early_stopping"] = path_early_stopping
                    
                torch.save(checkpoint, path)
                
            if(path_early_stopping is not None and early_stopping.early_stop):
                print("Early stopping")
                break
    
    if(path_early_stopping is not None):
        model.load_state_dict(torch.load(path_early_stopping))
            
    return train_loss, eval_loss


def get_prediction(model, loader, nb_step=None, verbose=False):
    
    progress_bar = tqdm if verbose else lambda first_arg, **kwargs: first_arg
    
    batchs = InfiniteLoader(loader)
    
    predictions = torch.tensor([])
    targets = torch.tensor([])
    
    with torch.no_grad():
        for step in progress_bar(range(nb_step)):
            
            data, labels = batchs.get_batch()
            outputs, _ = model(data)
            predictions = torch.cat((predictions, outputs.argmax(dim=1)))
            targets = torch.cat((targets, labels))
        
    return predictions, targets

def get_all_prediction(model, loader, verbose=False):
    
    progress_bar = tqdm if verbose else lambda first_arg, **kwargs: first_arg
    
    predictions = torch.tensor([])
    targets = torch.tensor([])
    
    with torch.no_grad():
        for data, labels in progress_bar(loader):
            outputs = model(data)
            predictions = torch.cat((predictions, outputs.argmax(dim=1)))
            targets = torch.cat((targets, labels))
        
    return predictions, targets

############################
########### Data ###########
############################

class DatasetTP9(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        data, target = self.dataset[index]
        return torch.tensor(data), target

    def __len__(self):
        return len(self.dataset)



class FolderText(Dataset):
    """Dataset basé sur des dossiers (un par classe) et fichiers"""

    def __init__(self, classes, folder: Path, tokenizer, load=False):
        self.tokenizer = tokenizer
        self.files = []
        self.filelabels = []
        self.labels = {}
        for ix, key in enumerate(classes):
            self.labels[key] = ix

        for label in classes:
            for file in (folder / label).glob("*.txt"):
                self.files.append(file.read_text(encoding="utf-8") if load else file)
                self.filelabels.append(self.labels[label])

    def __len__(self):
        return len(self.filelabels)

    def __getitem__(self, ix):
        s = self.files[ix]
        return self.tokenizer(s if isinstance(s, str) else s.read_text(encoding="utf-8")), self.filelabels[ix]

def get_imdb_data(embedding_size=50):
    """Renvoie l'ensemble des donnéees nécessaires pour l'apprentissage

    - dictionnaire word vers ID
    - embeddings (Glove)
    - DataSet (FolderText)

    """
    WORDS = re.compile(r"\S+")

    words, embeddings = prepare_dataset('edu.stanford.glove.6b.%d' % embedding_size).load()
    OOVID = len(words)
    PAD = len(words)+1
    words.append("__OOV__")
    words.append("__PAD__")

    word2id = {word: ix for ix, word in enumerate(words)}
    embeddings = np.vstack((embeddings, np.zeros(embedding_size)))
    embeddings = np.vstack((embeddings, np.zeros(embedding_size)))

    def tokenizer(t):
        return [word2id.get(x, OOVID) for x in re.findall(WORDS, t.lower())]

    logging.info("Loading embeddings")

    logging.info("Get the IMDB dataset")
    ds = prepare_dataset("edu.stanford.aclimdb")

    return word2id, embeddings, FolderText(ds.train.classes, ds.train.path, tokenizer, load=False), FolderText(ds.test.classes, ds.test.path, tokenizer, load=False)

