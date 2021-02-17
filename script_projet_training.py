import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchtext
from sklearn.metrics import accuracy_score

from utils.data import DatasetIMDB, DatasetDBpedia, DatasetLM1B, get_imdb_data
from utils.train import train, get_all_prediction
from utils.models import TransPerformer, Contextualiser



def collate_fn(samples):

    global word2id
    PAD_IX = word2id["__PAD__"]
    lenMax = np.max([len(e) for e, _ in samples])
    res = []
    targets = []

    for sample, target in samples:
        pads = torch.full((lenMax-len(sample),), PAD_IX, dtype=torch.int)
        res.append(torch.cat((sample, pads), 0))
        targets.append(target)

    return torch.stack(res).long(), torch.tensor(targets)


class SampleDataset(Dataset):

    def __init__(self, dataset, nb_sample):
        self.data = []
        indices = np.random.choice(len(dataset), size=nb_sample, replace=False)
        for ind in indices:
            self.data.append(dataset[ind])

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)



if(__name__ == "__main__"):

	#Declaration de constantes
	EMBEDDING_DIM = 50
	BATCH_SIZE = 16
	NB_CLASSE = 2
	CLASSIF_INTER_DIM = 10
	L = 3
	NUM_SAMPLE = False #valeur par défaut pour désactiver l'approximation du softmax
	MAX_LEN = False #valeur par défaut pour désactiver le positinal encodding

	#Récupération de l'embbedings et du dictionnaire
	
	word2id, embeddings, train_ds_imdb_tmp, test_ds_imdb_tmp = get_imdb_data(embedding_size=EMBEDDING_DIM)
	id2word = {value:key for key,value in word2id.items() }

	#Récupération des données sours forme de dataset/dataloader
	train_ds_imdb, test_ds_imdb = DatasetIMDB(train_ds_imdb_tmp), DatasetIMDB(test_ds_imdb_tmp)
	train_dataset, test_dataset = SampleDataset(train_ds_imdb, 200), SampleDataset(test_ds_imdb, 200) #Pour tester avec un extrait des données

	train_loader = DataLoader(train_dataset, collate_fn=collate_fn, shuffle=True, batch_size=BATCH_SIZE)
	test_loader = DataLoader(test_dataset, collate_fn=collate_fn, shuffle=True, batch_size=BATCH_SIZE)
	criterion = nn.CrossEntropyLoss()

	#Paramétrage du modèle
	context_model = nn.Identity()#Contextualiser(input_size=EMBEDDING_DIM, hidden_size=EMBEDDING_DIM)

	query_model = [nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM, bias=True) for _ in range(L)]
	key_model = [nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM, bias=True) for _ in range(L)]
	value_model = [nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM, bias=True) for _ in range(L)]

	mlp = [nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM, bias=True) for _ in range(L)]

	classifier = nn.Sequential( nn.Linear(EMBEDDING_DIM, CLASSIF_INTER_DIM, bias=True),
								nn.ReLU(),
								nn.Linear(CLASSIF_INTER_DIM, NB_CLASSE, bias=True) )


	model = TransPerformer(embeddings=embeddings, word2id=word2id, nb_classe=NB_CLASSE, \
                L=L, emb_dim=EMBEDDING_DIM, max_len=MAX_LEN, num_sample=NUM_SAMPLE, \
                norm1=True, norm2=True, \
                context_model=context_model, classifier=classifier, mlp=mlp, \
                query_model=query_model, key_model=key_model, value_model=value_model)


	#Paramétrage de l'entrainement & entrainement
	optimizer = torch.optim.Adam(params=model.parameters())
	nb_step = 20
	nb_step_val = 5
	interval_step_val = 25

	train_loss, test_loss = train(train_loader, test_loader, model, optimizer, criterion, \
											nb_step=nb_step, nb_step_val=nb_step_val, interval_step_val=interval_step_val,\
											verbose=True, path="checkpoint.pt", path_early_stopping="bestParams.pt")
