import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchtext
from sklearn.metrics import accuracy_score

from utils.data import DatasetIMDB, DatasetDBpedia, DatasetLM1B, get_imdb_data
from utils.train import train, get_all_prediction
from utils.models import TransPerformer, Contextualiser
import datetime
import os
import json


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
	params = {
		'dataset' : 'IMDB',
		'num_sample': False, #valeur par défaut pour désactiver l'approximation du softmax
		'orthogonal' : True,
		'Positive' : True,
		'max_len' : False,  #valeur par défaut pour désactiver le positinal encodding
		'embedding_dim' : 50,
		'batch_size' : 16,
		'classif_inter_dim':10,
		'L' : 3
	}
		#Declaration de constantes
	NB_CLASSE = 2
	DEVICE = 'cuda'

	if params['dataset'] == 'IMDB':
		#Récupération de l'embbedings et du dictionnaire
		word2id, embeddings, train_ds_imdb_tmp, test_ds_imdb_tmp = get_imdb_data(embedding_size=params['embedding_dim'])
		id2word = {value:key for key,value in word2id.items() }

		#Récupération des données sours forme de dataset/dataloader
		train_ds_imdb, test_ds_imdb = DatasetIMDB(train_ds_imdb_tmp), DatasetIMDB(test_ds_imdb_tmp)
		#train_dataset, test_dataset = SampleDataset(train_ds_imdb, 200), SampleDataset(test_ds_imdb, 200) #Pour tester avec un extrait des données
		train_dataset, test_dataset = train_ds_imdb, test_ds_imdb
	else :
		raise NotImplementedError()
	
	train_loader = DataLoader(train_dataset, collate_fn=collate_fn, shuffle=True, batch_size=params['batch_size'])
	test_loader = DataLoader(test_dataset, collate_fn=collate_fn, shuffle=True, batch_size=params['batch_size'])
	criterion = nn.CrossEntropyLoss()

	#Paramétrage du modèle
	context_model = nn.Identity()#Contextualiser(input_size=EMBEDDING_DIM, hidden_size=EMBEDDING_DIM)

	query_model = [nn.Linear(params['embedding_dim'], params['embedding_dim'], bias=True) for _ in range(params['L'])]
	key_model = [nn.Linear(params['embedding_dim'], params['embedding_dim'], bias=True) for _ in range(params['L'])]
	value_model = [nn.Linear(params['embedding_dim'], params['embedding_dim'], bias=True) for _ in range(params['L'])]

	mlp = [nn.Linear(params['embedding_dim'], params['embedding_dim'], bias=True) for _ in range(params['L'])]

	classifier = nn.Sequential( nn.Linear(params['embedding_dim'], params['classif_inter_dim'], bias=True),
								nn.ReLU(),
								nn.Linear(params['classif_inter_dim'], NB_CLASSE, bias=True) )


	model = TransPerformer(embeddings=embeddings, word2id=word2id, nb_classe=NB_CLASSE, \
                L=params['L'], emb_dim=params['embedding_dim'], max_len=params["max_len"], num_sample=params['num_sample'], \
                norm1=True, norm2=True, \
                context_model=context_model, classifier=classifier, mlp=mlp, \
                query_model=query_model, key_model=key_model, value_model=value_model)
	model.to(DEVICE)
	optimizer = torch.optim.Adam(params=model.parameters())
	nb_step = 2000
	nb_step_val = 200
	interval_step_val = 100

	path = f"Experiment/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}/"
	train_loss, test_loss = train(train_loader, test_loader, model, optimizer, criterion, \
											nb_step=nb_step, nb_step_val=nb_step_val, interval_step_val=interval_step_val,\
											verbose=True, path=path, path_early_stopping=None)
	model.to('cpu')
	res = get_all_prediction(model, train_loader, True)
	score = accuracy_score(res[1], res[0])
	print('test_acc : ',score)
	params['test_acc'] = score
	with open(os.path.join(path,'params.txt'),"w") as f :
		json.dump(params, f,indent=4)

