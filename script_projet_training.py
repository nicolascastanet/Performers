import logging
import re

import math
import numpy as np
#import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import numpy as np
from datamaestro import prepare_dataset
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.distributions import Categorical

from tp10 import DatasetTP9, get_imdb_data
from tp10 import train, get_prediction
from tp10 import TransformerQ1Q2, TransformerQ3, Transformer



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


if(__name__ == "__main__"):

	word2id, embeddings, train_dataset, test_dataset = get_imdb_data()
	id2word = {value:key for key,value in word2id.items() }

	BATCH_SIZE = 64
	train_loader = DataLoader(DatasetTP9(train_dataset), collate_fn=collate_fn, shuffle=True, batch_size=BATCH_SIZE)
	test_loader = DataLoader(DatasetTP9(test_dataset), collate_fn=collate_fn, shuffle=True, batch_size=BATCH_SIZE)
	criterion = nn.CrossEntropyLoss()
	inputDim = 50

	print("##############\n##############")
	print("Q1 : ...")
	#Training model 1
	
	modelQ1 = TransformerQ1Q2(embeddings=embeddings, word2id=word2id, context_model=None, \
                       values_model=None, keys_model=None, query_model=None)

	optimizer = torch.optim.Adam(params=modelQ1.parameters(), lr=5e-3)
	nb_step = 2000
	nb_step_val = 150
	interval_step_val = 200

	train_lossQ1, test_lossQ1 = train(train_loader, test_loader, modelQ1, optimizer, criterion, \
                                         nb_step=nb_step, nb_step_val=nb_step_val, interval_step_val=interval_step_val,\
                                         verbose=True, path="checkpointQ1.pt", path_early_stopping="bestParamsQ1.pt")
    

	print("##############\n##############")
	print("Q2 : ...")
	#Training model 2
	max_len = max([data.shape[1] for data,_ in train_loader])
	#max_len = 3000

	modelQ2 = TransformerQ1Q2(embeddings=embeddings, word2id=word2id, context_model=None, \
                       values_model=None, keys_model=None, query_model=None, max_len=max_len)

	optimizer = torch.optim.Adam(params=modelQ2.parameters(), lr=5e-3)
	nb_step = 2000
	nb_step_val = 150
	interval_step_val = 200

	train_lossQ2, test_lossQ2 = train(train_loader, test_loader, modelQ2, optimizer, criterion, \
                                         nb_step=nb_step, nb_step_val=nb_step_val, interval_step_val=interval_step_val,\
                                         verbose=True, path="checkpointQ2.pt", path_early_stopping="bestParamsQ2.pt")

	print("##############\n##############")
	print("Q3 : ...")
	#Training model 3
	modelQ3 = TransformerQ3(embeddings=embeddings, word2id=word2id, context_model=None, \
                       values_model=None, keys_model=None, query_model=None, max_len=max_len)

	optimizer = torch.optim.Adam(params=modelQ3.parameters(), lr=5e-3)
	nb_step = 2000
	nb_step_val = 150
	interval_step_val = 200

	train_lossQ3, test_lossQ3 = train(train_loader, test_loader, modelQ3, optimizer, criterion, \
	                                         nb_step=nb_step, nb_step_val=nb_step_val, interval_step_val=interval_step_val,\
	                                         verbose=True, path="checkpointQ3.pt", path_early_stopping="bestParamsQ3.pt")

	print("##############\n##############")
	print("QO : ...")
	#Training model original
	modelQO = Transformer(embeddings=embeddings, word2id=word2id, context_model=None, \
                       values_model=None, keys_model=None, query_model=None, max_len=max_len)

	optimizer = torch.optim.Adam(params=modelQ3.parameters(), lr=5e-3)
	nb_step = 2000
	nb_step_val = 150
	interval_step_val = 200

	train_lossQO, test_lossQO = train(train_loader, test_loader, modelQO, optimizer, criterion, \
	                                         nb_step=nb_step, nb_step_val=nb_step_val, interval_step_val=interval_step_val,\
	                                         verbose=True, path="checkpointQO.pt", path_early_stopping="bestParamsQO.pt")