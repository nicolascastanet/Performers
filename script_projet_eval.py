
import torch
import numpy as np
from torch.utils.data import DataLoader

from tp10 import get_all_prediction, get_imdb_data, DatasetTP9, TransformerQ1Q2, TransformerQ3, Transformer

if(__name__ == "__main__"):

	#get data
	word2id, embeddings, train_dataset, test_dataset = get_imdb_data()

	def collate_fn(samples):

		global word2id
		PAD_IX = word2id["__PAD__"]
		lenMax = max([len(e) for e, _ in samples])
		res = []
		targets = []

		for sample, target in samples:
			pads = torch.full((lenMax-len(sample),), PAD_IX, dtype=torch.int)
			res.append(torch.cat((sample, pads), 0))
			targets.append(target)

		return torch.stack(res).long(), torch.tensor(targets)

	BATCH_SIZE = 64
	train_loader = DataLoader(DatasetTP9(train_dataset), collate_fn=collate_fn, shuffle=True, batch_size=BATCH_SIZE)
	test_loader = DataLoader(DatasetTP9(test_dataset), collate_fn=collate_fn, shuffle=True, batch_size=BATCH_SIZE)

	#get model
	modelQ1 = TransformerQ1Q2(embeddings=embeddings, word2id=word2id, context_model=None, \
                       values_model=None, keys_model=None, query_model=None)
	modelQ1.load_state_dict(torch.load("bestParamsQ1.pt"))


	max_len = max([data.shape[1] for data,_ in train_loader])

	modelQ2 = TransformerQ1Q2(embeddings=embeddings, word2id=word2id, context_model=None, \
                       values_model=None, keys_model=None, query_model=None, max_len=max_len)
	modelQ2.load_state_dict(torch.load("bestParamsQ2.pt"))


	modelQ3 = TransformerQ3(embeddings=embeddings, word2id=word2id, context_model=None, \
                       values_model=None, keys_model=None, query_model=None, max_len=max_len)
	modelQ3.load_state_dict(torch.load("bestParamsQ3.pt"))


	modelQO = Transformer(embeddings=embeddings, word2id=word2id, context_model=None, \
                       values_model=None, keys_model=None, query_model=None, max_len=max_len)
	modelQO.load_state_dict(torch.load("bestParamsQO.pt"))


	#get predictions
	preds = {}
	preds["train_predQ1"] = get_all_prediction(modelQ1, train_loader, True)
	preds["test_predQ1"] = get_all_prediction(modelQ1, test_loader, True)

	preds["train_predQ2"] = get_all_prediction(modelQ2, train_loader, True)
	preds["test_predQ2"] = get_all_prediction(modelQ2, test_loader, True)

	preds["train_predQ3"] = get_all_prediction(modelQ3, train_loader, True)
	preds["test_predQ3"] = get_all_prediction(modelQ3, test_loader, True)

	preds["train_predQO"] = get_all_prediction(modelQO, train_loader, True)
	preds["test_predQO"] = get_all_prediction(modelQO, test_loader, True)

	torch.save(preds, "predictions.pt")