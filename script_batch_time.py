from utils.models import SelfAttention
import torch
import torch.nn as nn

import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from itertools import islice
DEVICE = 'cuda'

def compute_batch(model,X,label,criterion,optimizer):
    """
    Compute forward ans backward
    """
    output = model(X)
    pred = torch.sigmoid(output.sum(dim=(1,2))).view(-1,1)
    loss = criterion(pred,label)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def batchtime_experiment(embedding_dim,num_sample,seq_size,batch_size,model_type,nb_repeat):
    '''
    Return mean time of backward + forward for given params
    '''    
    # Set up experiment
    query_model = nn.Linear(embedding_dim, embedding_dim, bias=True)
    key_model = nn.Linear(embedding_dim, embedding_dim, bias=True)
    value_model = nn.Linear(embedding_dim, embedding_dim, bias=True) 
    if  model_type =='transformer':
        model = SelfAttention(embedding_dim, value_model, key_model, query_model, num_sample=False)
    elif model_type == 'performer':
        model = SelfAttention(embedding_dim, value_model, key_model, query_model, num_sample=num_sample)
    elif model_type == 'baseline':
        model = nn.Identity()
    else : 
        raise Exception("unknow model type")
    model.to(DEVICE)
    
    optimizer = torch.optim.Adam(params=model.parameters())
    criterion = nn.BCELoss()
    # compute batch time :
    torch.cuda.empty_cache()
    start = time.time()
    for i in range(nb_repeat):
        x = torch.randn((batch_size,seq_size,embedding_dim),device=DEVICE)
        mask = torch.randn((batch_size,seq_size,seq_size),device=DEVICE)
        label = torch.zeros((batch_size,1),device=DEVICE)
        compute_batch(model,(x,mask),label,criterion,optimizer)
    end = time.time()
    perf = (end- start)/nb_repeat
    return perf

def all_batchtime_experiment(params,batch_vals,seq_vals,model_types):
    """
    Launch batchtime_experiment for each combination possible.

    Args:
      params (dict): Default params for the experimen
      batch_vals (iterable): different batch size for the experiment
      seq_vals (iterable): different size for the input sequences of the model
      model_types (string): performer/transformer/ baseline
    """
    plots = dict()
    for b in batch_vals:
        plots[b] = dict()
        for m in model_types:
            plots[b][m] = np.zeros(len(seq_vals))
            for s_i,s in enumerate(seq_vals):
                params['model_type'] = m
                params['batch_size'] = b
                params['seq_size'] = s
                try:
                    elapsed = batchtime_experiment(**params)
                    plots[b][m][s_i] = elapsed
                except RuntimeError:
                    elapsed = float('nan')
                    plots[b][m][s_i] = elapsed
                    break # if out of memory no need to test bigger sizes
    return plots


def plot_cuvres(batch_vals,seq_vals,plots,logscale_y=True):
    '''
    Plot batch time curve with appropriate legend 
    '''
    fig = plt.figure(figsize=(5, 3), dpi=300)
    ax = plt.gca()

    handles = []
        # plot experiments
    for b in batch_vals:
        color=next(ax._get_lines.prop_cycler)['color']
        plt.plot((plots[b]['transformer']),'--',label=f'batch_{b}',color=color)
        plt.plot((plots[b]['performer']),'-',label=f'batch_{b}',color=color)
        handles+= [mpatches.Patch(color=color, label=f'batch_{b}')]
        # supp info in legend
    handles+=[Line2D([0], [0], color='black', linewidth=3, linestyle=style,label=label) 
                for style,label in zip(['--','-'],["tranformer",'performer'])]
    if logscale_y:
        ax.set_yscale('log', base=2)
    plt.xlabel('log seq size')
    plt.ylabel('log_2(time)')
    plt.legend(handles = handles)
    plt.savefig("batch_time.png", bbox_inches="tight")
    plt.show()


def main():
    params = {
        'embedding_dim' : 256,
        'num_sample' :50,
        'seq_size' : 1000,
        'batch_size' : 2,   
        'nb_repeat' : 200,
        'model_type' : 'performer', # performer / transformer / baseline
    }
    seq_vals = np.logspace(1,16,num=16,base=2,dtype=int)
    batch_vals = [1,2,4,8,32]
    model_types = ['transformer','performer',]

    batchtime_experiment(**params) # le premier lancement de cette fonction est toujours plus long. Je le lance une fois dans le vide pour stabiliser les stats suivantes. 
    plots = all_batchtime_experiment(params,batch_vals,seq_vals,model_types)
    plot_cuvres(batch_vals,seq_vals,plots)

if __name__ == '__main__':
    main()