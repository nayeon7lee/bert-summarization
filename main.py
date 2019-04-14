from model.transformer import Summarizer
from model.common_layer import evaluate
from utils import config
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import os
import time 
import numpy as np 
from utils.data import get_dataloaders, InputExample, InputFeatures

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_draft():
    train_dl, val_dl, test_dl, tokenizer = get_dataloaders(is_small=config.small)

    if(config.test):
        print("Test model",config.model)
        model = Transformer(model_file_path=config.save_path,is_eval=True)
        evaluate(model,data_loader_test,model_name=config.model,ty='test')
        exit(0)

    model = Summarizer(is_draft=True, toeknizer=tokenizer)
    print("TRAINABLE PARAMETERS",count_parameters(model))
    print("Use Cuda: ", config.USE_CUDA)

    best_rouge = 0 
    cnt = 0
    eval_iterval = 500
    for e in range(config.epochs):
        # model.train()
        print("Epoch", e)
        l = []
        pbar = tqdm(enumerate(train_dl),total=len(train_dl))
        for i, d in pbar:
            loss = model.train_one_batch(d)
            l.append(loss.item())
            pbar.set_description("TRAIN loss:{:.4f}".format(np.mean(l)))

            if i%eval_iterval==0:
                # model.eval()
                loss,r_avg = evaluate(model,val_dl,model_name=config.model,ty="train")
                # each epoch is long,so just do early stopping here. 
                if(r_avg > best_rouge):
                    best_rouge = r_avg
                    cnt = 0
                    model.save_model(loss,e,r_avg)
                else: 
                    cnt += 1
                if(cnt > 20): break
                # model.train()
        # model.eval()
        loss,r_avg = evaluate(model,val_dl,model_name=config.model,ty="valid")


if __name__ == "__main__":
    train_draft()