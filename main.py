from model.transformer import Transformer
from model.common_layer import evaluate
from utils import config
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import os
import time 
import numpy as np 
from utils.news_data_reader import get_dataloaders, InputExample, InputFeatures

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

train_dl, val_dl, test_dl = get_dataloaders(is_small=config.small)

if(config.test):
    print("Test model",config.model)
    model = Transformer(model_file_path=config.save_path,is_eval=True)
    evaluate(model,data_loader_test,model_name=config.model,ty='test')
    exit(0)

model = Transformer()
print("TRAINABLE PARAMETERS",count_parameters(model))
print("Use Cuda: ", config.USE_CUDA)

best_rouge = 0 
cnt = 0
eval_iterval = 1% 500
for e in range(config.epochs):
    print("Epoch", e)
    p, l = [],[]
    pbar = tqdm(enumerate(train_dl),total=len(train_dl))
    for i, d in pbar:
        loss, ppl, _ = model.train_one_batch(d)
        l.append(loss)
        p.append(ppl)
        pbar.set_description("loss:{:.4f} ppl:{:.1f}".format(np.mean(l),np.mean(p)))

        if i%eval_iterval==0:
            # evaluate(model,train_dl,model_name=config.model,ty="train")
            loss,ppl_val,r_avg = evaluate(model,val_dl,model_name=config.model,ty="valid")
            if(r_avg > best_rouge):
                best_rouge = r_avg
                cnt = 0
                model.save_model(loss,e,r_avg) # running_avg_ppl, iter, r1,r2,rl)
            else: 
                cnt += 1
            if(cnt > 10): break
    # evaluate(model,train_dl,model_name=config.model,ty="train")
    # loss,ppl_val,r_avg = evaluate(model,val_dl,model_name=config.model,ty="valid")
    # r1,r2,rl = 0,0,0
    # if(ppl_val <= best_ppl):
    #     best_ppl = ppl_val
    #     cnt = 0
    #     model.save_model(best_ppl,e,r1,r2,rl,r_avg) # running_avg_ppl, iter, r1,r2,rl)
    # else: 
    #     cnt += 1
    # if(cnt > 10): break

