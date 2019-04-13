import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import math
from model.common_layer import EncoderLayer, DecoderLayer, MultiHeadAttention, Conv, PositionwiseFeedForward, LayerNorm , _gen_bias_mask ,_gen_timing_signal, LabelSmoothing, NoamOpt, _get_attn_subsequent_mask,  get_input_from_batch, get_output_from_batch
from utils import config
import random
from numpy import random
import os
import pprint
from tqdm import tqdm
pp = pprint.PrettyPrinter(indent=1)
import os
import time

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel

random.seed(123)
torch.manual_seed(123)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(123)

class RefineDecoder(nn.Module):
    def __init__(self):
        super(RefineDecoder, self).__init__()
        pass

    def forward(self):
        pass

class DraftDecoder(nn.Module):
    """
    A Transformer Decoder module. 
    Inputs should be in the shape [batch_size, length, hidden_size]
    Outputs will have the shape [batch_size, length, hidden_size]
    Refer Fig.1 in https://arxiv.org/pdf/1706.03762.pdf
    """
    def __init__(self, embedding_size, hidden_size, num_layers, num_heads, total_key_depth, total_value_depth,
                 filter_size, max_length=config.max_enc_steps, input_dropout=0.0, layer_dropout=0.0, 
                 attention_dropout=0.0, relu_dropout=0.0):
        """
        Parameters:
            embedding_size: Size of embeddings
            hidden_size: Hidden size
            num_layers: Total layers in the Encoder
            num_heads: Number of attention heads
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            filter_size: Hidden size of the middle layer in FFN
            max_length: Max sequence length (required for timing signal)
            input_dropout: Dropout just after embedding
            layer_dropout: Dropout for each layer
            attention_dropout: Dropout probability after attention (Should be non-zero only during training)
            relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
        """
        
        super(DraftDecoder, self).__init__()
        self.num_layers = num_layers
        self.timing_signal = _gen_timing_signal(max_length, hidden_size)
        
        self.mask = _get_attn_subsequent_mask(max_length) # mask to hide future

        params =(hidden_size, 
                total_key_depth or hidden_size,
                total_value_depth or hidden_size,
                filter_size, 
                num_heads, 
                _gen_bias_mask(max_length), # mandatory
                layer_dropout, 
                attention_dropout, 
                relu_dropout)
        
        self.embedding_proj = nn.Linear(embedding_size, hidden_size, bias=False)

        # input to decoder: tuple consisting of decoder inputs and encoder output
        self.dec = nn.Sequential(*[DecoderLayer(*params) for l in range(num_layers)])
        
        self.layer_norm = LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(input_dropout)

    def forward(self, inputs, encoder_output, mask):
        mask_src, mask_trg = mask
        dec_mask = torch.gt(mask_trg + self.mask[:, :mask_trg.size(-1), :mask_trg.size(-1)], 0)
        #Add input dropout
        x = self.input_dropout(inputs)
        # Project to hidden size
        x = self.embedding_proj(x)
        x += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)

        # Project to hidden size
        encoder_output = self.embedding_proj(encoder_output) #.transpose
        # Run decoder. Input: x, encoder_outputs, attention_weight, mask_src, dec_mask
        y, _, attn_dist, _ = self.dec((x, encoder_output, [], (mask_src,dec_mask)))

        # Final layer normalization
        y = self.layer_norm(y)
        return y, attn_dist


class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)
        self.p_gen_linear = nn.Linear(config.hidden_dim, 1)
        self.copyloss = nn.BCELoss()
        self.m = nn.Sigmoid()

    def forward(self, x, attn_dist=None, enc_batch_extend_vocab=None, extra_zeros=None, temp=1, beam_search=False, copy_gate=None, copy_ptr=None, mask_trg=None):

        if config.pointer_gen:
            p_gen = self.p_gen_linear(x)
            p_gen = self.m(p_gen)

        logit = self.proj(x) # simple linear projection

        if config.pointer_gen:
            vocab_dist = F.softmax(logit/temp, dim=2)
            vocab_dist_ = p_gen * vocab_dist

            attn_dist_ = F.softmax(attn_dist/temp, dim=-1)
            attn_dist_ = (1 - p_gen) * attn_dist_           
            enc_batch_extend_vocab_ = torch.cat([enc_batch_extend_vocab.unsqueeze(1)]*x.size(1),1) ## extend for all seq
            if(beam_search):
                enc_batch_extend_vocab_ = torch.cat([enc_batch_extend_vocab_[0].unsqueeze(0)]*x.size(0),0) ## extend for all seq
            #print(enc_batch_extend_vocab_.size(), attn_dist_.size())
            logit = torch.log(vocab_dist_.scatter_add(2, enc_batch_extend_vocab_, attn_dist_))
            
            return logit
        else:
            return F.log_softmax(logit,dim=-1)

class Transformer(nn.Module):

    def __init__(self, model_file_path=None, is_eval=False, load_optim=False):
        super(Transformer, self).__init__()

        self.encoder = BertModel.from_pretrained('bert-base-uncased')
        self.embedding = self.encoder.embeddings
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.encoder.eval() # BERT model always in eval mode
        self.decoder = DraftDecoder(config.emb_dim, config.hidden_dim, num_layers=config.hop, num_heads=config.heads, 
                                total_key_depth=config.depth,total_value_depth=config.depth,
                                filter_size=config.filter)
        self.generator = Generator(config.hidden_dim,config.vocab_size)

        self.criterion = nn.NLLLoss(ignore_index=config.PAD_idx)
        self.criterion_ppl = nn.NLLLoss(ignore_index=config.PAD_idx)
        if (config.label_smoothing):
            self.criterion = LabelSmoothing(size=self.vocab_size, padding_idx=config.PAD_idx, smoothing=0.1)
        if is_eval:
            self.decoder = self.decoder.eval()
            self.generator = self.generator.eval()
            self.embedding = self.embedding.eval()

    
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config.lr)
        if(config.noam):
            self.optimizer = NoamOpt(config.hidden_dim, 1, 4000, torch.optim.Adam(self.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
        if model_file_path is not None:
            print("loading weights")
            state = torch.load(model_file_path, map_location= lambda storage, location: storage)
            print("LOSS",state['current_loss'])
            self.decoder.load_state_dict(state['decoder_state_dict'])
            self.generator.load_state_dict(state['generator_dict'])
            self.embedding.load_state_dict(state['embedding_dict'])
            if (load_optim):
                self.optimizer.load_state_dict(state['optimizer'])

        if (config.USE_CUDA):
            self.encoder = self.encoder.cuda()
            self.decoder = self.decoder.cuda()
            self.generator = self.generator.cuda()
            self.criterion = self.criterion.cuda()
            self.embedding = self.embedding.cuda()
        self.model_dir = config.save_path
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.best_path = ""

    def save_model(self, running_avg_ppl, iter, r1,r2,rl,r_avg):
        state = {
            'iter': iter,
            'decoder_state_dict': self.decoder.state_dict(),
            'generator_dict': self.generator.state_dict(),
            'embedding_dict': self.embedding.state_dict(),
            #'optimizer': self.optimizer.state_dict(),
            'current_loss': running_avg_ppl
        }
        model_save_path = os.path.join(self.model_dir, 'model_{}_{:.4f}_{:.4f}_{:.4f}_{:.4f}_{:.4f}'.format(iter,running_avg_ppl,r1,r2,rl,r_avg) )
        self.best_path = model_save_path
        torch.save(state, model_save_path)

    def train_one_batch(self, batch, train=True):
        ## pad and other stuff
        input_ids_batch, input_mask_batch, example_index_batch, enc_batch_extend_vocab, extra_zeros, _ = get_input_from_batch(batch)
        dec_batch, copy_gate, copy_ptr = get_output_from_batch(batch)
        
        if(config.noam):
            self.optimizer.optimizer.zero_grad()
        else:
            self.optimizer.zero_grad()
        encoder_outputs, _ = self.encoder(input_ids_batch, token_type_ids=enc_batch_extend_vocab, attention_mask=input_mask_batch)

        # Decode 
        sos_token = torch.LongTensor([config.SOS_idx] * input_ids_batch.size(0)).unsqueeze(1)
        if config.USE_CUDA: sos_token = sos_token.cuda()
        dec_batch_shift = torch.cat((sos_token,dec_batch[:, :-1]),1) # shift the decoder input (summary) by one step
        mask_trg = dec_batch_shift.data.eq(config.PAD_idx).unsqueeze(1)

        pre_logit, attn_dist = self.decoder(self.embedding(dec_batch_shift),encoder_outputs[-1], (None,mask_trg))
        ## compute output dist
        logit = self.generator(pre_logit,attn_dist,enc_batch_extend_vocab, extra_zeros, copy_gate=copy_gate, copy_ptr=copy_ptr, mask_trg= mask_trg)
        ## loss: NNL if ptr else Cross entropy
        loss = self.criterion(logit.contiguous().view(-1, logit.size(-1)), dec_batch.contiguous().view(-1))
        
        if(train):
            loss.backward()
            self.optimizer.step()
        #if(config.label_smoothing): 
        loss_ppl = self.criterion_ppl(logit.contiguous().view(-1, logit.size(-1)), dec_batch.contiguous().view(-1))
        return loss_ppl.item(), math.exp(min(loss_ppl.item(), 100)), loss

    def decoder_greedy(self, batch):
        input_ids_batch, input_mask_batch, example_index_batch, enc_batch_extend_vocab, extra_zeros, _ = get_input_from_batch(batch)
        # mask_src = enc_batch.data.eq(config.PAD_idx).unsqueeze(1)

        encoder_outputs, _ = self.encoder(input_ids_batch, token_type_ids=enc_batch_extend_vocab, attention_mask=input_mask_batch)

        ys = torch.ones(1, 1).fill_(config.SOS_idx).long()
        if config.USE_CUDA:
            ys = ys.cuda()
        mask_trg = ys.data.eq(config.PAD_idx).unsqueeze(1)
        decoded_words = []
        for i in range(config.max_dec_step):
            out, attn_dist = self.decoder(self.embedding(ys),encoder_outputs[-1], (None,mask_trg))
            prob = self.generator(out,attn_dist,enc_batch_extend_vocab, extra_zeros)
            _, next_word = torch.max(prob[:, -1], dim = 1)

            # decoded_words.append(['<EOS>'if ni.item() == config.EOS_idx 
            #                             else self.model.index2word[ni.item()] for ni in next_word.view(-1)])

            print(next_word)  # batch size 
            decoded_words.append(self.tokenizer.convert_ids_to_tokens(next_word.tolist()))
            # decoded_words.append(''.join(self.tokenizer.convert_ids_to_tokens(next_word.tolist())))


            next_word = next_word.data[0]

            if config.USE_CUDA:
                ys = torch.cat([ys, torch.ones(1, 1).long().fill_(next_word).cuda()], dim=1)
                ys = ys.cuda()
            else:
                ys = torch.cat([ys, torch.ones(1, 1).long().fill_(next_word)], dim=1)
            mask_trg = ys.data.eq(config.PAD_idx).unsqueeze(1)

        sent = []
        for _, row in enumerate(np.transpose(decoded_words)):
            print(row)
            st = ''
            for e in row:
                if e == '<EOS>': break
                else: st+= e + ' '
            sent.append(st)
        return sent