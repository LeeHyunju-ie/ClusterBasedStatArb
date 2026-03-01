import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import math
import gc




class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, r, ws, device, scale=None) :  
        self.device = device

        self.r_orig = r.fillna(0).values      

        self.ws = ws
        self.T = self.r_orig.shape[0]
        self.N = self.r_orig.shape[1]
        self.size = self.T - self.ws 
   
        if scale is None:
            scale = 1/r.values.std()             
        self.scale = scale 

        r_ws = np.zeros((self.size, self.ws, self.N))        
        for w in range(self.ws) : 
            r_ws[:, w] = self.r_orig[w:w+self.size]

        self.r_ws = torch.Tensor(r_ws*self.scale).to(device)
        self.r = torch.Tensor(np.exp(self.r_orig[-self.size:])-1).to(device)

    def __getitem__(self, t) :
        return self.r_ws[t], self.r[t]
        
    def __len__(self) :
        return self.size



def get_dataset(df_2d, ts, ws, device):
    t1, t2, t3, t4 = ts

    train_r, valid_r, test_r = df_2d[(df_2d.index >= t1) & (df_2d.index < t2)], df_2d[(df_2d.index >= t2) & (df_2d.index < t3)], df_2d[(df_2d.index >= t3) & (df_2d.index < t4)]
    valid_r = pd.concat((train_r.iloc[-ws:], valid_r), axis=0)
    test_r = pd.concat((valid_r.iloc[-ws:], test_r), axis=0)

    train_dataset = CustomDataset(train_r, ws, device)
    valid_dataset = CustomDataset(valid_r, ws, device, scale=train_dataset.scale)
    test_dataset = CustomDataset(test_r, ws, device, scale=train_dataset.scale)

    return train_dataset, valid_dataset, test_dataset



def get_dataset_adj(df_2d, ts, ws, device):
    t1, t2, t3, t4 = ts

    train_r, valid_r, test_r = df_2d[(df_2d.index >= t1) & (df_2d.index < t2)], df_2d[(df_2d.index >= t2) & (df_2d.index < t3)], df_2d[(df_2d.index >= t3) & (df_2d.index < t4)]
    valid_r = pd.concat((train_r.iloc[-ws-1:], valid_r), axis=0)
    test_r = pd.concat((valid_r.iloc[-ws-1:], test_r), axis=0)

    train_dataset = CustomDataset_adj(train_r, ws, device)
    valid_dataset = CustomDataset_adj(valid_r, ws, device, scale=train_dataset.scale)
    test_dataset = CustomDataset_adj(test_r, ws, device, scale=train_dataset.scale)

    return train_dataset, valid_dataset, test_dataset



class Transformer_module(nn.Module):
    def __init__(self, 
                 N, ws, 
                 device, 
                 n_head, 
                 n_layer,
                 d_model,
                 hidden_dim, 
                 dropout, 
                 ):
        
        super(Transformer_module, self).__init__()

        self.device = device
        
        self.N = N
        self.ws = ws
        self.hidden_dim = hidden_dim
        self.d_model = d_model

        pe_len = 10000
        positions = torch.arange(pe_len)[:, None].to(self.device)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)).to(self.device)
        self.pe = torch.zeros(pe_len, d_model).to(self.device)
        self.pe[:, 0::2] = torch.sin(positions * div_term) 
        self.pe[:, 1::2] = torch.cos(positions * div_term) 

        self.dropout = nn.Dropout(p=dropout)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, 
                                                        dim_feedforward=hidden_dim,
                                                        dropout=dropout, batch_first=True)
        self.encoder_norm = nn.LayerNorm(d_model)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=n_layer, norm=self.encoder_norm)
        self.dropout = nn.Dropout(p=dropout)

        self.fc = nn.Linear(N, d_model)
                    
    def forward(self,x):
        batch_size, T, N = x.shape
        
        x = self.fc(x)
        
        x = x * math.sqrt(self.d_model) + self.pe[None, :T, :]
        x = self.dropout(x)
            
        x = self.encoder(x) 
        return x 
    
class pt_model(nn.Module):
    def __init__(self, params, N, ws, K, device,):
        super().__init__()

        self.device = device
        
        self.params = params
        self.N = N
        self.ws = ws
        self.K = K

        self.clusterBlocks = nn.Sequential()
        self.arbitrageBlocks = nn.Sequential()
        
        self.clusterBlocks = Transformer_module(N=self.N, ws=self.ws, device=self.device, n_head=self.params['n_head'], 
                                                n_layer=self.params['n_layer'], d_model=self.params['d_model'],  hidden_dim=self.params['hidden_dim'],
                                                dropout=self.params['dropout'],
                                                )
        self.arbitrageBlocks = Transformer_module(N=self.N, ws=self.ws, device=self.device, n_head=self.params['n_head'], 
                                                    n_layer=self.params['n_layer'], d_model=self.params['d_model'],  hidden_dim=self.params['hidden_dim'],
                                                    dropout=self.params['dropout'],
                                                    )
        self.hidden_dim_fc = self.params['d_model']

        self.fc_cluster = nn.Linear(self.hidden_dim_fc, self.K)
        
        
        self.linear_w = nn.Linear(self.hidden_dim_fc,self.N)
        self.linear_z = nn.Linear(self.hidden_dim_fc,self.N)
        
        self.dropout_w = nn.Dropout(p=params['dropout'])
        self.dropout_z = nn.Dropout(p=params['dropout'])

        self.gamma = torch.nn.parameter.Parameter(torch.ones(self.N, self.K), requires_grad=True)

    def forward(self, r, z=None, c=None, temp=1, opt_z=False):
        batch_size = r.shape[0]
        ws = r.shape[1]

        r = torch.nan_to_num(r, nan=0) 
        if z is None:

            c = self.clusterBlocks(r)      
            c = self.fc_cluster(c)  
            # c = c - c.mean(axis=(1,), keepdim=True)
             
            gamma  = self.gamma[None, :, :] 
            r_hat = c[:, :, None, :] * gamma[:, None, :, :] # bs, ws, n, k

            z_prob = None
        else :
            r_hat = c[:, :, None, :]
            gamma = self.gamma[None, :, :] * 0 + 1
            z_prob = z
        s = r[:, :, :, None] - r_hat # bs, ws, n, k

        s = s.permute(0, 3, 1, 2).contiguous() # bs, k, ws, n
        s = s.view(-1, self.ws, self.N) # -1, 1, ws 

        if self.params['opt_cumsum']:
            h = self.arbitrageBlocks(s.cumsum(1)) # -1, hidden, ws / -1, ws, hidden
        else:
            h = self.arbitrageBlocks(s) # -1, hidden, ws / -1, ws, hidden

        
        w = self.linear_w(h[:,-1,:]).squeeze()
        w = self.dropout_w(w)
        
        w = w.view(batch_size, self.K, self.N).permute(0, 2, 1,)
        
        

        if z_prob is None:
            z_prob = self.linear_z(h[:,-1,:]).squeeze()
            z_prob = self.dropout_z(z_prob)
            z_prob = z_prob.view(batch_size, self.K, self.N).permute(0, 2, 1)
            
            z_prob = nn.Softmax(dim=-1)(temp*z_prob)
            
            z = torch.zeros_like(z_prob.view(-1, self.K))
            z.scatter_(1, (z_prob).view(-1, self.K).argmax(dim=-1)[:, None], 1)
            z = z.view(batch_size, self.N, self.K)
           
        return (z, z_prob), (c, s, r_hat, gamma.squeeze(0).repeat(batch_size, 1, 1)), w
    

def custom_loss(w, r, z, gamma, fee=0, eps=1e-32):
    # w: bs, n, k
    # z: bs, n, k
    # gamma: bs, n, k

    z_gamma = z*(gamma)
    w = w - torch.nan_to_num(z_gamma * torch.sum(w * z_gamma, axis=1, keepdim=True) / torch.sum(z_gamma * z_gamma, axis=1, keepdim=True), 0)
    w = w * z

    w = w/(torch.sum(torch.abs(w), axis=(1,2), keepdim=True)+eps)
    w = torch.nan_to_num(w, 0)
    
    rtns = torch.sum(r[:,:,None]*w, axis=1)    
    rtns -= 2*torch.sum(torch.abs(w), axis=1)*fee
    
    return -torch.mean(rtns)/(torch.std(rtns)+eps)


    
def get_measures(model, dataloader, fee=0, eps=1e-32, opt_return_all=False):
    weights_all = []
    r_all = []
    rws_all = []
    rhat_all = []
    c_all = []
    s_all = []
    z_all = []
    zprob_all = []
    gamma_all = []
    
    model.eval()
    with torch.no_grad():
        for r_ws, r in dataloader :
            (z, z_prob), (c, s, r_hat, gamma), w = model(r_ws)
        
            weights_all.append(w.detach().cpu().numpy().tolist())
            r_all.append(r.detach().cpu().numpy().tolist())
            z_all.append(z.detach().cpu().numpy().tolist())
            gamma_all.append(gamma.detach().cpu().numpy().tolist())
            if opt_return_all:
                rws_all.append(r_ws.detach().cpu().numpy().tolist())
                rhat_all.append(r_hat.detach().cpu().numpy().tolist())
                c_all.append(c.detach().cpu().numpy().tolist())
                s_all.append(s.detach().cpu().numpy().tolist())
                zprob_all.append(z_prob.detach().cpu().numpy().tolist())
            

    weights = np.concatenate(weights_all)
    r = np.concatenate(r_all)
    z_all = np.concatenate(z_all)
    gamma_all = np.concatenate(gamma_all)
    if opt_return_all:
        rws_all = np.concatenate(rws_all)
        rhat_all = np.concatenate(rhat_all)
        c_all = np.concatenate(c_all)
        s_all = np.concatenate(s_all)
        zprob_all = np.concatenate(zprob_all)


    sr_cluster = -custom_loss(torch.Tensor(weights), torch.Tensor(r),
                              torch.Tensor(z_all), torch.Tensor(gamma_all),
                              fee=fee).item()
    
    zgamma = z_all * gamma_all
    weights_beta = zgamma * np.sum(weights * zgamma, axis=1, keepdims=True) / np.sum(zgamma * zgamma, axis=1, keepdims=True)
    weights_beta[np.isnan(weights_beta)] = 0
    weights = weights - weights_beta
    weights = weights * z_all


    weights[np.isnan(weights)] = 0
    weights = np.sum(weights, axis=2)
    weights = weights / np.sum(abs(weights), axis=1, keepdims=True)
    weights[np.isnan(weights)] = 0


    if isinstance(fee, list):
        sr = []
        cumrtn = []
        r_pf = []
        for s in fee :
            _r_pf = (weights * r).sum(axis=1)
            weights_pre = weights[:-1] * (r+1)[:-1]
            weights_pre = weights_pre/abs(weights_pre).sum(axis=1, keepdims=True)
            weights_pre[np.isnan(weights_pre)] = 0
            _r_pf[0] -= np.sum(abs(weights[0])) * s
            _r_pf[1:] -= np.sum(abs(weights[1:] - weights_pre), axis=1) * s
            
            r_pf.append(_r_pf)
            cumrtn.append((_r_pf + 1).cumprod()[-1])
            sr.append(_r_pf.mean()/(_r_pf.std()+eps))

    else:
        r_pf = (weights * r).sum(axis=1)
        weights_pre = weights[:-1] * (r+1)[:-1]
        weights_pre = weights_pre/abs(weights_pre).sum(axis=1, keepdims=True)
        weights_pre[np.isnan(weights_pre)s] = 0
        r_pf[0] -= np.sum(abs(weights[0])) * fee
        r_pf[1:] -= np.sum(abs(weights[1:] - weights_pre), axis=1) * fee

        cumrtn = (r_pf + 1).cumprod()[-1]
        sr = r_pf.mean()/(r_pf.std()+eps)

    return (weights, r, rws_all, r_pf, z_all, zprob_all, c_all, s_all, rhat_all, gamma_all), (sr, cumrtn, sr_cluster)


def measure_performance(pf):
    sr = ((pf)*100).mean() / ((pf)*100).std() * ((252)**0.5)

    pf = (pf + 1).cumprod()
    rtn_cum = pf[-1]

    mdd = ((pf-np.maximum.accumulate(pf, axis=0)) / np.maximum.accumulate(pf, axis=0)).min()
    return rtn_cum, sr, mdd

