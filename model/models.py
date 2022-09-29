import torch
import torch.nn as nn

import torch.nn.functional as F
import numpy as np
from torch.distributions import multivariate_normal, normal



dtype = torch.FloatTensor
eps = 1e-5
    
class INDIG(nn.Module):
    def __init__(self, num_users, num_items, num_groups, embedding_dim, group_member_dict, group_item_dict, ui_dict, drop_ratio, device):
        super(INDIG, self).__init__()
        
        self.device = device
        
        self.ui_dict = ui_dict
        
        self.z_dim = embedding_dim
        self.alpha = drop_ratio/(1-drop_ratio)
        self.userembeds = nn.Embedding(num_users, embedding_dim) #UserEmbeddingLayer(num_users, embedding_dim)

        self.itemembeds = nn.Embedding(num_items, embedding_dim) #ItemEmbeddingLayer(num_items, embedding_dim)
        self.itemembeds_mf = nn.Embedding(num_items, embedding_dim) #ItemEmbeddingLayer(num_items, embedding_dim)
        self.itemembeds_mf2 = nn.Embedding(num_items, embedding_dim) #ItemEmbeddingLayer(num_items, embedding_dim)
        self.group_member_dict = group_member_dict       
        self.group_item_dict = group_item_dict
        self.groupembeds = nn.Embedding(num_groups, embedding_dim) #GroupEmbeddingLayer(num_groups, embedding_dim)
                
        self.funw = torch.nn.Parameter(torch.randn(1,embedding_dim).type(dtype), requires_grad=True)
        self.dropfunw = nn.Dropout(drop_ratio)
        self.dropz = nn.Dropout(drop_ratio)
        self.normalize = nn.LayerNorm(embedding_dim, eps=1e-6)
        self.normalize_mf = nn.LayerNorm(embedding_dim, eps=1e-6)

        self.relu = nn.ReLU()
        self.multihead_attn = nn.MultiheadAttention(embedding_dim, 1)

        
        self.group_encoder = MLP(embedding_dim, embedding_dim, 1)
        
        self.predictlayer = PredictLayer(3 * embedding_dim )
       
        self._softplus = nn.Softplus()
        self.logsigmoid = nn.LogSigmoid()

        self.q_std = 0.1
        # initial model
        for m in self.modules():
            if isinstance(m, nn.Linear):
                #nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)
                #nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    
    
    def sample_z(self, mu):
        """Reparametrization trick
        """
        eps = np.sqrt(self.alpha)*torch.randn(mu.shape, device = self.device)

        return mu + torch.abs(mu) * eps
    
    def get_z(self):
        return self.temp_z_mu
    
    def _KL(self, mu, sigma, mup, sigmap):
        t = 2*torch.log(sigmap) - 2*torch.log(sigma) + (sigma**2)* (1/sigmap**2) + (1/sigmap**2)*((mu-mup)**2)-1
        return t*0.5
    
    def forward(self, group_inputs, item_inputs, neg_item_inputs = None):
        out = None
        N = group_inputs.shape[0]        

        funw = F.normalize(self.funw.to(self.device), p=2, dim=-1)
        funw = self.dropfunw(funw)
        
        user_aggregation= []
        for i in group_inputs.tolist():
            members = self.group_member_dict[i]            

            
            uidx = torch.LongTensor(members).to(self.device)
            members_embeds = self.userembeds(uidx)
            if self.ui_dict is not None:
                item_aggregation = []
                for us in members:
                    items = self.ui_dict[us]   
                    vidx = torch.LongTensor(items).to(self.device)
                    item_embeds = self.itemembeds_mf2(vidx)

                    item_embeds = torch.mean(item_embeds, dim=0)
                    item_aggregation.append(item_embeds.squeeze())
                item_aggregation = torch.stack(item_aggregation)
                members_embeds = members_embeds + item_aggregation
            members_embeds2 = self.normalize(members_embeds).unsqueeze(1)
            attn_output, attn_output_weights = self.multihead_attn(funw.unsqueeze(0), members_embeds2, members_embeds2)
            user_aggregation.append(attn_output.squeeze())

        user_aggregation = torch.stack(user_aggregation)

        z_p_mu = self.group_encoder(user_aggregation)
        z_p_mu = F.normalize(z_p_mu, p=2, dim=-1)
        
        item_embed = self.itemembeds(item_inputs)

        if self.training:
            #transductive learning
            z_q_mu = self.groupembeds(group_inputs) 
            
            item_aggregation = []
            for o, i in enumerate(group_inputs.tolist()):
                items = self.group_item_dict[i]            
                vidx = torch.LongTensor(items).to(self.device)
                item_embeds = self.itemembeds_mf(vidx)    

                item_embeds = torch.mean(item_embeds, dim=0)
                item_aggregation.append(item_embeds.squeeze())
                
            item_aggregation = torch.stack(item_aggregation)
            z_q_mu = z_q_mu + item_aggregation
            z_q_mu = F.normalize(z_q_mu, p=2, dim=-1)
            #z = self.dropz(z_q_mu)
            es = np.sqrt(self.alpha)*torch.randn(z_q_mu.shape, device = self.device)
            z = z_q_mu + torch.abs(z_q_mu) * es

            ncf_input = torch.cat((z*item_embed, z, item_embed), dim=-1)
            y_hidden = self.predictlayer(ncf_input)
            y_mu, y_log_sigma = torch.split(y_hidden, 1, dim=-1)

            y_mu = torch.sigmoid(y_mu) #[B,1]
            y_sigma = 0.1 + 0.9 * self._softplus(y_log_sigma)

            #transductive learning for negative samples
            nitem_embed = self.itemembeds(neg_item_inputs)
            nncf_input = torch.cat((z*nitem_embed, z, nitem_embed), dim=-1)
            ny_hidden = self.predictlayer(nncf_input)
            ny_mu, ny_log_sigma = torch.split(ny_hidden, 1, dim=-1)

            ny_mu = torch.sigmoid(ny_mu) #[B,1]
            ny_sigma = 0.1 + 0.9 * self._softplus(ny_log_sigma)            
            
            #transductive loss
            #transduc_log_p = self.logsigmoid(1.702*y_mu/torch.sqrt(y_sigma)) + 0.25* self.logsigmoid(-1.702*ny_mu/torch.sqrt(ny_sigma))
            mu = y_mu - ny_mu 
            sigma = torch.sqrt(y_sigma**2 + ny_sigma**2)  
            transduc_log_p = self.logsigmoid(1.702*mu/sigma)
            transduc_loss = torch.mean(-transduc_log_p)

            
            #inductive learning
            #implicit function theorem
            #z2 = self.dropz(z_p_mu)
            z2 = z_p_mu + torch.abs(z_p_mu) * es

            ncf_input2 = torch.cat((z2*item_embed, z2, item_embed), dim=-1)
            y_hidden2 = self.predictlayer(ncf_input2)
            y_mu2, y_log_sigma2 = torch.split(y_hidden2, 1, dim=-1)
            y_mu2 = torch.sigmoid(y_mu2) #[B,1]
            y_sigma2 = 0.1 + 0.9 * self._softplus(y_log_sigma2)
            
            #inductive learning for negative samples
            nncf_input2 = torch.cat((z2*nitem_embed, z2, nitem_embed), dim=-1)
            ny_hidden2 = self.predictlayer(nncf_input2)
            ny_mu2, ny_log_sigma2 = torch.split(ny_hidden2, 1, dim=-1)

            ny_mu2 = torch.sigmoid(ny_mu2) #[B,1]
            ny_sigma2 = 0.1 + 0.9 * self._softplus(ny_log_sigma2)            
            
            #inductive loss
            mu = y_mu2 - ny_mu2
            sigma = torch.sqrt(y_sigma2**2 + ny_sigma2**2)  
            induc_log_p = self.logsigmoid(1.702*mu/sigma)
            induc_loss = torch.mean(-induc_log_p)            
            
            #KL
            #z_q_sigma = self.alpha*(torch.abs(z_q_mu)+eps)
            #dkl = torch.sum(self._KL( z_q_mu, z_q_sigma,  z_p_mu, z_p_sigma), dim=-1, keepdim=True)
            dkl = (1+self.alpha)*torch.mean((z_p_mu-z_q_mu).pow(2))
            
            selfsup_loss = 0

            #self-sup            
            hp = (z_p_mu - z_p_mu.mean(0)) / z_p_mu.std(0)
            hq = (z_q_mu - z_q_mu.mean(0)) / z_q_mu.std(0)
            
            hip = (item_embed - item_embed.mean(0)) / item_embed.std(0)
            hiq = (nitem_embed - nitem_embed.mean(0)) / nitem_embed.std(0)

            cp = torch.mm(hp.T, hp) / N
            cq = torch.mm(hq.T, hq) / N
            
            cip = torch.mm(hip.T, hip) / N
            ciq = torch.mm(hiq.T, hiq) / N
            
            iden = torch.tensor(np.eye(cp.shape[0])).to(self.device)
            loss_dec1 = (iden - cp).pow(2).sum() + (iden - cq).pow(2).sum()
            loss_dec2 = (iden - cip).pow(2).sum() + (iden - ciq).pow(2).sum()
            selfsup_loss = (loss_dec1 + loss_dec2)           

            
            out = [transduc_loss, dkl, induc_loss, selfsup_loss]
        else:
            self.temp_z_mu = z_p_mu.detach()
            z = z_p_mu

            ncf_input = torch.cat((z*item_embed, z, item_embed), dim=-1)
            y_hidden = self.predictlayer(ncf_input)
            y_mu, y_log_sigma = torch.split(y_hidden, 1, dim=-1)
            
            y_mu = torch.sigmoid(y_mu) #[B,1]
            y_sigma = 0.1 + 0.9 * self._softplus(y_log_sigma)


            out = [y_mu, y_sigma]
        return out


class Norm(nn.Module):
    def __init__(self, d_model, eps = 1e-6):
        super().__init__()
    
        self.size = d_model
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps
    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
        / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm



class PredictLayer(nn.Module):
    def __init__(self, embedding_dim, drop_ratio=0):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(embedding_dim, 8),
            nn.ReLU(),
            nn.Dropout(drop_ratio),
            nn.Linear(8, 2)
        )

    def forward(self, x):
        out = self.linear(x)
        return out

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, num_hidden=1, drop_ratio=0):
        super().__init__()
        layers = []
        out_dim = input_dim
        hidden_dim = int((input_dim+output_dim))
        for i in range(num_hidden):
            in_dim = input_dim if i == 0 else hidden_dim
            out_dim = hidden_dim
            layers += [nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(drop_ratio)]
        # Last layer without a ReLU
        layers += [nn.Linear(out_dim, output_dim)]
        self.mlp = nn.Sequential(*layers)
                   
    def forward(self, x):
        return self.mlp(x)