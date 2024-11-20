import os
import sys
import h5py
import inspect
import importlib
import numpy as np
import pandas as pd

import torch
import torchmetrics
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from nystrom_attention import NystromAttention

from Optimizer import create_optimizer
from utils import cross_entropy_torch, update_ema_variables, set_seed


class TransLayer(nn.Module):

    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim//8,
            heads = 8,
            num_landmarks = dim//2,
            pinv_iterations = 6,
            residual = True, 
            dropout=0.1
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))
        return x
        

class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7//2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat)+cnn_feat+self.proj1(cnn_feat)+self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


def generate_subbags(slide_embeddings, group_num, shuffle=False)->list:
    sub_length = int(slide_embeddings.shape[1]/group_num)
    subbags = list()
    
    if shuffle:
        index = np.random.permutation(slide_embeddings.shape[1])
        slide_embeddings = slide_embeddings[:, index, :]

    for i in range(group_num):
        subbag = slide_embeddings[:, int(i*sub_length):min(int((i+1)*sub_length), slide_embeddings.shape[1]), :]
        subbags.append(subbag)
    return subbags


class ATransMIL(nn.Module):
    def __init__(self, n_classes=186, embedding_size=1536, group_num=3, dim_age_embed=32):
        super(ATransMIL, self).__init__()
        self.pos_layer = PPEG(dim=512)
        self._fc1 = nn.Sequential(nn.Linear(embedding_size, 512), nn.ReLU())
        self.cls_tokens = nn.Parameter(torch.randn(1, 1, 512))
        self.n_classes = n_classes
        self.layer1 = TransLayer(dim=512)
        self.layer2 = TransLayer(dim=512)
        self.norm = nn.LayerNorm(512)
        self.dim_loc_embed = 7
        self._fc2 = nn.Linear(512+dim_age_embed+self.dim_loc_embed, self.n_classes)
        self.group_num = group_num
        self.predictor = nn.Sequential(*[nn.Linear((512+dim_age_embed+self.dim_loc_embed)*3, 4096, bias=False),
                                         nn.LayerNorm(4096),
                                         nn.ReLU(inplace=True),
                                         nn.Linear(4096, self.n_classes, bias=False),
                                         nn.LayerNorm(self.n_classes)])

    def forward(self, **kwargs):
        h = kwargs['data'].float() #[B, n, 1024]
        # Generate sub-bags from input embeddings
        sub_embedding_bag = generate_subbags(h, self.group_num, shuffle=kwargs['shuffle'])
        age = kwargs['age'].to(torch.float32)
        loc = kwargs['loc'].to(torch.float32)
        
        # Initialize dictionaries to store results for each sub-bag
        logits_dict = {}
        subembeddings_dict = {}
        instance_embeddings_dict = {}
        
        for sub_num, bag in enumerate(sub_embedding_bag):
            h_sub = bag
            h_sub = self._fc1(h_sub) #[B, n, 512]
            
            #---->pad
            H = h_sub.shape[1]
            _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
            add_length = _H * _W - H
            h_sub = torch.cat([h_sub, h_sub[:,:add_length,:]],dim = 1) #[B, N, 512]

            #---->add a unique token for each sub-bag
            B = h_sub.shape[0]
            cls_token = self.cls_tokens.expand(B, -1, -1).cuda()
            h_sub = torch.cat((cls_token, h_sub), dim=1)

            #---->Translayer x1
            h_sub = self.layer1(h_sub)

            #---->PPEG
            h_sub = self.pos_layer(h_sub, _H, _W) #[B, N, 512]
            
            #---->Translayer x2
            h_sub = self.layer2(h_sub) 

            #---->cls_token + tile embeddings
            h_sub = self.norm(h_sub)

            #---->predict
            integrated_embed = torch.cat([h_sub[:, 0], age.view(1, -1), loc.view(1, -1)], dim = 1)  # class token integrated with age and location
            integrated_logits = self._fc2(integrated_embed) # [B, n_classes]
            
            # Store logits and embeddings in dictionaries for each sub-group
            logits_dict[f'subbag_logits_{sub_num}'] = integrated_logits  # 
            subembeddings_dict[f'subbag_embed_{sub_num}'] = integrated_embed
            instance_embeddings_dict[f'subbag_instance_embed_{sub_num}'] = h_sub[:, 1:]

        # Concatenate sub-group embeddings to create a slide-level embedding
        slide_embed = torch.cat([subembeddings_dict[f'subbag_embed_{i}'] for i in range(self.group_num)], dim=1)  # [B, 3*(512+m+n)]
        # Calculate mean instance embedding for each sub-group
        mean_inst_embed = torch.cat([instance_embeddings_dict[f'subbag_instance_embed_{i}'] for i in range(self.group_num)], dim=1).mean(dim=1)  # [1, 512+m+n] 
        # Generate a list of mean instance embeddings for each sub-group
        sub_inst_embeddings = [item.mean(dim=1) for item in instance_embeddings_dict.values()]
        # Generate slide-level logits using the slide predictor
        slide_logit = self.predictor(slide_embed)  # [B, n_classes]
        logits_dict['slide_logit'] = slide_logit
        # Generate predictions and probabilities for each sub-group and slide-level logits
        Y_hat = {key: torch.argmax(values, dim=1) for key, values in logits_dict.items()}
        Y_prob = {key: F.softmax(values, dim=1) for key, values in logits_dict.items()} 
        
        results_dict = {
            'logits': logits_dict, 
            'Y_prob': Y_prob, 
            'Y_hat': Y_hat, 
            'embeddings': subembeddings_dict, 
            'slide_embed': slide_embed, 
            'mean_inst_embeddings': mean_inst_embed, 
            'sub_inst_embeddings': sub_inst_embeddings}
        return results_dict
    

class ContrastiveLoss(nn.Module):
    def __init__(self, gap=0.2, eps=1e-8):
        super(ContrastiveLoss, self).__init__()
        self.gap = gap
        self.eps = eps

    def forward(self, embeddings: torch.Tensor, template: torch.Tensor, label: int, sub_embeddings: list):
        # Normalize embeddings and template
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        template_norm = F.normalize(template, p=2, dim=1) #[:108]
        # nan_row_idx = torch.isnan(template_norm).any(dim=1)
        template_norm = torch.nan_to_num(template_norm, nan=0.0)
        
        # Normalize concatenated sub-embeddings and calculate similarity matrix
        sub_mean_embeddings = F.normalize(torch.cat(sub_embeddings, dim=0), p=2, dim=1)
        sim_mat = torch.mm(sub_mean_embeddings, sub_mean_embeddings.t())
        loss_inner = 3 - torch.sum(torch.tril(sim_mat, diagonal=-1))
        
        # Calculate cosine similarity between embeddings and template
        cos_sim = torch.mm(embeddings_norm, template_norm.t())
        
        # Calculate loss for the correct label
        loss_same = 1 - cos_sim[0, label]
        
        # Calculate loss for incorrect labels
        mask = (torch.arange(cos_sim.shape[1]) != label).cuda()# & (~nan_row_idx)
        loss_dif = torch.sum(torch.clamp(cos_sim-self.gap, min=0) * mask.float())
        # Compute final loss
        loss = 0.4 * loss_same + 0.5 * loss_dif / (cos_sim.shape[1] - 1) + 0.1 * loss_inner
        return loss


class ModelModule(pl.LightningModule):
    def __init__(self, Model, Optimizer, **kargs):
        super(ModelModule, self).__init__()
        self.save_hyperparameters()
        self.load_model()

        # Initialize loss and optimizer
        self.loss = nn.CrossEntropyLoss()
        self.contrastive_loss = ContrastiveLoss()
        self.cl_w = float(Model.cl_w)
        self.optimizer = Optimizer

        # Clustering template for hidden space
        self.cls_template = torch.full((int(Model.n_classes), 512), torch.nan).to('cuda')

        # Experiment settings
        self.exp_name = Model.exp_name
        self.n_classes = Model.n_classes
        self.fold = kargs['Data'].fold
        self.preds_save_dir = kargs['Data'].preds_save
        self.log_path = kargs['log_path']

        self.val_step_outputs = []
        self.test_step_outputs = []

        # Metrics for training, validation, and testing
        self.train_count = [{"count": 0, "correct": 0} for _ in range(self.n_classes)]
        self.val_count = [{"count": 0, "correct": 0} for _ in range(self.n_classes)]
        self.test_count = [{"count": 0, "correct": 0} for _ in range(self.n_classes)]

        # Metrics for multi-class and binary-class classification
        if self.n_classes > 2: 
            self.AUROC = torchmetrics.AUROC(num_classes=self.n_classes, average='macro', task='multiclass')
            metrics = torchmetrics.MetricCollection([
                torchmetrics.Accuracy(num_classes=self.n_classes, average='micro', task='multiclass'),
                torchmetrics.CohenKappa(num_classes=self.n_classes, task='multiclass'),
                torchmetrics.F1Score(num_classes=self.n_classes, average='macro', task='multiclass'),
                torchmetrics.Recall(num_classes=self.n_classes, average='macro', task='multiclass'),
                torchmetrics.Precision(num_classes=self.n_classes, average='macro', task='multiclass'),
                torchmetrics.Specificity(num_classes=self.n_classes, average='macro', task='multiclass')
            ])
        else: 
            self.AUROC = torchmetrics.AUROC(num_classes=2, average='macro')
            metrics = torchmetrics.MetricCollection([
                torchmetrics.Accuracy(num_classes=2, average='micro'),
                torchmetrics.CohenKappa(num_classes=2),
                torchmetrics.F1Score(num_classes=2, average='macro'),
                torchmetrics.Recall(num_classes=2, average='macro'),
                torchmetrics.Precision(num_classes=2, average='macro')
            ])
        
        self.valid_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')

    def training_step(self, batch, batch_idx):
        data, age, loc, label, _ = batch
        results_dict = self.model(data=data, label=label, age=age, loc=loc, shuffle=False)

        logits = results_dict['logits']
        Y_hat = results_dict['Y_hat']

        current_epoch = self.current_epoch
        mean_inst_embeddings = results_dict['mean_inst_embeddings']
        sub_inst_embeddings = results_dict['sub_inst_embeddings']

         # Compute loss
        if isinstance(logits, dict):
            Y_hat = int(Y_hat['slide_logit'].item())
            Y = int(torch.argmax(label).item())
            self.train_count[Y]["correct"] += (Y_hat == Y)
            self.train_count[Y]["count"] += 1

            # Classification loss
            loss = self.loss(logits['slide_logit'], label)  # Cross-entropy loss for slide-level logits
            loss += self.calculate_group_loss(logits, label)  # +Cross-entropy loss for sub-group logits
            
            # Log losses
            self.log('train_loss', loss, batch_size=data.shape[0], prog_bar=True, on_epoch=True, logger=True)
            
            # Add contrastive loss if epoch > 0
            if current_epoch > 0:
                feature_loss = self.contrastive_loss(mean_inst_embeddings, self.cls_template, Y, sub_inst_embeddings)
                loss += self.cl_w * feature_loss

            # Update template for hidden space clustering
            if torch.isnan(self.cls_template[Y]).any():
                self.cls_template[Y, :] = mean_inst_embeddings.detach()
            else:
                self.cls_template[Y, :] = update_ema_variables(self.cls_template[Y], mean_inst_embeddings.detach(), current_epoch)  
                        
        else:
            loss = self.loss(logits, label)
            Y = int(torch.argmax(label).item())
            self.train_count[Y]["correct"] += (Y_hat.item() == Y)
            self.train_count[Y]["count"] += 1
        #---->acc log

        return {'loss': loss}     
    
    def calculate_group_loss(self, logits, label):
        # Calculate group loss for each subgroup
        group_loss = 0
        for i in range(self.model.group_num):
            group_loss += 1 / self.model.group_num * self.loss(logits[f'subbag_logits_{i}'], label)
        return group_loss
    
    def on_train_epoch_end(self):
        cls_acc_train = []
        for c in range(self.n_classes):
            count = self.train_count[c]["count"]
            correct = self.train_count[c]["correct"]
            if count == 0:
                acc = None
            else:
                acc = float(correct) / count
                # print('class {}: acc {}, correct {}/{}'.format(c, acc, correct, count))
                cls_acc_train.append(acc)
        
        print("Macro Acc: ", np.mean(cls_acc_train))
        self.train_count = [{"count": 0, "correct": 0} for i in range(self.n_classes)]

    def validation_step(self, batch, batch_idx):
        data, age, loc, label, _ = batch
        results_dict = self.model(data=data, label=label, age=age, loc=loc, shuffle=False)
        logits = results_dict['logits']
        Y_prob = results_dict['Y_prob']
        Y_hat = results_dict['Y_hat']
        Y = label

        if isinstance(logits, dict):     
            self.val_count[Y]["correct"] += (int(Y_hat['slide_logit'].item()) == Y)
        else:
            self.val_count[Y]["correct"] += (int(Y_hat.item()) == Y)
        self.val_count[Y]["count"] += 1

        val_results = {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat, 'label': label}
        self.val_step_outputs.append(val_results)
        
        return val_results
    
    def on_validation_epoch_end(self):
        if isinstance(self.val_step_outputs[0]['logits'], dict):
            logits = torch.cat([x['logits']['slide_logit'] for x in self.val_step_outputs], dim=0)
            probs = torch.cat([x['Y_prob']['slide_logit'] for x in self.val_step_outputs], dim=0)
            max_probs = torch.stack([x['Y_hat']['slide_logit'] for x in self.val_step_outputs])
        else:
            logits = torch.cat([x['logits'] for x in self.val_step_outputs], dim=0)
            probs = torch.cat([x['Y_prob'] for x in self.val_step_outputs], dim=0)
            max_probs = torch.stack([x['Y_hat'] for x in self.val_step_outputs])

        target = torch.stack([x['label'] for x in self.val_step_outputs], dim=0)
        metric_results_dict = self.valid_metrics(max_probs.squeeze() , target.squeeze())

        self.log('multi_acc', metric_results_dict['val_MulticlassAccuracy'], prog_bar=True, on_epoch=True, logger=True)
        self.log('val_loss', cross_entropy_torch(logits, target), prog_bar=True, on_epoch=True, logger=True)
        self.log('auc', self.AUROC(probs, target.squeeze()), prog_bar=True, on_epoch=True, logger=True)
        self.log_dict(metric_results_dict, on_epoch=True, logger=True)

        for c in range(self.n_classes):
            count = self.val_count[c]["count"]
            correct = self.val_count[c]["correct"]
            if count == 0:
                acc = None
            else:
                acc = float(correct) / count
                print('class {}: acc {}, correct {}/{}'.format(c, acc, correct, count))

        self.val_count = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
        self.val_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        data, age, loc, label, slide_id = batch
        results_dict = self.model(data=data, label=label, age=age, loc=loc, shuffle=False)
        logits = results_dict['logits']
        Y_prob = results_dict['Y_prob']
        Y_hat = results_dict['Y_hat']
        embeddings = results_dict['slide_embed']
        Y = int(label) 

        if isinstance(logits, dict):
            self.test_count[Y]["correct"] += (int(Y_hat['slide_logit'].item()) == Y)
        else:
            self.test_count[Y]["correct"] += (int(Y_hat.item()) == Y)
        self.test_count[Y]["count"] += 1

        results_dict = {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat, 'label': label, 
                        'slide_id': slide_id, 'slide_features': embeddings} 
        self.test_step_outputs.append(results_dict)
        
        return results_dict

    def on_test_epoch_end(self):
        if isinstance(self.test_step_outputs[0]['logits'], dict):
            probs = torch.cat([x['Y_prob']['slide_logit'] for x in self.test_step_outputs], dim=0)
            max_probs = torch.stack([x['Y_hat']['slide_logit'] for x in self.test_step_outputs])
            target = torch.stack([x['label'] for x in self.test_step_outputs], dim=0)
        else:
            probs = torch.cat([x['Y_prob'] for x in self.test_step_outputs], dim=0)
            max_probs = torch.stack([x['Y_hat'] for x in self.test_step_outputs])
            target = torch.stack([x['label'] for x in self.test_step_outputs], dim=0)

        slide_embed = torch.cat([x['slide_features'] for x in self.test_step_outputs], dim=0)
        id_set = [x['slide_id'][0] for x in self.test_step_outputs]

        # Save predictions (slide IDs, predicted probabilities, labels, slide features) to a .h5 file
        os.makedirs(self.preds_save_dir, exist_ok=True)
        with h5py.File(os.path.join(self.preds_save_dir, self.exp_name+'_predictions.h5'), 'w') as prediction_file:
            prediction_file.create_dataset('slide_id', data=id_set)
            prediction_file.create_dataset('probs', data=probs.cpu().numpy())
            prediction_file.create_dataset('labels', data=target.cpu().numpy()[:, 0])
            prediction_file.create_dataset('embeddings', data=slide_embed.cpu().numpy())
        
        for c in range(self.n_classes):
            count = self.test_count[c]["count"]
            correct = self.test_count[c]["correct"]
            if count == 0:
                acc = None
            else:
                acc = float(correct) / count
                print('class {}: acc {}, correct {}/{}'.format(c, acc, correct, count))
        
        auc = self.AUROC(probs, target.squeeze())
        metrics = self.test_metrics(max_probs.squeeze() , target.squeeze())
        metrics['auc'] = auc
        for keys, values in metrics.items():
            print(f'{keys} = {values}')
            metrics[keys] = values.cpu().numpy()

        result = pd.DataFrame([metrics])
        result.to_csv(self.log_path / f'result{self.fold}.csv')
        self.test_step_outputs.clear()

    def load_model(self):
        model_name = self.hparams.Model.name
        try:
            Model = getattr(sys.modules[__name__], model_name)
        except:
            raise ValueError('Invalid Module File Name or Invalid Class Name!')
        self.model = self.instancialize(Model)
        pass

    def instancialize(self, Model, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.hparams.
        """
        class_args = inspect.getargspec(Model.__init__).args[1:]
        inkeys = self.hparams.Model.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = getattr(self.hparams.Model, arg)
        args1.update(other_args)
        return Model(**args1)

    def configure_optimizers(self):
        optimizer = create_optimizer(self.optimizer, self.model)
        return [optimizer]