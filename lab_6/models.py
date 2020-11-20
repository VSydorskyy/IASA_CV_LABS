from typing import Optional, Mapping, Any

import torchvision
import torch
import torch.nn as nn
import pretrainedmodels
import geffnet

from efficientnet_pytorch import EfficientNet

from custom_nn import CustomCNN

EFFNETB6_EMB_DIM = 2304
EFFNETB5_EMB_DIM = 2048
EFFNETB4_EMB_DIM = 1792
EFFNETB3_EMB_DIM = 1536
EFFNETB1_EMB_DIM = 1280


class CNNModel(nn.Module):
    def __init__(
        self,
        dropout_rate_first: float = 0.2,
        dropout_rate_second: float = 0.1,
        mlp_hidden_dim: int = 128, 
        n_mlp_layers: int = 1,
        out_channels: int = 1,
        device: str = 'cuda',
        model_type: str = 'effnet',
        pretrained: bool = True,
        custom_config: Optional[Mapping[str,Any]] = None
    ):
        super().__init__()

        if model_type == 'vgg16':   
            pretrained = 'imagenet' if pretrained else None 
            self.base_model = pretrainedmodels.vgg16(pretrained=pretrained)
            self.base_model.linear0 = nn.Identity()
            self.base_model.relu0 = nn.Identity()
            self.base_model.dropout0 = nn.Identity()
            self.base_model.linear1 = nn.Identity()
            self.base_model.relu1 = nn.Identity()
            self.base_model.dropout1 = nn.Identity()
            self.base_model.last_linear = nn.Identity()
            nn_embed_size = 8192
        elif model_type == 'effnet':
            self.base_model = geffnet.tf_efficientnet_b1_ns(pretrained=pretrained)
            self.base_model.classifier = nn.Identity()
            nn_embed_size = 1280
        elif model_type == 'resnet50':
            pretrained = 'imagenet' if pretrained else None 
            self.base_model = pretrainedmodels.resnet50(pretrained=pretrained)
            self.base_model.last_linear = nn.Identity()
            nn_embed_size = 2048
        elif model_type == 'custom':
            custom_config = {} if custom_config is None else custom_config
            self.base_model = CustomCNN(**custom_config)
            nn_embed_size = 64
        else:
            raise ValueError(f'{model_type} is invalid model_type')

        self.emb_drop = nn.Dropout(dropout_rate_first)

        self.mlp_layres = []
        for i in range(n_mlp_layers):
            if i == 0:
                in_mlp_dim = nn_embed_size
            else:
                in_mlp_dim = mlp_hidden_dim
            self.mlp_layres.append(nn.Sequential(
                nn.Linear(in_mlp_dim, mlp_hidden_dim),
                nn.ELU(),
                nn.Dropout(dropout_rate_second)
            ))
        self.mlp_layres = nn.ModuleList(self.mlp_layres)

        self.classifier = nn.Linear(mlp_hidden_dim, out_channels)
        self.to(device)
    
    def forward(self, image):
        emb = self.base_model(image)
        emb = self.emb_drop(emb)
        for mlp_layer in self.mlp_layres:
            emb = mlp_layer(emb)
        logits = self.classifier(emb)
        return logits
