import torchvision
import torch
import torch.nn as nn
import pretrainedmodels
import geffnet

from efficientnet_pytorch import EfficientNet

EFFNETB6_EMB_DIM = 2304
EFFNETB5_EMB_DIM = 2048
EFFNETB4_EMB_DIM = 1792
EFFNETB3_EMB_DIM = 1536
EFFNETB1_EMB_DIM = 1280

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class Effnet(nn.Module):
    def __init__(
        self,
        dropout_rate_first: float = 0.2,
        dropout_rate_second: float = 0.1,
        mlp_hidden_dim: int = 128, 
        out_channels: int = 1,
        device: str = 'cuda',
        model_type: str = 'effnet1',
        return_img_embed: bool = False,
        classifier_type: str = 'D_L_BN_S_D_L'
    ):
        super().__init__()
        self.return_img_embed = return_img_embed

        if model_type == 'effnet1':    
            self.base_model = geffnet.tf_efficientnet_b1_ns(pretrained=True)
            self.base_model.classifier = nn.Identity()
            nn_embed_size = EFFNETB1_EMB_DIM
        elif model_type == 'effnet3':
            self.base_model = geffnet.tf_efficientnet_b3_ns(pretrained=True)
            self.base_model.classifier = nn.Identity()
            nn_embed_size = EFFNETB3_EMB_DIM
        elif model_type == 'effnet4':
            self.base_model = geffnet.tf_efficientnet_b4_ns(pretrained=True)
            self.base_model.classifier = nn.Identity()
            nn_embed_size = EFFNETB4_EMB_DIM
        elif model_type == 'effnet5':
            self.base_model = geffnet.tf_efficientnet_b5_ns(pretrained=True)
            self.base_model.classifier = nn.Identity()
            nn_embed_size = EFFNETB5_EMB_DIM
        elif model_type == 'effnet6':
            self.base_model = geffnet.tf_efficientnet_b6_ns(pretrained=True)
            self.base_model.classifier = nn.Identity()
            nn_embed_size = EFFNETB6_EMB_DIM
        elif model_type == 'effnet3_imagenet':
            self.base_model = EfficientNet.from_pretrained('efficientnet-b3')
            self.base_model._dropout = nn.Identity()
            self.base_model._fc = nn.Identity()
            self.base_model._swish = nn.Identity()
            nn_embed_size = EFFNETB3_EMB_DIM
        else:
            raise ValueError(f'{model_type} is invalid model_type')

        if classifier_type == 'D_L_BN_S_D_L':
            self.classifier = nn.Sequential(
                nn.Dropout(dropout_rate_first),
                nn.Linear(nn_embed_size, mlp_hidden_dim),
                nn.BatchNorm1d(mlp_hidden_dim),
                Swish(),
                nn.Dropout(dropout_rate_second),
                nn.Linear(mlp_hidden_dim, out_channels)
            )
        elif classifier_type == 'D_L':
            self.classifier = nn.Sequential(
                nn.Dropout(dropout_rate_second),
                nn.Linear(nn_embed_size, out_channels)
            )
        elif classifier_type == 'D':
            self.classifier == nn.Linear(nn_embed_size, out_channels)
        elif classifier_type == 'D_L_ELU_MD_L':
            self.classifier_1 = nn.Sequential(
                nn.Dropout(dropout_rate_first),
                nn.Linear(nn_embed_size, mlp_hidden_dim),
                nn.ELU()
            )
            self.multiscale_dropout = nn.Dropout(0.5)
            self.classifier_2 = nn.Linear(mlp_hidden_dim, out_channels)
        else:
            raise ValueError(f'{classifier_type} is invalid classifier_type')

        self.classifier_type = classifier_type

        self.to(device)
    
    def forward(self, image):
        emb = self.base_model(image)
        if self.classifier_type == 'D_L_ELU_MD_L':
            emb = self.classifier_1(emb)
            logits = torch.mean(
                torch.stack(
                    [self.classifier_2(self.multiscale_dropout(emb)) for _ in range(5)],
                    dim=0,
                ),
                dim=0,
            )
        else:
            logits = self.classifier(emb)

        if self.return_img_embed:
            return logits, emb
        else:
            return logits
