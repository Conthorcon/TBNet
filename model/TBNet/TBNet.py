import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log

from .dataloader import tokenize

from .Res2Net import res2net50_v1b_26w_4s


class ConvBNR(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False):
        super(ConvBNR, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size, stride=stride, padding=dilation, dilation=dilation, bias=bias),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class Conv1x1(nn.Module):
    def __init__(self, inplanes, planes):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, 1)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class EAM(nn.Module):
    def __init__(self):
        super(EAM, self).__init__()
        self.reduce1 = Conv1x1(256, 64)
        self.reduce4 = Conv1x1(2048, 256)
        self.block = nn.Sequential(
            ConvBNR(256 + 64, 256, 3),
            ConvBNR(256, 256, 3),
            nn.Conv2d(256, 1, 1))

    def forward(self, x4, x1):
        size = x1.size()[2:]
        x1 = self.reduce1(x1)
        x4 = self.reduce4(x4)
        x4 = F.interpolate(x4, size, mode='bilinear', align_corners=False)
        out = torch.cat((x4, x1), dim=1)
        out = self.block(out)

        return out


class EFM(nn.Module):
    def __init__(self, channel):
        super(EFM, self).__init__()
        t = int(abs((log(channel, 2) + 1) / 2))
        k = t if t % 2 else t + 1
        self.conv2d = ConvBNR(channel, channel, 3)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, c, att):
        if c.size() != att.size():
            att = F.interpolate(att, c.size()[2:], mode='bilinear', align_corners=False)
        x = c * att + c
        x = self.conv2d(x)
        wei = self.avg_pool(x)
        wei = self.conv1d(wei.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        wei = self.sigmoid(wei)
        x = x * wei

        return x



class BGNet(nn.Module):
    def __init__(self):
        super(BGNet, self).__init__()
        self.resnet = res2net50_v1b_26w_4s(pretrained=True)
        # if self.training:
        # self.initialize_weights()

        self.eam = EAM()

        self.efm1 = EFM(256)
        self.efm2 = EFM(512)
        self.efm3 = EFM(1024)
        self.efm4 = EFM(2048)

        self.reduce1 = Conv1x1(256, 64)
        self.reduce2 = Conv1x1(512, 128)
        self.reduce3 = Conv1x1(1024, 320)
        self.reduce4 = Conv1x1(2048, 512)


    def forward(self, x):
        x1, x2, x3, x4 = self.resnet(x)

        edge = self.eam(x4, x1)
        edge_att = torch.sigmoid(edge)

        x1a = self.efm1(x1, edge_att)
        x2a = self.efm2(x2, edge_att)
        x3a = self.efm3(x3, edge_att)
        x4a = self.efm4(x4, edge_att)

        x1r = self.reduce1(x1a)
        x2r = self.reduce2(x2a)
        x3r = self.reduce3(x3a)
        x4r = self.reduce4(x4a)


        oe = F.interpolate(edge_att, scale_factor=4, mode='bilinear', align_corners=False)

        return x1r, x2r, x3r, x4r, oe
    
import sys


import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

def cosine_similarity_loss(text_features, visual_features):
    """Cosine Similarity Loss"""
    text_features = F.normalize(text_features, p=2, dim=1)
    visual_features = F.normalize(visual_features, p=2, dim=1)
    cosine_sim = torch.sum(text_features * visual_features, dim=1)
    return 1 - cosine_sim.mean()


import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F

from  .lib import *

from .clip336 import ViTCLIP 
from .CGD import Network


class TBNet(nn.Module):
    """Cross-Modality Guided Network"""
    def __init__(self, clip_encoder=None, encoder=None, feature_levels=[64, 128, 320, 512], backbone=Network):
        super().__init__()
        self.clip = ViTCLIP()
        self.encoder = encoder
        self.feature_levels = feature_levels
        self.hidden_dim = 768

        self.mlp_blocks = nn.ModuleList([ConvMlp(1024, self.hidden_dim) for _ in range(4)])
        self.cross_attention = CrossAttentionBlock(self.hidden_dim, guide_dim=self.hidden_dim)

        self.structure_merge_deep = StructureEnhancementBlock(feature_levels[3])

        self.seg_head1 = nn.Conv2d(self.hidden_dim, 1, 1)
        self.seg_head2 = nn.Conv2d(feature_levels[3], 1, 1)

        self.refinement_head = nn.Sequential(
            LNConvAct(512, 512, 3, 1, 1, act_name="relu"),
            nn.Conv2d(512, 1, 3, 1, 1)
        )
        
        self.text_projection = ProjectionNetwork(input_dim=self.hidden_dim, proj_dim=512)
        self.visual_projection_mid = ProjectionNetwork(input_dim=self.hidden_dim, proj_dim=feature_levels[3])
        self.visual_projection_deep = ProjectionNetwork(input_dim=512, proj_dim=feature_levels[3])

        self.body_encoder = MultiLevelVisualCollaborationModule(self.hidden_dim)
        self.neck = FPN(in_channels=[self.hidden_dim]*3, out_channels=[256, 512, 1024])
        self.decoder = TransformerDecoder(num_layers=1, d_model=512)
        self.backbone = backbone

    def get_visual_features(self, image, text_embeddings):
        visual_feats = self.clip.get_visual_feats_bchw(image)
        visual_feats = [mlp(f) for mlp, f in zip(self.mlp_blocks, visual_feats)]

        fused_feats = self.neck(visual_feats[:-1], text_embeddings)

        return *visual_feats, fused_feats

    def pool_features(self, features, pooling='avg'):
        return torch.mean(features, dim=1) if pooling == 'avg' else torch.max(features, dim=1)[0]

    def forward_pass(self, image, image_aux, text_embeddings):
        vis_feats = self.get_visual_features(image_aux, text_embeddings)

        res1, res2, res3, res_deep, fused = vis_feats  # res[2, 768, 24, 24], fuse[2, 512, 24, 25]
        text_proj = self.text_projection(text_embeddings)

        b, c, h, w = fused.shape
        decoded = self.decoder(fused).view(b, c, h, w) # fm [2, 512, 24, 24]
        refined = self.refinement_head(decoded)

        res1 = self.cross_attention(res1 * refined, text_embeddings)

        body_features = self.body_encoder(res1, res3, res2) # fv [2, 768, 24, 24]
        fv = self.seg_head1(body_features)  # [2, 1, 24, 24]
        fm = self.seg_head2(decoded)  # [2, 1, 24, 24]
        segmentation_map = [fv, fm]
        
        # x1 [2, 64, 112, 112], x2 [2, 128, 56, 56], x3 [2, 320, 28, 28], x4 [2, 512, 14, 14]
        enc1, enc2, enc3, enc4, oe = self.encoder(image)

        vis_mid = F.interpolate(    # fv [2, 512, 14, 14]
            self.visual_projection_mid(body_features.view(b, -1, self.hidden_dim)).view(b, -1, h, w),
            size=enc4.shape[2:], mode='bilinear', align_corners=True
        )
        vis_deep = F.interpolate(   # fm [2, 512, 14, 14]
            self.visual_projection_deep(decoded.contiguous().view(b, -1, c)).view(b, -1, h, w),
            size=enc4.shape[2:], mode='bilinear', align_corners=True
        )

        merged_output = self.structure_merge_deep(enc4, [vis_mid, vis_deep]) # Gc
        final_segmentation = self.backbone(enc1, enc2, enc3, enc4, merged_output)

        consistency = cosine_similarity_loss(self.pool_features(fused.view(b, -1, c)), text_proj) * 0.2

        return segmentation_map, consistency, oe, final_segmentation

    def forward(self, image, image_aux=None, class_names=None):
        if self.training:
            class_embs = self.clip.get_text_embeddings(class_names)
            return self.forward_pass(image, image_aux, class_embs)
        else:
            class_prompt = f"A photo of a camouflaged Object"
            class_names = tokenize(class_prompt, 77, truncate=True)
            image_aux = F.interpolate(
                            image,
                            size=(336, 336),
                            mode='bilinear',
                            align_corners=False
                        )

            class_embs = self.clip.get_text_embeddings(class_names.cuda())
            return self.forward_pass(image, image_aux, class_embs)
