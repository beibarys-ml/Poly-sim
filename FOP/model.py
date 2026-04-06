import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def fc_block(in_dim, out_dim):
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.BatchNorm1d(out_dim),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5)
    )

class EmbedBranch(nn.Module):
    def __init__(self, feat_dim, emb_dim):
        super().__init__()
        self.fc = fc_block(feat_dim, emb_dim)

    def forward(self, x):
        x = self.fc(x)
        return F.normalize(x, dim=1)

class LinearFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Parameter(torch.rand(1))
        self.w2 = nn.Parameter(torch.rand(1))

    def forward(self, face, voice):
        return self.w1 * face + self.w2 * voice

class FOP(nn.Module):
    def __init__(self, config, face_dim, voice_dim):
        super().__init__()
        self.face_branch = EmbedBranch(face_dim, config.embedding_dim)
        self.voice_branch = EmbedBranch(voice_dim, config.embedding_dim)

        self.fusion = LinearFusion()
        self.classifier = nn.Linear(config.embedding_dim, config.resolved_num_classes)

import torch
import torch.nn as nn
import torch.nn.functional as F

def fc_block(in_dim, out_dim):
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.BatchNorm1d(out_dim),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5)
    )

class SBNet(nn.Module):
    def __init__(self, config, face_dim, voice_dim):
        super().__init__()
        self.config = config
        
        # 1. Unified Shared Branch (Replaces face_branch and voice_branch)
        # Face is 512d, Voice will be padded to 512d. Both project to 128d.
        self.shared_fc = fc_block(face_dim, config.embedding_dim)

        # 2. Classifier for final evaluation 
        self.classifier = nn.Linear(config.embedding_dim, config.resolved_num_classes)

    def forward(self, face, voice):
        # --- A. Padding ---
        # Pad the voice tensor from 192 to 512 to match face dimensionality
        pad_size = face.size(1) - voice.size(1)
        if pad_size > 0:
            voice_padded = F.pad(voice, (0, pad_size), "constant", 0)
        else:
            voice_padded = voice

        # --- B. Shared Representations (Sequential Processing) ---
        # Process face through unified branch
        face_e = self.shared_fc(face)
        face_e = F.normalize(face_e, dim=1)

        # Process padded voice through the SAME unified branch
        voice_e = self.shared_fc(voice_padded)
        voice_e = F.normalize(voice_e, dim=1)

        # Obtain classifier logits
        face_logits = self.classifier(face_e)
        voice_logits = self.classifier(voice_e)

        # --- C. Late Fusion & Missing Modality Logic ---
        # Identify which modalities are present (not all zeros)
        face_present = (face.norm(p=2, dim=1, keepdim=True) > 0).float()
        voice_present = (voice.norm(p=2, dim=1, keepdim=True) > 0).float()

        # Count available modalities (clamp to 1 to avoid division by zero)
        total_present = torch.clamp(face_present + voice_present, min=1.0)

        # Apply late fusion (average) if both present, or passthrough if only one is present
        logits = (face_logits * face_present + voice_logits * voice_present) / total_present
        
        # Create a fused embedding (for t-SNE plotting compatibility)
        fused = (face_e * face_present + voice_e * voice_present) / total_present

        return fused, logits, face_e, voice_e

# Retain backward compatibility so you don't have to rename FOP everywhere immediately
FOP = SBNet
