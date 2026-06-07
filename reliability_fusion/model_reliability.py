import torch
import torch.nn as nn


class ModalityEncoder(nn.Module):
    """
    Projects a pre-extracted modality embedding into a shared hidden space.
    Used for both ECAPA audio embeddings and FaceNet visual embeddings.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 512, dropout: float = 0.3):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FeatureWiseReliabilityMLP(nn.Module):
    """
    Feature-wise reliability scoring module.

    Given a modality feature f in R^hidden_dim, predicts a reliability
    score s in [0, 1]^hidden_dim.

    This adapts AV-RelScore to the embedding-level setting.
    """

    def __init__(self, hidden_dim: int = 512, dropout: float = 0.3):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        return self.net(f)


class ReliabilityFusionNet(nn.Module):
    """
    Embedding-level AV-RelScore-style reliability fusion model.

    Inputs:
        audio: ECAPA-TDNN embedding, shape [B, 192]
        face:  FaceNet embedding, shape [B, 512]

    Core equations:
        s_a = R_a(f_a)
        s_v = R_v(f_v)

        f_a_hat = f_a + f_a ⊙ s_a
        f_v_hat = f_v + f_v ⊙ s_v

    where ⊙ is the Hadamard product / element-wise multiplication.
    """

    def __init__(
        self,
        audio_dim: int,
        face_dim: int,
        num_classes: int,
        hidden_dim: int = 512,
        dropout: float = 0.3,
        face_dropout_p: float = 0.3,
    ):
        super().__init__()

        self.face_dropout_p = face_dropout_p

        self.audio_encoder = ModalityEncoder(
            input_dim=audio_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        self.face_encoder = ModalityEncoder(
            input_dim=face_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        self.audio_reliability = FeatureWiseReliabilityMLP(
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        self.face_reliability = FeatureWiseReliabilityMLP(
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def apply_face_dropout(self, face: torch.Tensor) -> torch.Tensor:
        """
        Randomly zeros the visual embedding for some samples during training.
        This simulates missing visual modality and prepares the model for P4/P6.
        """

        if not self.training:
            return face

        if self.face_dropout_p <= 0:
            return face

        batch_size = face.size(0)
        keep_mask = (
            torch.rand(batch_size, 1, device=face.device) > self.face_dropout_p
        ).float()

        return face * keep_mask

    def forward(
        self,
        audio: torch.Tensor,
        face: torch.Tensor,
        force_audio_only: bool = False,
        return_reliability: bool = False,
    ):
        """
        Args:
            audio: [B, audio_dim]
            face: [B, face_dim]
            force_audio_only:
                If True, sets face embedding to zero.
                Used for P4 and P6 evaluation.
            return_reliability:
                If True, returns logits and reliability diagnostics.
        """

        if force_audio_only:
            face = torch.zeros_like(face)

        face = self.apply_face_dropout(face)

        # Detect samples where visual modality is missing.
        # Shape: [B, 1]
        face_present = (face.abs().sum(dim=1, keepdim=True) > 0).float()

        # Encode modalities
        f_audio = self.audio_encoder(audio)
        f_face = self.face_encoder(face)

        # Feature-wise reliability scores in [0, 1]^hidden_dim
        s_audio = self.audio_reliability(f_audio)
        s_face = self.face_reliability(f_face)

        # AV-RelScore-style emphasis:
        # f_hat = f + f ⊙ s
        f_audio_hat = f_audio + f_audio * s_audio
        f_face_hat = f_face + f_face * s_face

        # Important: remove visual branch contribution when face is missing.
        # This avoids visual encoder bias leaking information from zero inputs.
        f_face_hat = f_face_hat * face_present
        s_face = s_face * face_present

        fused = torch.cat([f_audio_hat, f_face_hat], dim=1)
        logits = self.classifier(fused)

        if return_reliability:
            diagnostics = {
                "s_audio_mean": s_audio.mean(dim=1),
                "s_face_mean": s_face.mean(dim=1),
                "face_present": face_present.squeeze(1),
            }
            return logits, diagnostics

        return logits