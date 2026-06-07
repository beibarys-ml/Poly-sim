from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class PolySimFeatureDataset(Dataset):
    """
    Dataset for POLYSIM/FOP pre-extracted features.

    It reads CSV rows with:
        - ecappa_feats_path
        - facenet_feats_path
        - label

    and loads:
        - ECAPA-TDNN audio embedding: shape [192]
        - FaceNet visual embedding: shape [512]

    Paths in the CSV are relative paths such as:
        ./ecappafeats/v3\\voices\\id0001\\English\\...

    We resolve them relative to the external FOP root.
    """

    def __init__(
        self,
        csv_path: Union[str, Path],
        fop_root: Union[str, Path],
        audio_only: bool = False,
        use_wavlm: bool = False,
    ):
        self.csv_path = Path(csv_path)
        self.fop_root = Path(fop_root)
        self.audio_only = audio_only
        self.use_wavlm = use_wavlm

        self.df = pd.read_csv(self.csv_path)

        required_cols = ["label", "facenet_feats_path"]
        audio_col = "wavlmsv_feats_path" if use_wavlm else "ecappa_feats_path"
        required_cols.append(audio_col)

        missing = [c for c in required_cols if c not in self.df.columns]
        if missing:
            raise ValueError(
                f"Missing columns in {self.csv_path}: {missing}. "
                f"Available columns: {self.df.columns.tolist()}"
            )

        self.audio_col = audio_col

    def __len__(self):
        return len(self.df)

    def _resolve_feature_path(self, rel_path: str) -> Path:
        """
        Converts CSV relative paths into absolute paths under FOP_ROOT.

        Example:
            ./ecappafeats/v3\\voices\\id0001\\English\\...npy
        becomes:
            C:/Users/beiba/Desktop/baseline/polysim-main/FOP/ecappafeats/v3/voices/...
        """

        rel_path = str(rel_path)

        # Normalize Windows/backslash and remove leading ./ or .\
        rel_path = rel_path.replace("\\", "/")
        rel_path = rel_path.replace("./", "").replace(".\\", "")

        return self.fop_root / rel_path

    def _load_npy(self, path: Path) -> np.ndarray:
        if not path.exists():
            raise FileNotFoundError(f"Feature file not found: {path}")

        arr = np.load(path).astype(np.float32)

        # Ensure 1D vector.
        arr = arr.reshape(-1)

        return arr

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        audio_path = self._resolve_feature_path(row[self.audio_col])
        face_path = self._resolve_feature_path(row["facenet_feats_path"])

        audio = self._load_npy(audio_path)

        if self.audio_only:
            # Shape is fixed from your inspection: FaceNet = 512
            face = np.zeros(512, dtype=np.float32)
        else:
            face = self._load_npy(face_path)

        label = int(row["label"])

        return {
            "audio": torch.from_numpy(audio),
            "face": torch.from_numpy(face),
            "label": torch.tensor(label, dtype=torch.long),
            "identity": row.get("identity", ""),
        }