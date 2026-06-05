import pandas as pd
import re
from pathlib import Path

FOP_ROOT = r"C:\Users\beiba\Desktop\baseline-polysim\Polysim-main\FOP"

test_csv = Path(FOP_ROOT) / "feature_tracker" / "v3_test_English.csv"
mixed_csv = Path(FOP_ROOT) / "feature_tracker" / "v3_train_German_mixed_10pct.csv"

test = pd.read_csv(test_csv)
mixed = pd.read_csv(mixed_csv)

def norm(p):
    return str(p).replace("\\", "/")

def key_from_test(path):
    path = norm(path)
    m = re.search(r"voices/([^/]+)/English/([^/]+)/([^/]+)\.npy", path)
    if not m:
        return None
    identity, video_id, clip_id = m.groups()
    return (identity, video_id, clip_id)

def key_from_synthetic(path):
    path = norm(path)
    m = re.search(r"voices/([^/]+)/German/([^/]+)/00000\.npy", path)
    if not m:
        return None

    identity, synth_video_id = m.groups()

    # Example: h_vamljclHE_00000 -> video_id=h_vamljclHE, clip_id=00000
    if "_" in synth_video_id and re.search(r"_\d+$", synth_video_id):
        video_id, clip_id = synth_video_id.rsplit("_", 1)
    else:
        video_id, clip_id = synth_video_id, "00000"

    return (identity, video_id, clip_id)

test_keys = set(
    k for k in test["ecappa_feats_path"].map(key_from_test)
    if k is not None
)

# Only synthetic rows have ecappafeats_synthetic in the path
mixed_synth = mixed[mixed["ecappa_feats_path"].astype(str).str.contains("ecappafeats_synthetic", na=False)].copy()
mixed_synth["source_key"] = mixed_synth["ecappa_feats_path"].map(key_from_synthetic)

overlap = mixed_synth[mixed_synth["source_key"].isin(test_keys)]

print("Mixed rows:", len(mixed))
print("Synthetic rows in mixed:", len(mixed_synth))
print("Overlapping synthetic rows:", len(overlap))
print("Overlapping unique clips:", overlap["source_key"].nunique())

if len(overlap) > 0:
    print("\nExample overlaps:")
    print(overlap[["identity", "ecappa_feats_path", "source_key"]].head())
else:
    print("\nNo leakage detected.")