import pandas as pd
import re

test = pd.read_csv("v3_test_English.csv")
synth = pd.read_csv("v3_train_German_mixed_10pct.csv")

def norm(s):
    return str(s).replace("\\", "/")

def key_from_test_path(p):
    p = norm(p)
    m = re.search(r"voices/([^/]+)/English/([^/]+)/([^/]+)\.npy", p)
    if not m:
        return None
    identity, video, clip = m.groups()
    return (identity, video, clip)

def key_from_synth_path(p):
    p = norm(p)
    m = re.search(r"voices/([^/]+)/German/([^/]+)/00000\.npy", p)
    if not m:
        return None
    identity, synth_vid = m.groups()

    if "_" in synth_vid and re.search(r"_\d+$", synth_vid):
        video, clip = synth_vid.rsplit("_", 1)
    else:
        video, clip = synth_vid, "00000"

    return (identity, video, clip)

test_keys = set(
    k for k in test["ecappa_feats_path"].map(key_from_test_path)
    if k is not None
)

synth["source_key"] = synth["ecappa_feats_path"].map(key_from_synth_path)

clean_synth = synth[~synth["source_key"].isin(test_keys)].copy()
clean_synth = clean_synth.drop(columns=["source_key"])

clean_synth.to_csv("v3_train_German_mixed_10pct_clean.csv", index=False)

print("Original synthetic:", synth.shape)
print("Clean synthetic:", clean_synth.shape)
print("Removed rows:", len(synth) - len(clean_synth))
print("Clean identities:", clean_synth["identity"].nunique())