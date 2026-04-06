import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from config import ExperimentConfig
from model import FOP
from main import make_loader

def collect_embeddings(model, dataset, config):
    model.eval()

    face = torch.from_numpy(dataset.face_feats).float().to(config.device)
    audio = torch.from_numpy(dataset.audio_feats).float().to(config.device)
    labels = torch.from_numpy(dataset.labels).long().cpu().numpy()

    with torch.no_grad():
        fused, _, face_e, voice_e = model(face, audio)

    return (
        face_e.cpu().numpy(),
        voice_e.cpu().numpy(),
        fused.cpu().numpy(),
        labels
    )

def plot_face_voice_tsne(face_e, voice_e, labels, max_points=300):
    n = min(len(labels), max_points)

    face_e = face_e[:n]
    voice_e = voice_e[:n]
    labels = labels[:n]

    X = np.concatenate([face_e, voice_e], axis=0)
    y = np.concatenate([labels, labels], axis=0)
    mod = np.array(["face"] * n + ["voice"] * n)

    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init="pca")
    X_2d = tsne.fit_transform(X)

    face_2d = X_2d[:n]
    voice_2d = X_2d[n:]

    plt.figure(figsize=(10, 8))

    unique_labels = np.unique(labels)
    cmap = plt.cm.get_cmap("tab20", len(unique_labels))

    label_to_color = {lab: cmap(i) for i, lab in enumerate(unique_labels)}

    for i in range(n):
        c = label_to_color[labels[i]]

        plt.scatter(face_2d[i, 0], face_2d[i, 1], color=c, marker="o", s=40)
        plt.scatter(voice_2d[i, 0], voice_2d[i, 1], color=c, marker="^", s=40)

        plt.plot(
            [face_2d[i, 0], voice_2d[i, 0]],
            [face_2d[i, 1], voice_2d[i, 1]],
            color=c,
            alpha=0.25,
            linewidth=0.8
        )

    plt.title("t-SNE of projected face and voice embeddings")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()

def plot_fused_tsne(fused, labels, max_points=300):
    n = min(len(labels), max_points)
    fused = fused[:n]
    labels = labels[:n]

    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init="pca")
    Z = tsne.fit_transform(fused)

    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(labels)
    cmap = plt.cm.get_cmap("tab20", len(unique_labels))

    for i, lab in enumerate(unique_labels):
        idx = labels == lab
        plt.scatter(Z[idx, 0], Z[idx, 1], s=35, color=cmap(i), label=str(lab), alpha=0.8)

    plt.title("t-SNE of fused embeddings")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()

def paired_vs_impostor_distance(face_e, voice_e, labels):
    paired = np.linalg.norm(face_e - voice_e, axis=1)

    impostor = []
    for i in range(len(labels)):
        candidates = np.where(labels != labels[i])[0]
        j = np.random.choice(candidates)
        impostor.append(np.linalg.norm(face_e[i] - voice_e[j]))
    impostor = np.array(impostor)

    print(f"Mean paired distance    : {paired.mean():.4f}")
    print(f"Mean impostor distance  : {impostor.mean():.4f}")

if __name__ == "__main__":
    config = ExperimentConfig()
    config.device = "cuda" if torch.cuda.is_available() else "cpu"

    test_csv = f"{config.home_dir}/feature_tracker/{config.version}_test_{config.seen_lang}.csv"
    test_dataset, _ = make_loader(test_csv, config, shuffle=False)

    face_dim = test_dataset.face_feats.shape[1]
    audio_dim = test_dataset.audio_feats.shape[1]

    model = FOP(config=config, face_dim=face_dim, voice_dim=audio_dim).to(config.device)

    ckpt_path = "C:/users/beiba/Desktop/polysim-main/FOP/checkpoints/v3_German_alpha0.0_best.pt"
    ckpt = torch.load(ckpt_path, map_location=config.device)
    model.load_state_dict(ckpt["model_state"])

    face_e, voice_e, fused, labels = collect_embeddings(model, test_dataset, config)

    paired_vs_impostor_distance(face_e, voice_e, labels)
    plot_face_voice_tsne(face_e, voice_e, labels, max_points=250)
    plot_fused_tsne(fused, labels, max_points=250)