"""
Fine-tune a frozen/unfrozen echo backbone for a BINARY cardiomyopathy label.

Built on the proven per-clip structure from the coworker's
03_fine_tune_single_trait.py, adapted from regression -> classification:
  * MSELoss            -> BCEWithLogitsLoss (with pos_weight for imbalance)
  * MAE / Pearson      -> AUC / precision / recall / F1
  * numeric trait      -> 0/1 label
  * patient-level split kept (prevents clips of one patient leaking across splits)
  * patient-level aggregation added (clip probs averaged per patient at test)

Trains one target at a time. Each ROW of CLIP_CSV is one video clip with its
patient's binary label, produced by build_clip_labels.py.

Required columns in CLIP_CSV:
  * tensor_file_path : path to .pt tensor (shape 1x3x16x224x224)
  * PMBB_ID          : patient ID (used for patient-level split + aggregation)
  * label            : 0/1 target

Run on Databricks (GPU):
  python finetune_cm.py
"""

# ────────────────────────────────────────────────────────────────────────────
# SECTION 0 │ Imports & Config
# ────────────────────────────────────────────────────────────────────────────
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

# ========== EDIT THESE ======================================================
TARGET          = "HCM_PLP"      # naming only — which label CLIP_CSV holds
ENCODER         = "panecho"      # panecho (proven) | echoprime
CLIP_CSV        = Path("/Workspace/VermaLab/Sahil_EchoCV/clip_labels_HCM_PLP.csv")
CKPT_PATH       = Path(f"/Workspace/VermaLab/Sahil_EchoCV/finetune_{TARGET}_{ENCODER}/best.pth")
PRED_PATH       = Path(f"/Workspace/VermaLab/Sahil_EchoCV/finetune_{TARGET}_{ENCODER}/test_predictions.csv")

FREEZE_BACKBONE = True           # True = train head only (safer for few positives)
                                 # False = full fine-tune at low LR (coworker recipe)
BATCH_SIZE      = 64
NUM_EPOCHS      = 15
EARLY_STOP      = 3              # stop after this many epochs with no val-AUC gain
SEED            = 42

# EchoPrime encoder weights (only used if ENCODER == "echoprime")
ECHOPRIME_ENCODER_PT = "model_data/weights/echo_prime_encoder.pt"
# ============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.backends.cuda.matmul.allow_tf32 = True


# ────────────────────────────────────────────────────────────────────────────
# SECTION 1 │ Load backbone (feature extractor only)
# ────────────────────────────────────────────────────────────────────────────
def load_backbone():
    """Return a backbone that maps (B,3,16,224,224) -> (B, D) embeddings."""
    if ENCODER == "panecho":
        # backbone_only=True returns the feature extractor (outputs 768-dim)
        return torch.hub.load("CarDS-Yale/PanEcho", "PanEcho",
                              backbone_only=True, force_reload=False)
    elif ENCODER == "echoprime":
        # EchoPrime's video encoder (MViT-v2-s) from load_for_finetuning.py (512-dim)
        import torchvision
        ckpt = torch.load(ECHOPRIME_ENCODER_PT, map_location="cpu")
        bb = torchvision.models.video.mvit_v2_s()
        bb.head[-1] = nn.Linear(bb.head[-1].in_features, 512)
        bb.load_state_dict(ckpt)
        return bb
    else:
        raise ValueError(f"Unknown ENCODER: {ENCODER}")


# ────────────────────────────────────────────────────────────────────────────
# SECTION 2 │ Classifier head on top of the backbone
# ────────────────────────────────────────────────────────────────────────────
class ClassifierHead(nn.Module):
    def __init__(self, backbone, emb_dim, freeze):
        super().__init__()
        self.backbone = backbone
        self.dropout  = nn.Dropout(0.5)
        self.fc       = nn.Linear(emb_dim, 1)
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x):
        x = self.backbone(x)          # (B, D)
        x = self.dropout(x)
        return self.fc(x).squeeze(1)  # (B,) raw logits


# ────────────────────────────────────────────────────────────────────────────
# SECTION 3 │ Dataset — one .pt tensor (one clip) per row
# ────────────────────────────────────────────────────────────────────────────
class ClipDataset(Dataset):
    def __init__(self, tbl):
        self.tbl = tbl.reset_index(drop=True)

    def __len__(self):
        return len(self.tbl)

    def __getitem__(self, i):
        r = self.tbl.iloc[i]
        x = torch.load(r["tensor_file_path"], map_location="cpu").squeeze(0)  # (3,16,224,224)
        y = torch.tensor(r["label"], dtype=torch.float32)
        return x, y


def make_loader(tbl, shuffle):
    return DataLoader(ClipDataset(tbl), batch_size=BATCH_SIZE, shuffle=shuffle,
                      num_workers=8, pin_memory=torch.cuda.is_available())


# ────────────────────────────────────────────────────────────────────────────
# SECTION 4 │ Main
# ────────────────────────────────────────────────────────────────────────────
def main():
    # ── Load clip table ──────────────────────────────────────────────────────
    df = pd.read_csv(CLIP_CSV)
    required = {"tensor_file_path", "PMBB_ID", "label"}
    miss = required - set(df.columns)
    if miss:
        raise ValueError(f"CLIP_CSV missing {miss}")

    # ── Patient-level split (70/20/10) so no patient spans two splits ────────
    ids = df["PMBB_ID"].unique().tolist()
    random.shuffle(ids)
    n = len(ids)
    train_ids = set(ids[: math.floor(0.7 * n)])
    val_ids   = set(ids[math.floor(0.7 * n): math.floor(0.9 * n)])
    test_ids  = set(ids[math.floor(0.9 * n):])

    df["split"] = df["PMBB_ID"].map(
        lambda p: "train" if p in train_ids else "val" if p in val_ids else "test")
    print("Clips per split:\n", df["split"].value_counts())
    print("Patients per split:",
          {k: df.loc[df.split == k, "PMBB_ID"].nunique() for k in ["train", "val", "test"]})

    train_loader = make_loader(df[df.split == "train"], shuffle=True)
    val_loader   = make_loader(df[df.split == "val"],   shuffle=False)
    test_tbl     = df[df.split == "test"].reset_index(drop=True)
    test_loader  = make_loader(test_tbl, shuffle=False)

    # ── Build model ──────────────────────────────────────────────────────────
    print(f"Loading {ENCODER} backbone (freeze={FREEZE_BACKBONE}) ...")
    backbone = load_backbone()
    # infer embedding dim with a dummy forward (768 panecho / 512 echoprime)
    backbone.eval()
    with torch.no_grad():
        emb_dim = backbone(torch.zeros(1, 3, 16, 224, 224)).shape[1]
    print(f"Embedding dim: {emb_dim}")

    model = ClassifierHead(backbone, emb_dim, FREEZE_BACKBONE).to(DEVICE)

    # ── Class imbalance: weight positives in the loss ────────────────────────
    n_pos = int((df[df.split == "train"].label == 1).sum())
    n_neg = int((df[df.split == "train"].label == 0).sum())
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=DEVICE)
    print(f"Train clips — pos:{n_pos} neg:{n_neg} | pos_weight={pos_weight.item():.2f}")
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # ── Optimizer — head only if frozen, else two-LR group like coworker ─────
    if FREEZE_BACKBONE:
        opt = optim.Adam(model.fc.parameters(), lr=1e-3)
    else:
        opt = optim.Adam([
            {"params": model.backbone.parameters(), "lr": 1.6e-6},
            {"params": model.fc.parameters(),       "lr": 1.6e-5},
        ])
    sched = ReduceLROnPlateau(opt, "max", patience=2, factor=0.1)  # max: higher AUC better
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    # ── Train loop ───────────────────────────────────────────────────────────
    best_auc, no_improve = 0.0, 0
    for epoch in range(NUM_EPOCHS):
        model.train()
        for vids, tgts in tqdm(train_loader, desc=f"Ep{epoch+1} [train]"):
            vids, tgts = vids.to(DEVICE, non_blocking=True), tgts.to(DEVICE, non_blocking=True)
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits = model(vids)
                loss = crit(logits, tgts)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

        # ── Validation (clip-level AUC) ──────────────────────────────────────
        val_prob, val_true = predict(model, val_loader)
        val_auc = safe_auc(val_true, val_prob)
        sched.step(val_auc if not math.isnan(val_auc) else 0.0)
        print(f"Ep{epoch+1:2d}: val AUC {val_auc:.3f} | LR {opt.param_groups[0]['lr']:.2e}")

        # always keep at least one checkpoint (so test never crashes if val AUC is
        # nan — e.g. a single-class val fold); update it whenever val AUC improves
        improved = (not math.isnan(val_auc)) and (val_auc > best_auc)
        if improved or not CKPT_PATH.exists():
            if improved:
                best_auc, no_improve = val_auc, 0
            CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"epoch": epoch + 1, "model_state_dict": model.state_dict(),
                        "val_auc": best_auc}, CKPT_PATH)
            print(f"  saved (best val AUC {best_auc:.3f})")
        else:
            no_improve += 1
            if no_improve >= EARLY_STOP:
                print(f"Early stop: no val-AUC gain for {EARLY_STOP} epochs")
                break

    # ── Test with best checkpoint — report BOTH clip- and patient-level ──────
    print("\nEvaluating best checkpoint on TEST ...")
    model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE)["model_state_dict"])
    test_prob, test_true = predict(model, test_loader)

    # clip-level
    report("TEST (clip-level)", test_true, test_prob)

    # patient-level: average clip probabilities per patient (the clinical unit)
    test_tbl = test_tbl.copy()
    test_tbl["prob"] = test_prob
    pat = test_tbl.groupby("PMBB_ID").agg(prob=("prob", "mean"), label=("label", "first"))
    report("TEST (patient-level)", pat["label"].values, pat["prob"].values)

    PRED_PATH.parent.mkdir(parents=True, exist_ok=True)
    pat.reset_index().to_csv(PRED_PATH, index=False)
    print(f"\nSaved patient-level predictions to {PRED_PATH}")


# ────────────────────────────────────────────────────────────────────────────
# SECTION 5 │ Helpers
# ────────────────────────────────────────────────────────────────────────────
def predict(model, loader):
    model.eval()
    probs, trues = [], []
    with torch.no_grad():
        for vids, tgts in loader:
            vids = vids.to(DEVICE, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits = model(vids)
            probs.append(torch.sigmoid(logits).float().cpu())
            trues.append(tgts)
    return torch.cat(probs).numpy(), torch.cat(trues).numpy()


def safe_auc(y_true, y_prob):
    try:
        return roc_auc_score(y_true, y_prob)
    except ValueError:
        return float("nan")


def report(tag, y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)
    print(f"  {tag}: "
          f"AUC {safe_auc(y_true, y_prob):.3f} | "
          f"P {precision_score(y_true, y_pred, zero_division=0):.3f} | "
          f"R {recall_score(y_true, y_pred, zero_division=0):.3f} | "
          f"F1 {f1_score(y_true, y_pred, zero_division=0):.3f} | "
          f"N={len(y_true)}")


if __name__ == "__main__":
    main()
