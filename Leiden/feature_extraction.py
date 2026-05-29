import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    CLIPProcessor,
    CLIPVisionModelWithProjection,
)
from torchvision import transforms
from sklearn.preprocessing import normalize
import scipy.linalg

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Config
CLIP_MODEL_NAME  = "openai/clip-vit-large-patch14"   
BATCH_SIZE       = 16
SAVE_DIR         = "."                                 
ZCA_EPSILON       = 1e-5

ds = load_dataset("taesiri/imagenet-hard", split="validation")
print(f"      Dataset size: {len(ds)} samples")

# Model
clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
clip_model     = CLIPVisionModelWithProjection.from_pretrained(CLIP_MODEL_NAME).to(device)
clip_model.eval()

five_crop_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.FiveCrop(224),
])
hflip = transforms.RandomHorizontalFlip(p=1.0)

def extract_batch(batch):
    all_crops = []
    n_imgs = len(batch["image"])

    for img in batch["image"]:
        img_rgb = img.convert("RGB")
        crops = five_crop_transform(img_rgb)       
        flipped = [hflip(c) for c in crops]         
        all_crops.extend(list(crops) + flipped)    

    inputs = clip_processor(
        images=all_crops,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        emb = clip_model(**inputs).image_embeds

    emb = emb.float()
    emb = emb.reshape(n_imgs, 10, -1).mean(dim=1)  

    return {"embedding": emb.cpu().numpy()}

ds = ds.map(
    extract_batch,
    batched=True,
    batch_size=BATCH_SIZE,
    remove_columns=["image"],
    desc="Extracting embeddings",
)

X_raw = np.vstack(ds["embedding"]).astype(np.float32)  
print(f"      Raw feature matrix : {X_raw.shape}")

labels_raw = ds["label"]
y_single   = np.array([lbl[0] for lbl in labels_raw]) 

X_norm = normalize(X_raw, norm="l2")                   

def zca_whiten(X, eps=ZCA_EPSILON):
    """
    ZCA (Mahalanobis) whitening:
      1. Zero-centre the data
      2. Compute covariance matrix
      3. Eigendecompose: Cov = V Λ Vᵀ
      4. W = V Λ^(-1/2) Vᵀ  (whitening matrix)
      5. Apply: X_white = (X - mean) @ W
    This decorrelates all dimensions AND standardises variance,
    improving cosine-similarity quality in downstream kNN.
    ZCA (vs PCA-whiten) is preferred here because it minimally
    rotates the data — original feature directions are preserved.
    """
    mean   = X.mean(axis=0)
    Xc     = X - mean                          

    cov    = (Xc.T @ Xc) / (len(X) - 1)       
    eigvals, eigvecs = scipy.linalg.eigh(cov)  

    eigvals = np.maximum(eigvals, 0)

    scale   = 1.0 / np.sqrt(eigvals + eps)
    W       = (eigvecs * scale) @ eigvecs.T    

    X_white = Xc @ W
    return X_white.astype(np.float32), mean, W  

X_white, _, _ = zca_whiten(X_norm)

X_final = normalize(X_white, norm="l2").astype(np.float32)

print(f"      Final feature shape : {X_final.shape}")

norms   = np.linalg.norm(X_final, axis=1)
print(f"\n── Sanity checks ──────────────────────────────────────")
print(f"   L2 norm  mean={norms.mean():.4f}  std={norms.std():.6f}  (should be ≈1.0 / 0.0)")
print(f"   Feature  mean={X_final.mean():.4f}  std={X_final.std():.4f}")
print(f"   NaN count: {np.isnan(X_final).sum()}")
print(f"   Inf count: {np.isinf(X_final).sum()}")

labels_raw = ds["label"]
y_single = np.array([lbl[0] for lbl in labels_raw])

np.save("features.npy", X_final)
np.save("labels.npy", y_single)