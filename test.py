"""
Stylized-ImageNet (SIN) style augmentation for resume-image classification
with a selection / representation bias audit.

Data layout
-----------
data/
  meta.csv            columns: path,label,group,split
                      path  : image path relative to data/
                      label : 1 = shortlisted, 0 = rejected
                      group : protected attribute (e.g. gender) - used ONLY for auditing
                      split : train / val / test
  styles/             any folder of paintings / textures (jpg/png) used as style sources

Usage
-----
pip install torch torchvision pandas numpy pillow scikit-learn
python sin_resume_bias.py --data data --epochs 10 --mode both

Optional: --decoder decoder.pth  (pretrained AdaIN decoder from naoto0804/pytorch-AdaIN)
          for true AdaIN stylization. Without it, a fast pixel-space AdaIN
          + texture-noise fallback is used.
"""
import argparse, glob, random
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms as T

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


# ----------------------------------------------------------------------------
# 1. Style transfer (SIN-style augmentation)
# ----------------------------------------------------------------------------
def calc_mean_std(feat, eps=1e-5):
    n, c = feat.shape[:2]
    v = feat.view(n, c, -1)
    return v.mean(2).view(n, c, 1, 1), (v.var(2) + eps).sqrt().view(n, c, 1, 1)


def adain(content, style):
    cm, cs = calc_mean_std(content)
    sm, ss = calc_mean_std(style)
    return (content - cm) / cs * ss + sm


def build_decoder():
    p = lambda: nn.ReflectionPad2d((1, 1, 1, 1))
    return nn.Sequential(
        p(), nn.Conv2d(512, 256, 3), nn.ReLU(), nn.Upsample(scale_factor=2),
        p(), nn.Conv2d(256, 256, 3), nn.ReLU(),
        p(), nn.Conv2d(256, 256, 3), nn.ReLU(),
        p(), nn.Conv2d(256, 256, 3), nn.ReLU(),
        p(), nn.Conv2d(256, 128, 3), nn.ReLU(), nn.Upsample(scale_factor=2),
        p(), nn.Conv2d(128, 128, 3), nn.ReLU(),
        p(), nn.Conv2d(128, 64, 3), nn.ReLU(), nn.Upsample(scale_factor=2),
        p(), nn.Conv2d(64, 64, 3), nn.ReLU(),
        p(), nn.Conv2d(64, 3, 3))


class Stylizer:
    """Applies random style transfer to a batch of ImageNet-normalised tensors."""

    def __init__(self, style_dir, decoder_path=None, size=224):
        self.files = sorted(sum([glob.glob(f"{style_dir}/*.{e}") for e in ("jpg", "jpeg", "png")], []))
        assert self.files, f"No style images found in {style_dir}"
        self.tf = T.Compose([T.Resize(size), T.CenterCrop(size), T.ToTensor(), T.Normalize(MEAN, STD)])
        self.decoder = None
        if decoder_path:
            vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features[:21]
            self.encoder = vgg.eval().to(DEVICE)
            self.decoder = build_decoder().eval().to(DEVICE)
            self.decoder.load_state_dict(torch.load(decoder_path, map_location=DEVICE))
            for m in (self.encoder, self.decoder):
                for p_ in m.parameters():
                    p_.requires_grad = False

    def _styles(self, n):
        imgs = [self.tf(Image.open(random.choice(self.files)).convert("RGB")) for _ in range(n)]
        return torch.stack(imgs).to(DEVICE)

    @torch.no_grad()
    def __call__(self, x, alpha_range=(0.4, 1.0)):
        s = self._styles(x.size(0))
        alpha = random.uniform(*alpha_range)
        if self.decoder is not None:                       # true AdaIN
            fc, fs = self.encoder(x), self.encoder(s)
            out = self.decoder(alpha * adain(fc, fs) + (1 - alpha) * fc)
            return F.interpolate(out, size=x.shape[-2:])
        # fallback: pixel-space AdaIN (colour / contrast statistics) + texture noise
        out = alpha * adain(x, s) + (1 - alpha) * x
        out = out + 0.15 * torch.randn_like(out) * random.random()
        return out


# ----------------------------------------------------------------------------
# 2. Dataset
# ----------------------------------------------------------------------------
class ResumeDS(Dataset):
    def __init__(self, root, df, train):
        self.root, self.df = root, df.reset_index(drop=True)
        aug = [T.RandomResizedCrop(224, scale=(0.8, 1.0)), T.RandomRotation(2)] if train else \
              [T.Resize((224, 224))]
        self.tf = T.Compose(aug + [T.ToTensor(), T.Normalize(MEAN, STD)])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(f"{self.root}/{r.path}").convert("RGB")
        return self.tf(img), int(r.label), str(r.group)


# ----------------------------------------------------------------------------
# 3. Model / training
# ----------------------------------------------------------------------------
def make_model():
    m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    m.fc = nn.Linear(m.fc.in_features, 2)
    return m.to(DEVICE)


def train(model, loader, val_loader, epochs, stylizer=None, p_style=0.5, lr=1e-4):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for ep in range(epochs):
        model.train()
        tot = 0
        for x, y, _ in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            if stylizer is not None:                       # SIN: stylize a fraction of the batch
                k = int(p_style * x.size(0))
                if k:
                    x = torch.cat([stylizer(x[:k]), x[k:]])
            loss = F.cross_entropy(model(x), y, label_smoothing=0.05)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * x.size(0)
        sched.step()
        acc = evaluate(model, val_loader)["acc"]
        print(f"  epoch {ep+1}/{epochs}  loss={tot/len(loader.dataset):.4f}  val_acc={acc:.4f}")
    return model


# ----------------------------------------------------------------------------
# 4. Evaluation + bias audit
# ----------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, stylizer=None):
    model.eval()
    P, Y, G = [], [], []
    for x, y, g in loader:
        x = x.to(DEVICE)
        if stylizer is not None:
            x = stylizer(x, alpha_range=(1.0, 1.0))        # full stylization = texture-shift test
        P += model(x).argmax(1).cpu().tolist()
        Y += y.tolist(); G += list(g)
    P, Y, G = np.array(P), np.array(Y), np.array(G)
    out = {"acc": (P == Y).mean(), "pred": P, "y": Y, "g": G}
    return out


def fairness_report(res, name):
    P, Y, G = res["pred"], res["y"], res["g"]
    sel, tpr = {}, {}
    for g in np.unique(G):
        m = G == g
        sel[g] = P[m].mean()                               # selection rate
        pos = m & (Y == 1)
        tpr[g] = P[pos].mean() if pos.any() else np.nan    # true-positive rate
    sr = list(sel.values())
    dpd = max(sr) - min(sr)                                # demographic parity difference
    dir_ = min(sr) / max(sr) if max(sr) > 0 else np.nan    # disparate impact ratio (80% rule)
    eod = np.nanmax(list(tpr.values())) - np.nanmin(list(tpr.values()))  # equal opportunity diff
    print(f"\n[{name}] accuracy={res['acc']:.4f}")
    print(f"  selection rate per group : { {k: round(v,3) for k,v in sel.items()} }")
    print(f"  demographic parity diff  : {dpd:.4f}   (lower is fairer)")
    print(f"  disparate impact ratio   : {dir_:.4f}   (>= 0.80 passes the four-fifths rule)")
    print(f"  equal opportunity diff   : {eod:.4f}   (lower is fairer)")
    return dict(acc=res["acc"], dpd=dpd, dir=dir_, eod=eod)


def label_group_balance(df):
    print("\nRepresentation check (train split):")
    print(pd.crosstab(df.group, df.label, normalize="index").round(3))
    print(df.group.value_counts(normalize=True).round(3))


# ----------------------------------------------------------------------------
# 5. Main
# ----------------------------------------------------------------------------
def make_loaders(root, df, bs):
    tr, va, te = (df[df.split == s] for s in ("train", "val", "test"))
    label_group_balance(tr)
    # re-weight sampling so every (group, label) cell is equally represented in training
    cell = tr.group.astype(str) + "_" + tr.label.astype(str)
    w = 1.0 / cell.map(cell.value_counts())
    sampler = torch.utils.data.WeightedRandomSampler(w.values, len(tr), replacement=True)
    mk = lambda d, t, **kw: DataLoader(ResumeDS(root, d, t), batch_size=bs, num_workers=4, **kw)
    return mk(tr, True, sampler=sampler), mk(va, False), mk(te, False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--decoder", default=None)
    ap.add_argument("--mode", choices=["baseline", "sin", "both"], default="both")
    a = ap.parse_args()

    df = pd.read_csv(f"{a.data}/meta.csv")
    tr_l, va_l, te_l = make_loaders(a.data, df, a.bs)
    stylizer = Stylizer(f"{a.data}/styles", a.decoder)
    results = {}

    if a.mode in ("baseline", "both"):
        print("\n=== Baseline (original images only) ===")
        m = train(make_model(), tr_l, va_l, a.epochs)
        results["baseline"] = fairness_report(evaluate(m, te_l), "baseline / clean test")
        fairness_report(evaluate(m, te_l, stylizer), "baseline / stylized test (style shift)")

    if a.mode in ("sin", "both"):
        print("\n=== SIN-augmented (stylized + original) ===")
        m = train(make_model(), tr_l, va_l, a.epochs, stylizer=stylizer, p_style=0.5)
        results["sin"] = fairness_report(evaluate(m, te_l), "SIN / clean test")
        fairness_report(evaluate(m, te_l, stylizer), "SIN / stylized test (style shift)")
        torch.save(m.state_dict(), "resume_sin_resnet50.pth")

    if len(results) == 2:
        print("\n=== Summary (clean test) ===")
        print(pd.DataFrame(results).T.round(4))


if __name__ == "__main__":
    main()
