import os, io, traceback, time
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import cv2

import cloudinary
import cloudinary.uploader

from dotenv import load_dotenv
load_dotenv()

# ===== CONFIG =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pth")  # hoặc model.pth.tar
USE_LOGITS_HEAD = False        # True nếu checkpoint KHÔNG có Sigmoid trong head
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = (DEVICE.type == "cuda")
TOPK = 5
ALPHA = 0.45                   # alpha overlay

CLASS_NAMES = [
    'Atelectasis','Cardiomegaly','Effusion','Infiltration','Mass','Nodule','Pneumonia',
    'Pneumothorax','Consolidation','Edema','Emphysema','Fibrosis','Pleural_Thickening','Hernia'
]
N_CLASSES = 14

# ===== CLOUDINARY CONFIG =====
if os.getenv("CLOUDINARY_URL"):
    cloudinary.config(secure=True)
else:
    cloudinary.config(
        cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
        api_key=os.getenv("CLOUDINARY_API_KEY"),
        api_secret=os.getenv("CLOUDINARY_API_SECRET"),
        secure=True,
    )

# ===== MODEL =====
class DenseNet121_Sigmoid(nn.Module):
    def __init__(self, n_classes=N_CLASSES, imagenet_weights=True):
        super().__init__()
        try:
            weights = torchvision.models.DenseNet121_Weights.IMAGENET1K_V1 if imagenet_weights else None
            net = torchvision.models.densenet121(weights=weights)
        except Exception:
            net = torchvision.models.densenet121(pretrained=imagenet_weights)
        in_feats = net.classifier.in_features
        net.classifier = nn.Sequential(nn.Linear(in_feats, n_classes), nn.Sigmoid())
        self.net = net

    def forward(self, x):
        return self.net(x)

class DenseNet121_Logits(nn.Module):
    def __init__(self, n_classes=N_CLASSES, imagenet_weights=True):
        super().__init__()
        try:
            weights = torchvision.models.DenseNet121_Weights.IMAGENET1K_V1 if imagenet_weights else None
            net = torchvision.models.densenet121(weights=weights)
        except Exception:
            net = torchvision.models.densenet121(pretrained=imagenet_weights)
        in_feats = net.classifier.in_features
        net.classifier = nn.Linear(in_feats, n_classes)
        self.net = net

    def forward(self, x):
        return self.net(x)

def _remap_state_keys(state: dict) -> dict:
    # map 'module.' → '' và 'densenet121.' → 'net.' (phù hợp lớp ở trên)
    new_state = {}
    for k, v in state.items():
        if k.startswith("module."):
            k = k[7:]
        if k.startswith("densenet121."):
            k = "net." + k[12:]
        new_state[k] = v
    return new_state

def load_checkpoint_into(model: nn.Module, ckpt_path: str, device: torch.device):
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    # nhận .pth (state_dict thuần) hoặc dict có 'state_dict'
    if isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        state = ckpt
    else:
        state = (ckpt.get("state_dict") or ckpt.get("model_state_dict") or ckpt)
    state = _remap_state_keys(state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[WARN] Missing keys: {missing[:6]}{'...' if len(missing)>6 else ''}")
    if unexpected:
        print(f"[WARN] Unexpected keys: {unexpected[:6]}{'...' if len(unexpected)>6 else ''}")

# ===== PREPROCESS (match training/eval) =====
normalize = transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])

# ===== Grad-CAM (center aligned) =====
def gradcam_overlay_center_aligned(
    model,
    x_norm_224: torch.Tensor,  # [1,3,224,224]
    orig_img: Image.Image,     # PIL RGB
    class_idx: int,
    use_logits_head: bool,
    alpha: float = ALPHA
):
    feats = model.net.features(x_norm_224)             # [1,C,h,w]
    acts  = F.relu(feats, inplace=False)
    gap   = F.adaptive_avg_pool2d(acts, (1,1))
    vec   = torch.flatten(gap, 1)
    out   = model.net.classifier(vec)                  # [1,14] (logits hoặc prob)
    score = out[0, class_idx]
    if not use_logits_head:
        p = score.clamp(1e-6, 1-1e-6)
        score = torch.log(p/(1-p))

    grads = torch.autograd.grad(
        score, acts,
        grad_outputs=torch.ones_like(score),
        retain_graph=False,
        create_graph=False
    )[0]     # [1,C,h,w]
    w   = grads.mean(dim=(2,3), keepdim=True)                                   # [1,C,1,1]
    cam = F.relu((w * acts).sum(dim=1, keepdim=False))[0]                       # [h,w]
    cam = (cam - cam.min()) / (cam.max() + 1e-8)

    cam_224 = cv2.resize(cam.detach().cpu().numpy(), (224, 224), interpolation=cv2.INTER_LINEAR)

    H0, W0 = orig_img.size[1], orig_img.size[0]     # (height, width)
    scale = 256.0 / min(H0, W0)
    Wr, Hr = int(round(W0 * scale)), int(round(H0 * scale))
    img_resized = np.array(orig_img.resize((Wr, Hr), Image.BILINEAR))

    y0 = (Hr - 224) // 2; x0 = (Wr - 224) // 2
    y1, x1 = y0 + 224, x0 + 224

    cam_full = np.zeros((Hr, Wr), dtype=np.float32)
    cam_full[y0:y1, x0:x1] = cam_224

    cam_full = cv2.GaussianBlur(cam_full, (0,0), 1.0)

    cam_uint8 = np.uint8(np.clip(cam_full, 0, 1) * 255)
    heat_bgr  = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heat      = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)

    overlay_r = (alpha * heat + (1 - alpha) * img_resized).astype("uint8")
    overlay   = cv2.resize(overlay_r, (W0, H0), interpolation=cv2.INTER_LINEAR)
    return Image.fromarray(overlay)

# ===== HELPER: inference + Grad-CAM =====
def run_inference_with_gradcam(pil_img: Image.Image):
    """
    Trả về:
      - best_class (str)
      - best_prob (float)
      - topk_list: list[ (class_name, prob) ]
      - overlay_pil: PIL.Image (ảnh Grad-CAM)
    """
    x = preprocess(pil_img).unsqueeze(0).to(DEVICE, non_blocking=True)

    with torch.no_grad():
        if USE_AMP and DEVICE.type == "cuda":
            with torch.amp.autocast("cuda", dtype=torch.float16):
                out = MODEL(x)
        else:
            out = MODEL(x)

    probs = (torch.sigmoid(out) if USE_LOGITS_HEAD else out)[0].float().cpu().numpy().tolist()

    idx = int(np.argmax(probs))
    best_class = CLASS_NAMES[idx]
    best_prob = float(probs[idx])

    pairs = list(zip(CLASS_NAMES, probs))
    pairs.sort(key=lambda p: p[1], reverse=True)
    topk_list = pairs[:TOPK]

    overlay_pil = gradcam_overlay_center_aligned(
        MODEL, x_norm_224=x, orig_img=pil_img,
        class_idx=idx,
        use_logits_head=USE_LOGITS_HEAD,
        alpha=ALPHA
    )

    return best_class, best_prob, topk_list, overlay_pil


# ===== FLASK (API ONLY) =====
app = Flask(__name__)

@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "message": "VietCXR API is running",
        "device": str(DEVICE),
        "use_logits_head": USE_LOGITS_HEAD
    })

@app.route("/api/predict", methods=["POST"])
def api_predict():
    start_t = time.time()

    if "image" not in request.files:
        return jsonify({
            "status": "error",
            "message": "Thiếu file 'image' trong form-data.",
            "code": 400
        }), 400

    f = request.files["image"]
    if f.filename == "":
        return jsonify({
            "status": "error",
            "message": "Tên file rỗng.",
            "code": 400
        }), 400

    try:
        img = Image.open(io.BytesIO(f.read())).convert("RGB")

        best_class, best_prob, topk_list, overlay_pil = run_inference_with_gradcam(img)

        # Upload Grad-CAM lên Cloudinary
        buf = io.BytesIO()
        overlay_pil.save(buf, format="PNG")
        buf.seek(0)

        upload_result = cloudinary.uploader.upload(
            buf,
            folder="vietcxr/gradcam",
            resource_type="image"
        )
        gradcam_url = upload_result.get("secure_url")

        elapsed = time.time() - start_t

        resp = {
            "status": "success",
            "message": "Inference completed.",
            "latency_sec": elapsed,
            "prediction": {
                "best": {
                    "class": best_class,
                    "prob": best_prob
                },
                "topk": [
                    {"class": name, "prob": float(p)}
                    for (name, p) in topk_list
                ]
            },
            "gradcam": {
                "target_class": best_class,
                "url": gradcam_url
            },
        }
        return jsonify(resp), 200

    except Exception as e:
        print("[ERROR] /api/predict failed:\n" + traceback.format_exc(), flush=True)
        return jsonify({
            "status": "error",
            "message": "Internal server error.",
            "detail": str(e),
            "code": 500
        }), 500


# ===== BOOT =====
def _build_model():
    model = DenseNet121_Logits(N_CLASSES) if USE_LOGITS_HEAD else DenseNet121_Sigmoid(N_CLASSES)
    model.to(DEVICE).eval()
    load_checkpoint_into(model, MODEL_PATH, DEVICE)
    return model

print(f"[INFO] Loading model from {MODEL_PATH} on {DEVICE} (logits_head={USE_LOGITS_HEAD}, amp={USE_AMP})")
MODEL = _build_model()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port, debug=False)
