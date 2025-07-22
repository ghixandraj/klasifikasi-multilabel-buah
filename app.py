import streamlit as st
import torch
import torch.nn as nn
from safetensors.torch import safe_open
import torchvision.transforms as transforms
from PIL import Image
import gdown
import os
import numpy as np

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# --- 1. Setup ---
MODEL_URL = 'https://drive.google.com/uc?id=1U-phCmWnChUJqxvon8DK-C1rJud1WXv4'
MODEL_PATH = 'model_1.safetensors'
LABELS = [
    'alpukat', 'alpukat_matang', 'alpukat_mentah',
    'belimbing', 'belimbing_matang', 'belimbing_mentah',
    'mangga', 'mangga_matang', 'mangga_mentah'
]
THRESHOLD = 0.5

def download_model():
    with st.spinner('🔄 Mengunduh ulang model dari Google Drive...'):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False, fuzzy=True)

if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 50000:
    if os.path.exists(MODEL_PATH):
        st.warning("📦 Ukuran file model terlalu kecil, kemungkinan korup. Mengunduh ulang...")
        os.remove(MODEL_PATH)
    download_model()

# --- 2. Komponen Model ---
class PatchEmbedding(nn.Module):
    def __init__(self, img_size: int, patch_size: int, emb_size: int):
        super().__init__()
        self.proj = nn.Conv2d(3, emb_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class WordEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()

    def forward(self, x):
        return x

class FeatureFusion(nn.Module):
    def forward(self, v, t):
        if t.size(1) < v.size(1):
            pad_len = v.size(1) - t.size(1)
            pad = torch.zeros(t.size(0), pad_len, t.size(2), device=t.device)
            t = torch.cat([t, pad], dim=1)
        elif t.size(1) > v.size(1):
            t = t[:, :v.size(1), :]
        return torch.cat([v, t], dim=-1)

class ScaleTransformation(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.linear(x)

class ChannelUnification(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x)

class InteractionBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads=16, batch_first=True)

    def forward(self, x):
        return self.attn(x, x, x)[0]

class CrossScaleAggregation(nn.Module):
    def forward(self, x):
        return x.unsqueeze(1).mean(dim=1)

class HamburgerHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.linear(x)

class MLPClassifier(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.mlp(x)

class HSVLTModel(nn.Module):
    def __init__(self, img_size=224, patch_size=16, emb_size=768, num_classes=9):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, emb_size)
        self.word_embed = WordEmbedding(emb_size)
        self.concat = FeatureFusion()
        self.scale_transform = ScaleTransformation(emb_size * 2, emb_size)
        self.channel_unification = ChannelUnification(emb_size)
        self.interaction_blocks = nn.ModuleList([
            InteractionBlock(emb_size),
            InteractionBlock(emb_size)
        ])
        self.csa = CrossScaleAggregation()
        self.head = HamburgerHead(emb_size, emb_size)
        self.classifier = MLPClassifier(emb_size, num_classes)

    def forward(self, image):
        batch_size = image.size(0)
        dummy_text = torch.zeros((batch_size, 1, 768)).to(image.device)
        image_feat = self.patch_embed(image)
        text_feat = self.word_embed(dummy_text)
        x = self.concat(image_feat, text_feat)
        x = self.scale_transform(x)
        x = self.channel_unification(x)
        for block in self.interaction_blocks:
            x = block(x)
        x = self.csa(x)
        x = self.head(x)
        x = x.mean(dim=1)
        output = self.classifier(x)
        return output

# --- 3. Load Model (.safetensors) ---
use_cuda = torch.cuda.is_available()
device = 'cuda' if use_cuda else 'cpu'

try:
    with safe_open(MODEL_PATH, framework="pt", device=device) as f:
        state_dict = {k: f.get_tensor(k) for k in f.keys()}
    model = HSVLTModel(img_size=224, patch_size=14, emb_size=768, num_classes=len(LABELS)).to(device)
    model.load_state_dict(state_dict)
    model.eval()
except Exception as e:
    st.error(f"❌ Gagal memuat model: {e}")
    st.stop()

# --- 4. Transformasi Gambar ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
    # Jika model dilatih dengan normalization:
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- 5. Antarmuka Streamlit ---
st.title("🍉 Klasifikasi Multilabel Buah")
st.write("Upload gambar buah, sistem akan mendeteksi beberapa label sekaligus.")

uploaded_file = st.file_uploader("Unggah gambar buah", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Gambar Input", use_container_width=True)

    input_tensor = transform(image).unsqueeze(0).to(device)

    # --- Prediksi ---
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.sigmoid(outputs).cpu().numpy()[0].tolist()

    detected_labels = [(label, prob) for label, prob in zip(LABELS, probs) if prob >= THRESHOLD]
    detected_labels.sort(key=lambda x: x[1], reverse=True)

    st.subheader("🔍 Label Terdeteksi:")
    if detected_labels:
        for label, prob in detected_labels:
            st.write(f"✅ **{label}** ({prob:.2%})")
    else:
        st.write("🚫 Tidak ada label yang terdeteksi.")

    with st.expander("📊 Lihat Semua Probabilitas"):
        for label, prob in zip(LABELS, probs):
            st.write(f"{label}: {prob:.2%}")

    # --- Grad-CAM ---
    st.subheader("📸 Heatmap perhatian model (Grad-CAM)")
    try:
        top_idx = int(torch.tensor(probs).argmax().item())
        targets = [ClassifierOutputTarget(top_idx)]
        target_layers = [model.patch_embed.proj]

        cam = GradCAM(model=model, target_layers=target_layers, use_cuda=use_cuda)
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]

        rgb_img = np.array(image.resize((224, 224))) / 255.0
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        st.image(cam_image, caption=f"Area fokus untuk label '{LABELS[top_idx]}'")
    except Exception as e:
        st.warning(f"⚠️ Tidak bisa membuat heatmap Grad-CAM: {e}")
