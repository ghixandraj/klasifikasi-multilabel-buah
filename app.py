import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import gdown
import os

# --- 1. Setup ---
MODEL_URL = 'https://drive.google.com/uc?id=1U-phCmWnChUJqxvon8DK-C1rJud1WXv4'  # Link langsung (direct) dari Google Drive
MODEL_PATH = 'model_hsvlt_trained.pt'
LABELS = [
    'alpukat', 'alpukat_matang', 'alpukat_mentah',
    'belimbing', 'belimbing_matang', 'belimbing_mentah',
    'mangga', 'mangga_matang', 'mangga_mentah'
]
THRESHOLD = 0.5

# --- 2. Download model dari Google Drive jika belum ada ---
def download_model():
    with st.spinner('🔄 Mengunduh ulang model dari Google Drive...'):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# Jika belum ada model, unduh
if not os.path.exists(MODEL_PATH):
    download_model()

# Jika ukuran file mencurigakan kecil, hapus & unduh ulang
elif os.path.getsize(MODEL_PATH) < 50000:  # Threshold bisa disesuaikan
    st.warning("📦 Ukuran file model terlalu kecil, kemungkinan korup. Mengunduh ulang...")
    os.remove(MODEL_PATH)
    download_model()


# --- 5. Komponen Model (semua class yang dibutuhkan) ---
class PatchEmbedding(nn.Module):
    def __init__(self, img_size: int, patch_size: int, emb_size: int):
        super().__init__()
        assert isinstance(patch_size, int), f"patch_size must be int, got {type(patch_size)}"
        assert isinstance(emb_size, int), f"emb_size must be int, got {type(emb_size)}"
        self.proj = nn.Conv2d(3, emb_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # [B, emb_size, H/patch, W/patch]
        x = x.flatten(2).transpose(1, 2)  # [B, N, emb_size]
        return x

class WordEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()

    def forward(self, x):
        return x  # Dummy placeholder

class FeatureFusion(nn.Module):
    def forward(self, v, t):
        # Pastikan dimensi temporal/textual dan visual sepadan
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
        return x.unsqueeze(1).mean(dim=1)  # Cross-scale average

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

# --- 4. Load model ---
class HSVLTModel(nn.Module):
    def __init__(self, img_size=224, patch_size=16, emb_size=768, num_classes=9):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, emb_size)
        self.word_embed = WordEmbedding(emb_size)
        self.concat = FeatureFusion()
        self.scale_transform = ScaleTransformation(emb_size * 2, emb_size)
        self.channel_unification = ChannelUnification(emb_size)
        self.interaction_block = InteractionBlock(emb_size)
        self.csa = CrossScaleAggregation()
        self.head = HamburgerHead(emb_size, emb_size)
        self.classifier = MLPClassifier(emb_size, num_classes)

    def forward(self, image):
        # Dummy text input
        batch_size = image.size(0)
        dummy_text = torch.zeros((batch_size, 1, 768)).to(image.device)  # Sesuaikan dimensi

        image_feat = self.patch_embed(image)
        text_feat = self.word_embed(dummy_text)

        x = self.concat(image_feat, text_feat)
        x = self.scale_transform(x)
        x = self.channel_unification(x)
        x = self.interaction_block(x)
        x = self.csa(x)
        x = self.head(x)
        x = x.mean(dim=1)
        output = self.classifier(x)
        return output

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# sesuaikan parameter di dalam kode dengan parameter ketika membuat model
model = HSVLTModel(
    img_size=224,
    patch_size=14,
    emb_size=768,
    num_classes=len(LABELS)
).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# --- 5. Transformasi Gambar ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# --- 6. Antarmuka Streamlit ---
st.title("🍉 Klasifikasi Multilabel Buah")
st.write("Upload gambar buah, sistem akan mendeteksi beberapa label sekaligus.")

uploaded_file = st.file_uploader("Unggah gambar buah", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Gambar yang diunggah', use_column_width=True)

    input_tensor = transform(image).unsqueeze(0).to(device)
     # --- Dummy output supaya UI tetap jalan ---
    probs = [0.1] * len(LABELS)  # Semua probabilitas 10%
    # Jika model sudah diaktifkan, ganti dengan:
    # with torch.no_grad():
    #     outputs = model(input_tensor)
    #     probs = outputs.cpu().numpy()[0]

    # Deteksi label dengan threshold
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