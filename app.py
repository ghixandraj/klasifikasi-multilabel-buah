import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import gdown
import os

# --- 1. Setup ---
MODEL_URL = 'https://drive.google.com/file/d/1wImyFjOLuZN_x69j9n9CZPDmvOkyimlV/view?usp=sharing'  # Ganti dengan File ID model
MODEL_PATH = 'model_hsvlt_trained.pt'
LABELS = ['alpukat', 'alpukat_matang', 'alpukat_mentah', 'belimbing', 'belimbing_matang', 'belimbing_mentah', 'mangga', 'mangga_matang', 'mangga_mentah']
THRESHOLD = 0.5

# --- 2. Download model dari Google Drive jika belum ada ---
if not os.path.exists(MODEL_PATH):
    with st.spinner('Mengunduh model dari Google Drive...'):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# --- 3. Load model ---
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
        dummy_text = torch.zeros((batch_size, 1)).to(image.device)

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
model = HSVLTModel().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# --- 4. Transformasi Gambar ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# --- 5. Antarmuka Streamlit ---
st.title("🍉 Klasifikasi Multilabel Buah")
st.write("Upload gambar buah, sistem akan mendeteksi beberapa label sekaligus.")

uploaded_file = st.file_uploader("Unggah gambar buah", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Gambar yang diunggah', use_column_width=True)

    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = outputs.cpu().numpy()[0]

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
