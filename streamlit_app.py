import io, re
from typing import Optional, Tuple, List

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageOps
import easyocr  # OCR CPU (pas de Tesseract)

# ------------------- UI -------------------
st.set_page_config(page_title="Remplacement in-place (OCR+Inpaint)", layout="centered")
st.title("🔁 Remplacer un texte au même endroit (automatique)")

uploaded = st.file_uploader("Choisis une image (PNG/JPG)", type=["png", "jpg", "jpeg"])
st.caption('Exemples : **Remplacer 1400 par 1500** • **Remplacer 11,00 par 13 500**')

instruction = st.text_input("Instruction (FR)", value="Remplacer 1400 par 1500")
replace_all = st.checkbox("Remplacer toutes les occurrences", value=True)

# Police TTF optionnelle (pour un style précis). Sinon, fallback système.
uploaded_font = st.file_uploader("Police .ttf (optionnel)", type=["ttf"])

apply_watermark = st.checkbox("Ajouter watermark « Mockup »", value=False)

# ---------------- OCR (EasyOCR) ----------------
@st.cache_resource(show_spinner=False)
def get_reader():
    return easyocr.Reader(["fr", "en"], gpu=False)

# ---------------- Utils ----------------
def pil_to_cv(im: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGBA2BGR if im.mode == "RGBA" else cv2.COLOR_RGB2BGR)

def cv_to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))

def load_font(ttf_bytes: Optional[bytes], size: int) -> ImageFont.FreeTypeFont:
    if ttf_bytes:
        try:
            return ImageFont.truetype(io.BytesIO(ttf_bytes), size)
        except Exception:
            pass
    # Fallback (DejaVu si présent, sinon bitmap par défaut)
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()

def measure_text(text: str, font: ImageFont.FreeTypeFont) -> Tuple[int, int]:
    im = Image.new("RGB", (1,1))
    d = ImageDraw.Draw(im)
    x0,y0,x1,y1 = d.textbbox((0,0), text, font=font)
    return (x1-x0, y1-y0)

def estimate_font_size_for_box(text: str, ttf_bytes: Optional[bytes], target_w: int, target_h: int) -> int:
    lo, hi = 6, max(12, target_h*3)
    best = lo
    for _ in range(16):
        mid = (lo + hi)//2
        font = load_font(ttf_bytes, mid)
        w,h = measure_text(text, font)
        if h <= target_h and w <= int(target_w*1.15):  # tolérance largeur
            best = mid; lo = mid + 1
        else:
            hi = mid - 1
    return max(6, best)

def estimate_text_color(im: Image.Image, box: Tuple[int,int,int,int]) -> Tuple[int,int,int,int]:
    """Couleur approx du texte (moyenne des 20% pixels les + sombres de la zone)."""
    x,y,w,h = box
    crop = im.crop((x,y,x+w,y+h)).convert("RGB")
    arr = np.asarray(crop).reshape(-1, 3)
    if arr.size == 0: return (255,255,255,255)
    lum = 0.2126*arr[:,0] + 0.7152*arr[:,1] + 0.0722*arr[:,2]
    k = max(1, int(0.2 * len(lum)))
    idx = np.argpartition(lum, k)[:k]
    r,g,b = arr[idx].mean(axis=0).astype(int)
    return (int(r),int(g),int(b),255)

def inpaint_rect(cv_img: np.ndarray, box: Tuple[int,int,int,int], inflate: int = 4) -> np.ndarray:
    x,y,w,h = box
    x1 = max(0, x - inflate); y1 = max(0, y - inflate)
    x2 = min(cv_img.shape[1]-1, x + w + inflate); y2 = min(cv_img.shape[0]-1, y + h + inflate)
    mask = np.zeros(cv_img.shape[:2], dtype=np.uint8); mask[y1:y2, x1:x2] = 255
    return cv2.inpaint(cv_img, mask, 3, cv2.INPAINT_TELEA)

def normalize(s: str) -> str:
    return (s or "").lower().replace(" ", "").replace(",", "").replace("’","").replace("'","")

def ocr_boxes(im: Image.Image) -> List[Tuple[str, Tuple[int,int,int,int]]]:
    """Retourne [(texte, (x,y,w,h))] via EasyOCR."""
    cv = pil_to_cv(im)
    result = get_reader().readtext(cv, detail=1)  # list of (pts, text, conf)
    out = []
    for (pts, txt, conf) in result:
        if not txt: 
            continue
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        x, y, w, h = int(min(xs)), int(min(ys)), int(max(xs)-min(xs)), int(max(ys)-min(ys))
        if w*h <= 0: 
            continue
        out.append((txt.strip(), (x,y,w,h)))
    return out

# --------- Parse instruction "Remplacer X par Y" ----------
def parse_replace(t: str):
    t = (t or "").strip()
    m = re.search(r"(?i)(?:remplacer|replace|changer|change)\s+(.+?)\s+(?:par|en)\s+(.+)$", t)
    if not m: 
        return None, None
    old = m.group(1).strip()
    new = m.group(2).strip()
    return old, new

# ------------------- App logique -------------------
if uploaded:
    im_src = Image.open(uploaded)
    # Respect EXIF (rotation)
    im = ImageOps.exif_transpose(im_src).convert("RGBA")
    st.image(im, caption="Image d’entrée", use_container_width=True)

    old_text, new_text = parse_replace(instruction)

    if st.button("🚀 Exécuter"):
        if not old_text or not new_text:
            st.error('Instruction invalide. Exemple : "Remplacer 1400 par 1500"')
        else:
            boxes = ocr_boxes(im)
            cv_img = pil_to_cv(im)
            done = 0

            norm_old = normalize(old_text)

            # Parcours des boîtes OCR et remplace "in place"
            for txt, box in boxes:
                if norm_old in normalize(txt) or normalize(txt) in norm_old:
                    # 1) Efface l'ancien
                    cv_img = inpaint_rect(cv_img, box, inflate=6)
                    x,y,w,h = box

                    # 2) Taille auto pour la nouvelle valeur
                    ttf_bytes = uploaded_font.read() if uploaded_font is not None else None
                    size = estimate_font_size_for_box(new_text, ttf_bytes, w, h)
                    font = load_font(ttf_bytes, size)

                    # 3) Couleur auto proche de l'original
                    color_rgba = estimate_text_color(im, box)

                    # 4) Écrit au même endroit (mêmes coordonnées)
                    pil_after = cv_to_pil(cv_img).convert("RGBA")
                    d = ImageDraw.Draw(pil_after)
                    d.text((x, y), new_text, font=font, fill=color_rgba)
                    cv_img = pil_to_cv(pil_after)

                    done += 1
                    if not replace_all:
                        break  # une seule occurrence

            out = cv_to_pil(cv_img).convert("RGBA")

            if done == 0:
                st.warning("Aucune occurrence trouvée par l’OCR. Essaie une variante (ex. sans espaces/virgules) ou augmente la netteté de l’image.")
            else:
                if apply_watermark:
                    d = ImageDraw.Draw(out)
                    wm_font = load_font(uploaded_font.read() if uploaded_font else None, max(18, out.size[0]//40))
                    x0,y0,x1,y1 = d.textbbox((0,0), "Mockup", font=wm_font)
                    w,h = x1-x0, y1-y0
                    X,Y = out.size[0]-w-16, out.size[1]-h-16
                    d.rectangle([X-8,Y-4,X+w+8,Y+h+4], fill=(0,0,0,100))
                    d.text((X,Y), "Mockup", font=wm_font, fill=(255,255,255,180))

                st.success(f"✅ Terminé — {done} occurrence(s) remplacée(s).")
                st.image(out, caption="Résultat", use_container_width=True)
                buf = io.BytesIO()
                fmt = (im_src.format or "PNG")
                out.save(buf, format=fmt)
                st.download_button("⬇️ Télécharger l’image", data=buf.getvalue(),
                                   file_name=f"result.{fmt.lower()}",
                                   mime=f"image/{fmt.lower()}")

else:
    st.info("➡️ Téléverse une image puis tape :  Remplacer 1400 par 1500")
