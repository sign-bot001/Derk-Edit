import io, re
from typing import Optional, Tuple, List

import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFont
import streamlit as st

# OCR stable (CPU, ONNX)
from rapidocr_onnxruntime import RapidOCR

# Inpainting sans OpenCV
from skimage.restoration import inpaint_biharmonic
from skimage.draw import rectangle

# ==================== UI ====================
st.set_page_config(page_title="Remplacement in-place (OCR + Inpaint)", layout="centered")
st.title("🔁 Remplacer un texte au même endroit (automatique)")

uploaded = st.file_uploader("Choisis une image (PNG/JPG)", type=["png", "jpg", "jpeg"])
st.caption('Exemples : **Remplacer 1400 par 1500** • **Remplacer 11,00 par 13 500**')

instruction = st.text_input("Instruction (FR)", value="Remplacer 1400 par 1500")
replace_all = st.checkbox("Remplacer toutes les occurrences", value=True)

uploaded_font = st.file_uploader("Police .ttf (optionnel)", type=["ttf"])
apply_watermark = st.checkbox("Ajouter watermark « Mockup »", value=False)

# ==================== OCR ====================
@st.cache_resource(show_spinner=False)
def get_ocr():
    return RapidOCR()

# ==================== Utils ====================
def pil_to_np(im: Image.Image) -> np.ndarray:
    """PIL -> numpy (RGB uint8)"""
    return np.array(im.convert("RGB"))

def np_to_pil(arr: np.ndarray) -> Image.Image:
    """numpy (RGB uint8) -> PIL"""
    return Image.fromarray(arr.astype(np.uint8), mode="RGB")

def normalize(s: str) -> str:
    return (s or "").lower().replace(" ", "").replace(",", "").replace("’","").replace("'","")

def estimate_text_color(pil_im: Image.Image, box: Tuple[int,int,int,int]) -> Tuple[int,int,int,int]:
    x,y,w,h = box
    crop = pil_im.crop((x,y,x+w,y+h)).convert("RGB")
    arr = np.asarray(crop).reshape(-1, 3)
    if arr.size == 0: return (255,255,255,255)
    lum = 0.2126*arr[:,0] + 0.7152*arr[:,1] + 0.0722*arr[:,2]
    k = max(1, int(0.2 * len(lum)))
    idx = np.argpartition(lum, k)[:k]
    r,g,b = arr[idx].mean(axis=0).astype(int)
    return int(r), int(g), int(b), 255

def measure_text(text: str, font: ImageFont.FreeTypeFont) -> Tuple[int, int]:
    dummy = Image.new("RGB", (1,1)); d = ImageDraw.Draw(dummy)
    x0,y0,x1,y1 = d.textbbox((0,0), text, font=font)
    return x1-x0, y1-y0

def load_font(ttf_bytes: Optional[bytes], size: int) -> ImageFont.FreeTypeFont:
    if ttf_bytes:
        try:
            return ImageFont.truetype(io.BytesIO(ttf_bytes), size)
        except Exception:
            pass
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()

def estimate_font_size_for_box(text: str, ttf_bytes: Optional[bytes], target_w: int, target_h: int) -> int:
    lo, hi = 6, max(12, target_h*3); best = lo
    for _ in range(16):
        mid = (lo + hi)//2
        w,h = measure_text(text, load_font(ttf_bytes, mid))
        if h <= target_h and w <= int(target_w*1.15):
            best = mid; lo = mid + 1
        else:
            hi = mid - 1
    return max(6, best)

def inpaint_rect_np(img_rgb: np.ndarray, box: Tuple[int,int,int,int], inflate: int = 4) -> np.ndarray:
    """Inpainting rectangle via biharmonic (scikit-image)."""
    x,y,w,h = box
    H, W = img_rgb.shape[:2]
    x1 = max(0, x - inflate); y1 = max(0, y - inflate)
    x2 = min(W, x + w + inflate); y2 = min(H, y + h + inflate)

    # masque booléen
    mask = np.zeros((H, W), dtype=bool)
    rr, cc = rectangle(start=(y1, x1), end=(y2-1, x2-1), shape=mask.shape)
    mask[rr, cc] = True

    # inpaint canal par canal (biharmonic fonctionne sur float [0,1])
    img_float = img_rgb.astype(np.float32) / 255.0
    out = np.empty_like(img_float)
    for c in range(3):
        out[..., c] = inpaint_biharmonic(img_float[..., c], mask, multichannel=False)
    out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
    return out

def ocr_boxes(pil_im: Image.Image) -> List[Tuple[str, Tuple[int,int,int,int]]]:
    """[(texte, (x,y,w,h))] avec RapidOCR."""
    img_bgr = np.array(pil_im.convert("RGB"))[:, :, ::-1]  # PIL->BGR
    result, _ = get_ocr()(img_bgr)
    out = []
    if result:
        for (pts, txt, conf) in result:
            if not txt: continue
            xs = [int(p[0]) for p in pts]; ys = [int(p[1]) for p in pts]
            x, y, w, h = min(xs), min(ys), max(xs)-min(xs), max(ys)-min(ys)
            if w*h > 0:
                out.append((txt.strip(), (x,y,w,h)))
    return out

def parse_replace(t: str):
    m = re.search(r"(?i)(?:remplacer|replace|changer|change)\s+(.+?)\s+(?:par|en)\s+(.+)$", (t or "").strip())
    if not m: return None, None
    return m.group(1).strip(), m.group(2).strip()

# ==================== App ====================
if uploaded:
    im_src = Image.open(uploaded)
    im = ImageOps.exif_transpose(im_src).convert("RGBA")
    st.image(im, caption="Image d’entrée", use_container_width=True)

    old_text, new_text = parse_replace(instruction)

    if st.button("🚀 Exécuter"):
        if not old_text or not new_text:
            st.error('Instruction invalide. Exemple : "Remplacer 1400 par 1500"')
        else:
            boxes = ocr_boxes(im)
            img_np = pil_to_np(im)  # RGB uint8
            done = 0
            norm_old = normalize(old_text)
            ttf_bytes = uploaded_font.read() if uploaded_font else None

            for txt, box in boxes:
                if norm_old in normalize(txt) or normalize(txt) in norm_old:
                    # 1) efface (inpaint sans OpenCV)
                    img_np = inpaint_rect_np(img_np, box, inflate=6)

                    # 2) taille + couleur auto
                    x,y,w,h = box
                    size = estimate_font_size_for_box(new_text, ttf_bytes, w, h)
                    font = load_font(ttf_bytes, size)
                    color_rgba = estimate_text_color(im, box)

                    # 3) écris au même endroit
                    pil_after = np_to_pil(img_np).convert("RGBA")
                    d = ImageDraw.Draw(pil_after)
                    d.text((x, y), new_text, font=font, fill=color_rgba)
                    img_np = pil_to_np(pil_after)
                    done += 1
                    if not replace_all:
                        break

            out = np_to_pil(img_np).convert("RGBA")

            if done == 0:
                st.warning("Aucune occurrence trouvée par l’OCR. Essaie une variante (ex. sans espaces/virgules) ou une image plus nette.")
            else:
                if apply_watermark:
                    d = ImageDraw.Draw(out)
                    wm_font = load_font(ttf_bytes, max(18, out.size[0]//40))
                    x0,y0,x1,y1 = d.textbbox((0,0), "Mockup", font=wm_font)
                    W,H = out.size; w,h = x1-x0, y1-y0
                    X,Y = W-w-16, H-h-16
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
