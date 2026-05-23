# KATHMANDU UNIVERSITY
## Department of Artificial Intelligence
### Course: AICL 311 — Image Processing
### Lab 1: Introduction to Basic Image Processing Using Python

**Program:** Bachelor of Technology in Artificial Intelligence (BTech AI)  
**Semester:** 6th Semester  
**Date:** May 2026

---

## Objectives

By the end of this lab, the student is able to:

1. Load and display images using five Python libraries: OpenCV, Pillow, Matplotlib, ImageIO, and scikit-image.
2. Extract and manipulate image properties — dimensions, channels, pixel values, and data types.
3. Implement image resizing using all seven interpolation methods available in OpenCV.
4. Perform image quantization at 2, 4, 8, and 16 gray levels, supported by step-by-step calculations.
5. Write Python functions `n_4()` and `n_8()` that return the 4-connected and 8-connected pixel neighbors of a given pixel, with proper boundary handling.

---

## Theoretical Background

### Digital Images and Pixel Representation

A digital image is a 2D matrix (or 3D for colour) of discrete values called **pixels**. Each pixel stores an intensity value. For an 8-bit image, values range from 0 (black) to 255 (white) for grayscale, or from (0,0,0) to (255,255,255) in RGB colour space.

A colour image stored as a NumPy array has shape `(H, W, C)`:
- **H** — height in pixels
- **W** — width in pixels  
- **C** — number of channels (3 for RGB/BGR, 1 for grayscale)

### Image Loading Libraries

Different Python libraries read images into different internal formats:

| Library | Return Type | Channel Order | Value Range |
|---------|------------|---------------|------------|
| OpenCV (`cv2`) | `numpy.ndarray` | **BGR** | 0–255 (`uint8`) |
| Pillow (`PIL.Image`) | `PIL.Image` object | RGB | 0–255 (`uint8`) |
| Matplotlib (`mpimg`) | `numpy.ndarray` | RGB | 0.0–1.0 (`float32` for PNG) |
| ImageIO (`imageio.v3`) | `numpy.ndarray` | RGB | 0–255 (`uint8`) |
| scikit-image (`skimage.io`) | `numpy.ndarray` | RGB | 0–255 (`uint8`) |

OpenCV's BGR channel order is a historical convention inherited from older camera hardware. Converting BGR to RGB before display with Matplotlib is always required: `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)`.

### Color Spaces

Images can be represented in multiple color spaces, each suited to different tasks:

| Color Space | Channels | Primary Use |
|-------------|----------|-------------|
| RGB | Red, Green, Blue | Display, general processing |
| Grayscale | Intensity | Edge detection, intensity analysis |
| HSV | Hue, Saturation, Value | Color-based segmentation |
| LAB | Lightness, A (green–red), B (blue–yellow) | Perceptually uniform comparisons |
| YCrCb | Luma, Chroma-red, Chroma-blue | JPEG compression, video coding |

### Image Interpolation

When resizing an image, the pixel grid changes size and new pixel values must be estimated. This estimation is called **interpolation**:

| Method | OpenCV Flag | Description |
|--------|-------------|-------------|
| Nearest Neighbour | `INTER_NEAREST` | Copies the value of the nearest source pixel. Fastest; blocky artefacts on upscale. |
| Bilinear | `INTER_LINEAR` | Weighted average of 4 surrounding pixels. Default; good speed/quality balance. |
| Bicubic | `INTER_CUBIC` | Weighted average of 16 surrounding pixels. Sharper than bilinear; slower. |
| Pixel Area | `INTER_AREA` | Averages over the source pixel area. Best for downscaling; prevents moiré. |
| Lanczos | `INTER_LANCZOS4` | Sinc-based 8×8 window filter. Highest quality; slowest. |
| Bit-Exact Bilinear | `INTER_LINEAR_EXACT` | Reproducible bilinear across platforms. Same quality as `INTER_LINEAR`. |
| Bit-Exact Nearest | `INTER_NEAREST_EXACT` | Reproducible nearest-neighbour across platforms. |

**General guidelines:**  
- For upscaling: prefer `INTER_CUBIC` or `INTER_LANCZOS4`.  
- For downscaling: prefer `INTER_AREA`.

### Image Quantization

Quantization reduces the number of distinct intensity levels in an image. A standard 8-bit grayscale image has 2⁸ = 256 levels (0–255). Reducing to N levels decreases image fidelity but also reduces the storage requirement.

**Uniform Quantization Formula:**

$$
\text{step} = \frac{256}{N}
$$

$$
q(p) = \left\lfloor \frac{p}{\text{step}} \right\rfloor \times \text{step}
$$

where p is the original pixel value and q(p) is the quantized output value.

**Calculation Table (example pixel = 150):**

| Levels (N) | Bits | Step = 256÷N | Calculation | Result |
|-----------|------|-------------|-------------|--------|
| 2 | 1-bit | 128 | ⌊150÷128⌋ × 128 = 1 × 128 | **128** |
| 4 | 2-bit | 64 | ⌊150÷64⌋ × 64 = 2 × 64 | **128** |
| 8 | 3-bit | 32 | ⌊150÷32⌋ × 32 = 4 × 32 | **128** |
| 16 | 4-bit | 16 | ⌊150÷16⌋ × 16 = 9 × 16 | **144** |

### Pixel Connectivity and Neighborhoods

In image processing, the **neighborhood** of a pixel determines which surrounding pixels are considered adjacent. Two models are used:

**4-Connectivity (Cross Neighborhood)**  
Only horizontal and vertical neighbors are included — up to 4 neighbors:

```
        (r-1, c)
           ↑
(r, c-1) ← P → (r, c+1)
           ↓
        (r+1, c)
```

**8-Connectivity (Square Neighborhood)**  
Horizontal, vertical, and diagonal neighbors — up to 8 neighbors:

```
(r-1,c-1)  (r-1,c)  (r-1,c+1)
(r,  c-1)     P     (r,  c+1)
(r+1,c-1)  (r+1,c)  (r+1,c+1)
```

Pixels at the image boundary have fewer valid neighbors. Proper boundary handling returns `None` for out-of-bounds positions.

**Applications:** Region growing, flood fill, connected component labelling, morphological operations, edge detection.

---

## Task 1: Loading and Displaying Images with Five Libraries

### Code

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import imageio.v3 as iio
from skimage import io as skio

IMG_COLOR = 'cat.png'

# OpenCV (BGR → RGB conversion required for display)
img_bgr = cv2.imread(IMG_COLOR)
img_cv  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
print(f'OpenCV  shape={img_bgr.shape}  dtype={img_bgr.dtype}')

# Pillow
img_pil = Image.open(IMG_COLOR)
print(f'Pillow  size={img_pil.size}  mode={img_pil.mode}')

# Matplotlib (PNG returns float32, 0.0–1.0)
img_mpl = mpimg.imread(IMG_COLOR)
print(f'Matplotlib  shape={img_mpl.shape}  dtype={img_mpl.dtype}  range=[{img_mpl.min():.2f},{img_mpl.max():.2f}]')

# ImageIO
img_iio = iio.imread(IMG_COLOR)
print(f'ImageIO  shape={img_iio.shape}  dtype={img_iio.dtype}')

# scikit-image
img_sk = skio.imread(IMG_COLOR)
print(f'scikit-image  shape={img_sk.shape}  dtype={img_sk.dtype}')
```

### Output and Explanation

All five libraries load the same `cat.png` file. The key differences are:

- **OpenCV** returns a BGR-ordered `uint8` array — must be converted before display.  
- **Pillow** returns a `PIL.Image` object; `.size` gives `(width, height)`, the reverse of NumPy's `(H, W)`.  
- **Matplotlib** returns `float32` with values in `[0.0, 1.0]` for PNG files — multiply by 255 before arithmetic with integer images.  
- **ImageIO** and **scikit-image** both return `uint8` RGB arrays; no conversion needed.

All five panels display the same cat image with identical visual appearance, confirming that the different internal representations encode the same pixel data.

---

## Task 2: Extracting and Manipulating Image Properties

### Code — Property Extraction

```python
img = img_cv  # shape: (H, W, 3), dtype: uint8
height, width, channels = img.shape

print(f'Height          : {height} pixels')
print(f'Width           : {width} pixels')
print(f'Channels        : {channels}')
print(f'Total pixels    : {height * width:,}')
print(f'Data type       : {img.dtype}')
print(f'Memory used     : {img.nbytes / 1024:.2f} KB')

for i, ch_name in enumerate(['Red', 'Green', 'Blue']):
    ch = img[:, :, i]
    print(f'{ch_name}: min={ch.min()}  max={ch.max()}  mean={ch.mean():.2f}')
```

### Output

```
Height          : 500 pixels
Width           : 500 pixels
Channels        : 3
Total pixels    : 250,000
Data type       : uint8
Memory used     : 732.42 KB

Red  : min=16   max=255  mean=150.21
Green: min=6    max=255  mean=133.67
Blue : min=0    max=253  mean=66.00
```

The Red channel has the highest mean (150.21) and Blue the lowest (66.00), consistent with the warm tones of a cat photograph. Total memory = 500 × 500 × 3 bytes = 750,000 bytes ≈ 732 KB.

### Code — Spatial Manipulations

```python
h, w = img.shape[:2]

# Crop — NumPy slicing extracts a rectangular region
img_crop   = img[h//4 : 3*h//4, w//4 : 3*w//4]   # center 50%

# Flip — flipCode: 1=horizontal, 0=vertical, -1=both
img_flip_h = cv2.flip(img, 1)
img_flip_v = cv2.flip(img, 0)
img_flip_b = cv2.flip(img, -1)

# Rotation — build affine matrix, then apply with warpAffine
center = (w//2, h//2)
M_45 = cv2.getRotationMatrix2D(center, 45, 1.0)
img_rot45 = cv2.warpAffine(img, M_45, (w, h))
M_90 = cv2.getRotationMatrix2D(center, 90, 1.0)
img_rot90 = cv2.warpAffine(img, M_90, (w, h))
```

### Explanation

The crop extracts the center 25%–75% region in both dimensions, yielding a `(250, 250, 3)` array. Horizontal flip mirrors the cat left-to-right; vertical flip mirrors top-to-bottom. The 45° rotation introduces black triangular corners (unfilled regions outside the original frame), while the 90° rotation is corner-free because the image is square.

### Code — Color Space Conversions

```python
img_gray  = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
img_hsv   = cv2.cvtColor(img_cv, cv2.COLOR_RGB2HSV)
img_lab   = cv2.cvtColor(img_cv, cv2.COLOR_RGB2LAB)
img_ycrcb = cv2.cvtColor(img_cv, cv2.COLOR_RGB2YCrCb)
```

Grayscale reduces the image to a single `(500, 500)` channel. HSV, LAB, and YCrCb retain 3 channels but encode different quantities. When displayed directly as RGB, these conversions appear visually unusual because their channel values represent hue, lightness, or chroma — not red, green, blue intensities.

---

## Task 3: Image Resizing with Interpolation Methods

### Code

```python
interp_methods = {
    'NEAREST':       cv2.INTER_NEAREST,
    'LINEAR':        cv2.INTER_LINEAR,
    'CUBIC':         cv2.INTER_CUBIC,
    'AREA':          cv2.INTER_AREA,
    'LANCZOS4':      cv2.INTER_LANCZOS4,
    'LINEAR_EXACT':  cv2.INTER_LINEAR_EXACT,
    'NEAREST_EXACT': cv2.INTER_NEAREST_EXACT,
}

up_size = (1000, 1000)   # ×2 upscale
dn_size = (250, 250)     # ÷2 downscale

for name, flag in interp_methods.items():
    resized_up = cv2.resize(img_cv, up_size, interpolation=flag)
    resized_dn = cv2.resize(img_cv, dn_size, interpolation=flag)
```

### Output and Explanation

**Upscaling (×2):**

| Method | Visual Result |
|--------|--------------|
| `INTER_NEAREST` | Blocky, pixelated — each source pixel becomes a 2×2 block |
| `INTER_LINEAR` | Smooth, slight blur — blends 4 neighboring pixels |
| `INTER_CUBIC` | Sharper edges — blends 16 pixels using cubic weighting |
| `INTER_AREA` | Similar to bilinear at this scale |
| `INTER_LANCZOS4` | Sharpest result — sinc-based 8×8 kernel preserves detail |
| `INTER_LINEAR_EXACT` | Visually same as LINEAR; guarantees bit-exact reproducibility |
| `INTER_NEAREST_EXACT` | Visually same as NEAREST; bit-exact |

**Downscaling (÷2):** `INTER_AREA` performs best, averaging pixel contributions over the corresponding source region and avoiding moiré patterns. Nearest-neighbour can show aliasing along diagonal edges.

---

## Task 4: Image Quantization

### Quantization Function

```python
def quantize_image(image, levels):
    step = 256 // levels
    quantized = (image // step) * step
    return quantized.astype(np.uint8)
```

Each pixel is assigned to a bin of width `step = 256 // N`. The floor-division `p // step` gives the bin index; multiplying by `step` recovers the representative value for that bin.

### Calculation (pixel = 150)

| Levels | Step | Calculation | Result |
|--------|------|-------------|--------|
| 2 | 128 | ⌊150÷128⌋ × 128 = 1 × 128 | **128** |
| 4 | 64 | ⌊150÷64⌋ × 64 = 2 × 64 | **128** |
| 8 | 32 | ⌊150÷32⌋ × 32 = 4 × 32 | **128** |
| 16 | 16 | ⌊150÷16⌋ × 16 = 9 × 16 | **144** |

### Output and Explanation

- **2 levels:** Only two shades (0 and 128). Heavy posterisation; all tonal detail lost.  
- **4 levels:** Cat's form recognisable but with large uniform regions.  
- **8 levels:** Near-natural appearance; slight banding visible in smooth background areas.  
- **16 levels:** Nearly indistinguishable from the original 256-level image.

The pixel-value histograms make this concrete: the original histogram is broadly spread across 256 bins. After quantization, counts concentrate into tall spikes at the representative values (e.g., 0 and 128 for 2-level). The spike heights grow as the number of levels decreases — each spike represents many source pixels collapsed into one output value.

---

## Task 5: Pixel Neighborhood Functions — `n_4()` and `n_8()`

### Code

```python
def n_4(image, row, col):
    height, width = image.shape[:2]
    offsets = {
        'top':    (-1,  0),
        'bottom': ( 1,  0),
        'left':   ( 0, -1),
        'right':  ( 0,  1),
    }
    neighbors = {}
    for direction, (dr, dc) in offsets.items():
        r, c = row + dr, col + dc
        if 0 <= r < height and 0 <= c < width:
            neighbors[direction] = (r, c, image[r, c])
        else:
            neighbors[direction] = None
    return neighbors


def n_8(image, row, col):
    height, width = image.shape[:2]
    offsets = {
        'top-left':     (-1, -1),
        'top':          (-1,  0),
        'top-right':    (-1,  1),
        'right':        ( 0,  1),
        'bottom-right': ( 1,  1),
        'bottom':       ( 1,  0),
        'bottom-left':  ( 1, -1),
        'left':         ( 0, -1),
    }
    neighbors = {}
    for direction, (dr, dc) in offsets.items():
        r, c = row + dr, col + dc
        if 0 <= r < height and 0 <= c < width:
            neighbors[direction] = (r, c, image[r, c])
        else:
            neighbors[direction] = None
    return neighbors
```

**Return type:** `dict` where each key is a direction string and each value is either `(row, col, pixel_value)` or `None` (out of bounds).  
**Arguments:** Any NumPy image array (grayscale or colour), and the `(row, col)` indices of the target pixel.

### Output — Center Pixel (2,2) of a 5×5 Array

```
5×5 test array:
[[  0  10  20  30  40]
 [ 50  60  70  80  90]
 [100 110 120 130 140]
 [150 160 170 180 190]
 [200 210 220 230 240]]

4-Neighbors of (2,2):
  top    : (1,2) value=70
  bottom : (3,2) value=170
  left   : (2,1) value=110
  right  : (2,3) value=130

8-Neighbors of (2,2) — additional diagonal neighbors:
  top-left    : (1,1) value=60
  top-right   : (1,3) value=80
  bottom-right: (3,3) value=180
  bottom-left : (3,1) value=160
```

### Boundary Handling

| Position | 4-neighbors valid | 8-neighbors valid |
|----------|------------------|------------------|
| Corner pixel | 2 / 4 | 3 / 8 |
| Edge midpoint | 3 / 4 | 5 / 8 |
| Interior pixel | 4 / 4 | 8 / 8 |

Out-of-bounds positions return `None`, preventing index errors when querying any pixel regardless of location.

### Visualization

The neighborhood functions are visualised on the cat image using color-coded pixel markers:  
- **Red** — target pixel  
- **Green** — 4-connected (cardinal) neighbors  
- **Blue** — additional diagonal neighbors (8-connected only)

The zoomed panel (×8 nearest-neighbor zoom) displays the 3×3 neighborhood block clearly, distinguishing the two connectivity models.

---

## Mini Exercises

### Exercise 1 — Load and Inspect Properties

Load the cat image with OpenCV and print its shape, width, height, and number of channels.

```python
img_ex1 = cv2.imread('cat.png')
img_ex1_rgb = cv2.cvtColor(img_ex1, cv2.COLOR_BGR2RGB)
h, w, c = img_ex1_rgb.shape
print(f'Shape: {img_ex1_rgb.shape}  Width: {w}  Height: {h}  Channels: {c}')
```

**Output:** Shape: (500, 500, 3) — Width: 500 px, Height: 500 px, Channels: 3.

---

### Exercise 2 — Grayscale Conversion and Save

```python
img_gray = cv2.cvtColor(img_ex1, cv2.COLOR_BGR2GRAY)
cv2.imwrite('cat_output_grayscale.jpg', img_gray)
```

**Output:** The grayscale image has shape `(500, 500)` — the three colour channels are reduced to a single luminance channel. The file `cat_output_grayscale.jpg` is saved to disk.

---

### Exercise 3 — Resize to 200×200 and 500×500

```python
img_200 = cv2.resize(img_ex1_rgb, (200, 200), interpolation=cv2.INTER_LINEAR)
img_500 = cv2.resize(img_ex1_rgb, (500, 500), interpolation=cv2.INTER_LINEAR)
```

**Output:** The 200×200 image appears noticeably softer due to the reduction in pixel count (downscaling from the original). The 500×500 image retains the full original detail since no size change occurred. Side-by-side comparison illustrates how spatial resolution directly affects perceived sharpness.

---

### Exercise 4 — Center Crop

```python
h, w = img_ex1_rgb.shape[:2]
crop_h, crop_w = h // 2, w // 2
r1 = (h - crop_h) // 2
c1 = (w - crop_w) // 2
img_crop = img_ex1_rgb[r1 : r1+crop_h, c1 : c1+crop_w]
```

**Output:** The cropped image is `(250, 250, 3)` — the central 50% of the original in both dimensions. For a portrait cat image, this typically isolates the face region.

---

### Exercise 5 — Quantization at 2, 4, and 8 Levels

```python
img_gray_ex = cv2.cvtColor(img_ex1, cv2.COLOR_BGR2GRAY)
for lv in [2, 4, 8]:
    q = quantize_image(img_gray_ex, lv)
    print(f'{lv}-level: {len(np.unique(q))} distinct values → {np.unique(q).tolist()}')
```

**Output:**  
- 2-level: 2 values → [0, 128]  
- 4-level: 4 values → [0, 64, 128, 192]  
- 8-level: 8 values → [0, 32, 64, 96, 128, 160, 192, 224]

The 2-level image shows maximum posterisation. The 8-level result appears much closer to natural grayscale, demonstrating that even modest increases in quantization level greatly improve perceived image quality.

---

## Key Takeaways

- A digital image is a matrix of pixels; colour images have shape `(H, W, 3)` and grayscale images `(H, W)`.  
- Different Python libraries load the same image with different internal representations — understanding channel order and dtype prevents subtle bugs.  
- Interpolation method selection matters: use `INTER_AREA` for downscaling, `INTER_CUBIC` or `INTER_LANCZOS4` for upscaling.  
- Quantization trades image fidelity for bit-depth reduction; 16 levels (4-bit) is nearly visually lossless for natural images.  
- 4-connectivity considers only cardinal neighbors; 8-connectivity adds diagonal neighbors. Boundary checking is essential in both cases.
