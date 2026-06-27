# -*- coding: utf-8 -*-
"""Clean Korean title redraw, CORRECT 2D OBJ tile mapping.

Replaces the gold katakana スーパーロボット大戦 with gold Korean 슈퍼로봇대전,
keeping the blue banner outline and the red K untouched. Per logo sprite, every
covered pixel is sampled from a TARGET screen image and quantised to the nearest
colour in THAT sprite's own palette, so the multi-palette gradient is preserved.

Outputs _redraw2_preview.png (re-rendered result) and, on --inject, writes the
modified FG tile bytes to _fg_new.bin for archive rebuild.
"""
import struct, sys
import numpy as np
from PIL import Image, ImageFont, ImageDraw

ss = open('_sstate.bin', 'rb').read()
vram = bytearray(open('_vram.bin', 'rb').read())
pal  = open('_pal.bin', 'rb').read()
i = ss.find(b'OAMS'); oam = ss[i+12:i+12+1024]
FG_OFF = 0x90000
FG = vram[FG_OFF:FG_OFF+0x8000]            # mutable OBJ tiles (1024)

def objpal(pn):
    base = 0x200 + pn*32
    out = []
    for k in range(16):
        c = pal[base+k*2] | (pal[base+k*2+1] << 8)
        out.append(((c & 31) << 3, ((c >> 5) & 31) << 3, ((c >> 10) & 31) << 3))
    return out

def nearest(palcols, rgb):                 # nearest non-transparent index (1..15)
    best, bi = 1 << 30, 1
    for k in range(1, 16):
        pr, pg, pb = palcols[k]
        d = (pr-rgb[0])**2 + (pg-rgb[1])**2 + (pb-rgb[2])**2
        if d < best: best, bi = d, k
    return bi

DIM = {(0,0):(1,1),(0,1):(2,2),(0,2):(4,4),(0,3):(8,8),(1,0):(2,1),(1,1):(4,1),
       (1,2):(4,2),(1,3):(8,4),(2,0):(1,2),(2,1):(1,4),(2,2):(2,4),(2,3):(4,8)}

sprites = []
for s in range(128):
    a0, a1, a2 = struct.unpack_from('<HHH', oam, s*8)
    if ((a0 >> 8) & 1) == 0 and ((a0 >> 9) & 1):
        continue
    shape = (a0 >> 14) & 3; size = (a1 >> 14) & 3
    y = a0 & 0xFF; x = a1 & 0x1FF
    if x >= 256: x -= 512
    tile = a2 & 0x3FF; pn = (a2 >> 12) & 0xF
    w, h = DIM.get((shape, size), (1, 1))
    sprites.append(dict(s=s, x=x, y=y, w=w, h=h, tile=tile, pn=pn))
logo = [sp for sp in sprites if sp['y'] < 140]
maxtile = max(sp['tile'] + sp['w']*sp['h'] for sp in logo)
print("logo sprites:", len(logo), " max tile used:", maxtile)

# ---- pixel access with CORRECT 2D mapping (row stride = 32 tiles) ----
def tile_index_2d(sp, tx, ty):
    return sp['tile'] + ty*32 + tx
def get_px(ti, py, px):
    o = ti*32 + py*4 + px//2
    b = FG[o]
    return (b >> 4) if (px & 1) else (b & 0xF)
def set_px(ti, py, px, val):
    o = ti*32 + py*4 + px//2
    b = FG[o]
    if px & 1: b = (b & 0x0F) | (val << 4)
    else:      b = (b & 0xF0) | (val & 0xF)
    FG[o] = b

# ---- render BASE screen (topmost sprite wins) ----
H = W = 256
base = np.zeros((H, W, 3), np.uint8)
owner = np.full((H, W), -1, np.int32)      # index into logo[] that owns each pixel (topmost)
goldmask = np.zeros((H, W), bool)
for li in range(len(logo)-1, -1, -1):      # back to front so spr0 wins
    sp = logo[li]; cols = objpal(sp['pn'])
    for ty in range(sp['h']):
        for tx in range(sp['w']):
            ti = tile_index_2d(sp, tx, ty)
            for py in range(8):
                sy = sp['y'] + ty*8 + py
                if not (0 <= sy < H): continue
                for px in range(8):
                    sx = sp['x'] + tx*8 + px
                    if not (0 <= sx < W): continue
                    v = get_px(ti, py, px)
                    if v == 0: continue
                    base[sy, sx] = cols[v]; owner[sy, sx] = li

# ---- detect gold katakana region (warm bright), EXCLUDE the red K (x>=176) ----
r = base[:, :, 0].astype(int); g = base[:, :, 1].astype(int); b = base[:, :, 2].astype(int)
gold = (r > 120) & (g > 70) & (r - b > 35) & (r >= g - 10)
gold[:, 176:] = False                       # keep K
goldmask = gold
ys, xs = np.where(gold)
print("gold katakana bbox: x %d..%d  y %d..%d  (%d px)" % (xs.min(), xs.max(), ys.min(), ys.max(), len(xs)))
RX0, RX1 = xs.min(), xs.max()
RY0, RY1 = ys.min(), ys.max()

# designed bright gold gradient (top highlight -> bottom deep gold), per row;
# nearest-colour then picks the brightest gold each sprite palette actually has
TOP = (255, 238, 160); BOT = (208, 142, 56)
ramp = {}
for yy in range(RY0, RY1+1):
    t = (yy - RY0) / max(1, RY1 - RY0)
    ramp[yy] = tuple(int(TOP[c] + (BOT[c]-TOP[c]) * t) for c in range(3))
# 8-neighbour dilation helpers
def dilate(m):
    d = m.copy()
    d[1:, :] |= m[:-1, :]; d[:-1, :] |= m[1:, :]
    d[:, 1:] |= m[:, :-1]; d[:, :-1] |= m[:, 1:]
    d[1:, 1:] |= m[:-1, :-1]; d[:-1, :-1] |= m[1:, 1:]
    d[1:, :-1] |= m[:-1, 1:]; d[:-1, 1:] |= m[1:, :-1]
    return d
def dilate_n(m, n):
    for _ in range(n): m = dilate(m)
    return m

inR = np.zeros((H, W), bool); inR[RY0:RY1+1, RX0:RX1+1] = True
# erase region: gold bbox padded slightly, but NEVER into the red K (x<176)
EX0, EX1 = max(8, RX0-2), min(175, RX1+2); EY0, EY1 = max(0, RY0-2), RY1+2
inER = np.zeros((H, W), bool); inER[EY0:EY1+1, EX0:EX1+1] = True
bright = base.max(axis=2).astype(int)
bch = base[:, :, 2].astype(int); rch = base[:, :, 0].astype(int)
# navy plate behind the katakana = dark bluish median inside R
navy_px = inR & (owner >= 0) & (~gold) & (bch >= rch) & (bright < 115)
NAVY = tuple(int(v) for v in np.median(base[navy_px], axis=0)) if navy_px.any() else (16, 16, 40)
# katakana outline colour = dark ring just outside the gold
ring = dilate_n(gold, 3) & (~gold) & (owner >= 0) & (bright < 95)
OUTLINE = tuple(int(v) for v in np.median(base[ring], axis=0)) if ring.any() else (20, 18, 36)
# full katakana silhouette to erase (gold fill + its outline ring), clipped to erase region
erase_full = dilate_n(gold, 3) & inER
print("NAVY plate:", NAVY, " katakana outline:", OUTLINE, " ramp:", ramp[RY0], ramp[RY1],
      " navy_px:", int(navy_px.sum()), " erase:", int(erase_full.sum()))

# ---- render Korean ink mask (fill + outline) fitted to R ----
TEXT = "슈퍼로봇대전"
fnt = ImageFont.truetype("C:/Windows/Fonts/malgunbd.ttf", 96)
tmp = Image.new("L", (1400, 200), 0); dt = ImageDraw.Draw(tmp)
bb = dt.textbbox((0, 0), TEXT, font=fnt)
tw, th = bb[2]-bb[0], bb[3]-bb[1]
glyph = Image.new("L", (tw+8, th+8), 0)
ImageDraw.Draw(glyph).text((4-bb[0], 4-bb[1]), TEXT, font=fnt, fill=255)
# scale to fit R (a touch of inset)
RW, RH = RX1-RX0+1, RY1-RY0+1
sw, sh = int(RW*0.99), int(RH*0.92)
glyph = glyph.resize((sw, sh), Image.LANCZOS)
gnp = np.array(glyph)
ox = RX0 + (RW - sw)//2; oy = RY0 + (RH - sh)//2
fill_mask = np.zeros((H, W), bool)
for yy in range(gnp.shape[0]):
    for xx in range(gnp.shape[1]):
        if gnp[yy, xx] > 90:
            fill_mask[oy+yy, ox+xx] = True
# outline = 2px dilation of the Korean fill, minus the fill itself
outline_mask = dilate_n(fill_mask, 2) & (~fill_mask)

# ---- build TARGET screen ----
target = base.copy()
# erase entire katakana (gold + outline ring) -> navy plate
target[erase_full] = NAVY
# stamp Korean outline then fill
for yy, xx in zip(*np.where(outline_mask)):
    target[yy, xx] = OUTLINE
for yy, xx in zip(*np.where(fill_mask)):
    target[yy, xx] = ramp[min(max(yy, RY0), RY1)]

# ---- write back per sprite ----
korea_ink = fill_mask | outline_mask
erase_here = erase_full & (~korea_ink)
changed = 0
for sp in logo:
    cols = objpal(sp['pn'])
    for ty in range(sp['h']):
        for tx in range(sp['w']):
            ti = tile_index_2d(sp, tx, ty)
            for py in range(8):
                sy = sp['y'] + ty*8 + py
                if not (0 <= sy < H): continue
                for px in range(8):
                    sx = sp['x'] + tx*8 + px
                    if not (0 <= sx < W): continue
                    if korea_ink[sy, sx]:                      # draw Korean
                        cur = get_px(ti, py, px)
                        nv = nearest(cols, tuple(int(c) for c in target[sy, sx]))
                        if nv != cur: set_px(ti, py, px, nv); changed += 1
                    elif erase_here[sy, sx]:                   # wipe katakana -> navy
                        cur = get_px(ti, py, px)
                        if cur != 0:
                            nv = nearest(cols, NAVY)
                            if nv != cur: set_px(ti, py, px, nv); changed += 1
print("pixels changed:", changed)

# ---- re-render preview with the modified FG (2D) ----
prev = np.zeros((H, W, 3), np.uint8); prev[:] = (8, 8, 20)
for li in range(len(logo)-1, -1, -1):
    sp = logo[li]; cols = objpal(sp['pn'])
    for ty in range(sp['h']):
        for tx in range(sp['w']):
            ti = tile_index_2d(sp, tx, ty)
            for py in range(8):
                sy = sp['y'] + ty*8 + py
                if not (0 <= sy < H): continue
                for px in range(8):
                    sx = sp['x'] + tx*8 + px
                    if not (0 <= sx < W): continue
                    v = get_px(ti, py, px)
                    if v: prev[sy, sx] = cols[v]
Image.fromarray(prev).resize((512, 448), Image.NEAREST).save('_redraw2_preview.png')
print("saved _redraw2_preview.png")

if '--inject' in sys.argv:
    open('_fg_new.bin', 'wb').write(bytes(FG))
    print("wrote _fg_new.bin (", len(FG), "bytes )")
