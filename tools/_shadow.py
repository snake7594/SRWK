# -*- coding: utf-8 -*-
"""Turn the behind text layer (pal3 = the 3D extrusion) into a clean uniform dark
shadow, so it reads as depth instead of a dim ghost. Renders a preview."""
import struct, sys
import numpy as np
from PIL import Image
ss = open('_sstate.bin', 'rb').read(); pal = open('_pal.bin', 'rb').read()
i = ss.find(b'OAMS'); oam = ss[i+12:i+12+1024]
FG = bytearray(open('_fg_new.bin', 'rb').read())
def palcol(pn):
    base=0x200+pn*32; out=[]
    for k in range(16):
        c=pal[base+k*2]|(pal[base+k*2+1]<<8); out.append(((c&31)<<3,((c>>5)&31)<<3,((c>>10)&31)<<3))
    return out
DIM={(0,0):(1,1),(0,1):(2,2),(0,2):(4,4),(0,3):(8,8),(1,0):(2,1),(1,1):(4,1),
     (1,2):(4,2),(1,3):(8,4),(2,0):(1,2),(2,1):(1,4),(2,2):(2,4),(2,3):(4,8)}
sprites=[]
for s in range(128):
    a0,a1,a2=struct.unpack_from('<HHH',oam,s*8)
    if ((a0>>8)&1)==0 and ((a0>>9)&1): continue
    shape=(a0>>14)&3; size=(a1>>14)&3; y=a0&0xFF; x=a1&0x1FF
    if x>=256: x-=512
    tile=a2&0x3FF; pn=(a2>>12)&0xF; prio=(a2>>10)&3; w,h=DIM.get((shape,size),(1,1))
    if y<140: sprites.append(dict(s=s,x=x,y=y,w=w,h=h,tile=tile,pn=pn,prio=prio))

SHADOW_PAL = int(sys.argv[1]) if len(sys.argv) > 1 else 3
# darkest non-transparent index in that palette = the shadow colour
cols = palcol(SHADOW_PAL)
lum = [(cols[k][0]*299+cols[k][1]*587+cols[k][2]*114) for k in range(16)]
dark_index = min(range(1, 16), key=lambda k: lum[k])
print('pal%d darkest index=%d rgb=%s' % (SHADOW_PAL, dark_index, cols[dark_index]))

changed = 0
for sp in sprites:
    if sp['pn'] != SHADOW_PAL: continue
    for ty in range(sp['h']):
        for tx in range(sp['w']):
            ti = sp['tile'] + ty*32 + tx
            for py in range(8):
                for px in range(8):
                    o = ti*32 + py*4 + px//2
                    v = (FG[o] >> 4) if (px & 1) else (FG[o] & 0xF)
                    if v != 0 and v != dark_index:
                        if px & 1: FG[o] = (FG[o] & 0x0F) | (dark_index << 4)
                        else: FG[o] = (FG[o] & 0xF0) | dark_index
                        changed += 1
print('pixels set to shadow:', changed)
open('_fg_shadow.bin', 'wb').write(bytes(FG))

# render priority-correct
cv = np.zeros((160, 256, 3), np.uint8); cv[:] = (10, 12, 28)
for sp in sorted(sprites, key=lambda q: (q['prio'], -q['s'])):
    c = palcol(sp['pn'])
    for ty in range(sp['h']):
        for tx in range(sp['w']):
            ti = sp['tile'] + ty*32 + tx
            for py in range(8):
                sy = sp['y']+ty*8+py
                if not (0 <= sy < 160): continue
                for px in range(8):
                    sx = sp['x']+tx*8+px
                    if not (0 <= sx < 256): continue
                    o = ti*32+py*4+px//2; v=(FG[o]>>4) if(px&1) else (FG[o]&0xF)
                    if v: cv[sy, sx] = c[v]
Image.fromarray(cv).resize((768, 480), Image.NEAREST).save('_shadow_preview.png')
print('saved _shadow_preview.png + _fg_shadow.bin')
