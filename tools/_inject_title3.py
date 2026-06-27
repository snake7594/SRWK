# -*- coding: utf-8 -*-
"""Re-encode the CORRECT 2D logo redraw (_fg_new.bin[0:0x7000] = new block 2360
pixels, verified verbatim==VRAM) and rebuild add02dat.bin -> kr/add02_patched.bin."""
import struct
from srwk_rom import Rom
from _codec1024 import decomp_1024
from _enc1024 import build_ecd
from _inject_title import rebuild_archive, archive_blocks_raw

KR_ROM = '../Super Robot Wars K (Korean)-기존패치.nds'
pixb = open('_fg_new.bin', 'rb').read()[0:0x7000]      # new block 2360 pixel bytes
print('new pixel bytes:', len(pixb))

rom = Rom(KR_ROM)
d = rom.get('data/add02dat.bin')
ne, offs = archive_blocks_raw(d)
orig = d[offs[2360]:offs[2361]]
f3 = struct.unpack_from('>I', orig, 12)[0]
assert len(pixb) == f3 - 8, (len(pixb), f3 - 8)

new_block = build_ecd(orig, pixb)
dec = decomp_1024(new_block)[0][8:8+(f3-8)]
assert dec == pixb, "re-encode round-trip FAILED"
print('re-encode OK: %d B (orig %d, delta %+d)' % (len(new_block), len(orig), len(new_block)-len(orig)))

new_arc, _ = rebuild_archive(d, {2360: new_block})
ne2, offs2 = archive_blocks_raw(new_arc)
dec2 = decomp_1024(new_arc[offs2[2360]:offs2[2361]])[0][8:8+(f3-8)]
assert dec2 == pixb, "archive decode mismatch"
open('kr/add02_patched.bin', 'wb').write(new_arc)
print('saved kr/add02_patched.bin (%d B, orig %d, delta %+d); verify OK'
      % (len(new_arc), len(d), len(new_arc)-len(d)))
