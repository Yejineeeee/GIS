# -*- coding: utf-8 -*-
"""국토정보플랫폼 공개 DEM(ERDAS IMG, 90m, EPSG:5179)을 읽는다 (WBS 3.4).

외부 패키지(GDAL 등) 없이 IMG 파일의 압축되지 않은 float32 블록을 직접 읽는다.
여러 도엽을 넣으면 앞에서부터 찾아 값이 있는 도엽을 쓴다.

    from dem import Dem
    d = Dem(["37808.img", "37804.img"])
    d.height(x, y)         # 지면 높이(m), 없으면 None
    d.slope_deg(x, y)      # 주변 3x3칸(270m) 경사(도), 없으면 None
"""
import math
import struct

NODATA_BELOW = -9000.0


class _Tile:
    def __init__(self, path):
        b = open(path, "rb").read()
        u = lambda o: struct.unpack_from("<I", b, o)[0]
        root = u(u(16) + 8)
        ent = {}

        def walk(p):
            while p:
                nxt, _prev, _par, child, data, size = struct.unpack_from("<IIIIIi", b, p)
                ent[b[p + 88:p + 120].split(b"\0")[0].decode()] = (data, size)
                if child:
                    walk(child)
                p = nxt

        walk(root)
        d, _ = ent["Eimg_Layer"]
        self.w, self.h = struct.unpack_from("<ii", b, d)
        d, _ = ent["Eprj_MapInfo"]
        n = struct.unpack_from("<i", b, d)[0]
        ulx, uly = struct.unpack_from("<dd", b, d + 8 + n + 8)   # 왼쪽 위 칸의 중심
        o = d + 8 + n + 8 + 16 + 8 + 16 + 8
        self.cell = struct.unpack_from("<d", b, o)[0] or 90.0
        self.x0 = ulx - self.cell / 2
        self.y0 = uly + self.cell / 2
        d, _ = ent["Edms_State"]
        cnt = struct.unpack_from("<i", b, d + 14)[0]
        blocks = [struct.unpack_from("<hIiHH", b, d + 22 + 14 * i) for i in range(cnt)]
        bw = bh = 128   # IMG 기본 블록 크기
        nbx = math.ceil(self.w / bw)
        self.grid = [[None] * self.w for _ in range(self.h)]
        for bi, (_fc, off, _sz, valid, comp) in enumerate(blocks):
            if not valid:
                continue
            if comp:
                raise ValueError(f"{path}: 압축된 블록은 읽을 수 없음")
            by, bx = divmod(bi, nbx)
            vals = struct.unpack_from("<%df" % (bw * bh), b, off)
            for r in range(bh):
                R = by * bh + r
                if R >= self.h:
                    break
                row = self.grid[R]
                for c in range(bw):
                    C = bx * bw + c
                    if C < self.w:
                        v = vals[r * bw + c]
                        row[C] = v if v > NODATA_BELOW else None

    def rc(self, x, y):
        c = int((x - self.x0) // self.cell)
        r = int((self.y0 - y) // self.cell)
        return r, c

    def value(self, r, c):
        if 0 <= r < self.h and 0 <= c < self.w:
            return self.grid[r][c]
        return None


class Dem:
    def __init__(self, paths):
        self.tiles = [_Tile(p) for p in paths]
        self.cell = self.tiles[0].cell

    def _tile_for(self, x, y):
        for t in self.tiles:
            r, c = t.rc(x, y)
            if t.value(r, c) is not None:
                return t, r, c
        return None, None, None

    def height(self, x, y):
        t, r, c = self._tile_for(x, y)
        return None if t is None else t.value(r, c)

    def slope_deg(self, x, y):
        """추정 좌표가 속한 칸을 가운데로 한 3x3칸의 경사 (Horn 방법)."""
        t, r, c = self._tile_for(x, y)
        if t is None:
            return None
        z = [[t.value(r + i, c + j) for j in (-1, 0, 1)] for i in (-1, 0, 1)]
        if any(v is None for row in z for v in row):
            return None
        s = t.cell
        dzdx = ((z[0][2] + 2 * z[1][2] + z[2][2]) - (z[0][0] + 2 * z[1][0] + z[2][0])) / (8 * s)
        dzdy = ((z[2][0] + 2 * z[2][1] + z[2][2]) - (z[0][0] + 2 * z[0][1] + z[0][2])) / (8 * s)
        return math.degrees(math.atan(math.hypot(dzdx, dzdy)))
