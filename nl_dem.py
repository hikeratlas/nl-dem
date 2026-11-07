#!/usr/bin/env python3
"""Republish Newfoundland and Labrador's 5 m DEM as Terrarium tiles.

The source is the province's island-wide 5 m DEM (NL Open Government
Licence): one GeoTIFF of whole-metre heights in NAD83(CSRS) geographic
coordinates, inside a 4 GB zip. Outside the island (sea, Labrador, Quebec,
Saint-Pierre) it holds 0, which is "no data" here, not an elevation.

The output is a PMTiles archive of 512 px Terrarium-encoded WebP tiles on the
web-mercator grid, z10-z13 by default, the same encoding as Mapterhorn
(https://mapterhorn.com/), so a client can use these tiles instead of
Mapterhorn's wherever they exist:

- Heights are smoothed with a Gaussian (sigma 7.5 m), counting only valid
  pixels, to remove the terracing from whole-metre heights without dragging
  coasts towards the 0 around them (see hikeratlas/basemap#181).
- Where the source has no data, a tile takes Mapterhorn's heights for the same
  tile, feathered across the edge, so every tile is complete and seamless
  next to Mapterhorn's.
- z13 tiles (~6 m/px at this latitude, close to the source's 5 m) are rendered
  from the source; lower zooms average their four children.

Usage:
    nl_dem.py fetch DEM_5m.tif            # stream the GeoTIFF out of the zip
    nl_dem.py build DEM_5m.tif nl-dem.pmtiles [--bbox W,S,E,N] [--jobs N]
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import struct
import sys
import tempfile
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache

import numpy as np

TILE = 512
ZMAX = 13
ZMIN = 10
SIGMA_M = 7.5
SOURCE_ZIP = 'https://www.arcgis.com/sharing/rest/content/items/7203aef91a024ece843c4be4b3727dda/data'
SOURCE_MEMBER = 'DEM/DEM_5m_NAD83_CSRS.tif'
FILL_URL = 'https://tiles.mapterhorn.com/{z}/{x}/{y}.webp'
ATTRIBUTION = (
    "Contains information licensed under the Open Government Licence – Newfoundland and Labrador; "
    "<a href='https://mapterhorn.com/attribution'>© Mapterhorn</a>"
)
EARTH = 2 * math.pi * 6378137.0  # web-mercator world width, metres


# --- Terrarium ---------------------------------------------------------------

def terrarium_encode(h: np.ndarray) -> np.ndarray:
    """Heights (metres) to Terrarium RGB: h = R*256 + G + B/256 - 32768."""
    v = np.clip(np.asarray(h, dtype=np.float64) + 32768.0, 0, 65535.996)
    r = np.floor(v / 256)
    g = np.floor(v) - r * 256
    b = np.floor((v - np.floor(v)) * 256)
    return np.stack([r, g, b], axis=-1).astype(np.uint8)


def terrarium_decode(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float64)
    return (rgb[..., 0] * 256 + rgb[..., 1] + rgb[..., 2] / 256 - 32768).astype(np.float32)


def encode_webp(h: np.ndarray) -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(terrarium_encode(h), 'RGB').save(buf, 'WEBP', lossless=True, quality=50, method=4)
    return buf.getvalue()


def decode_image(data: bytes) -> np.ndarray:
    from PIL import Image
    return terrarium_decode(np.asarray(Image.open(io.BytesIO(data)).convert('RGB')))


# --- Tile maths ----------------------------------------------------------------

def tile_bounds(z: int, x: int, y: int) -> tuple[float, float, float, float]:
    """A tile's (left, bottom, right, top) in EPSG:3857 metres."""
    size = EARTH / 2**z
    left = -EARTH / 2 + x * size
    top = EARTH / 2 - y * size
    return left, top - size, left + size, top


def lonlat_to_tile(lon: float, lat: float, z: int) -> tuple[int, int]:
    n = 2**z
    x = int((lon + 180) / 360 * n)
    y = int((1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n)
    return min(max(x, 0), n - 1), min(max(y, 0), n - 1)


def downsample(children: list[np.ndarray]) -> np.ndarray:
    """The parent of four child tiles [nw, ne, sw, se], by 2x2 averaging."""
    nw, ne, sw, se = children
    m = np.block([[nw, ne], [sw, se]])
    return m.reshape(TILE, 2, TILE, 2).mean(axis=(1, 3)).astype(np.float32)


def upsample_quadrant(parent: np.ndarray, qx: int, qy: int) -> np.ndarray:
    """The (qx, qy) quadrant of a tile's heights, bilinearly upsampled 2x."""
    n = parent.shape[0]
    c = (np.arange(n) + 0.5) / 2 - 0.5
    sx, sy = qx * n / 2 + c, qy * n / 2 + c
    x0 = np.clip(np.floor(sx).astype(int), 0, n - 1); x1 = np.clip(x0 + 1, 0, n - 1)
    y0 = np.clip(np.floor(sy).astype(int), 0, n - 1); y1 = np.clip(y0 + 1, 0, n - 1)
    fx = np.clip(sx - np.floor(sx), 0, 1)[None, :]; fy = np.clip(sy - np.floor(sy), 0, 1)[:, None]
    p = parent
    top = p[np.ix_(y0, x0)] * (1 - fx) + p[np.ix_(y0, x1)] * fx
    bot = p[np.ix_(y1, x0)] * (1 - fx) + p[np.ix_(y1, x1)] * fx
    return (top * (1 - fy) + bot * fy).astype(np.float32)


# --- Smoothing and blending ------------------------------------------------------

def masked_gaussian(h: np.ndarray, valid: np.ndarray, sigma: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian-smooth h counting only valid pixels.

    Returns the smoothed heights and the smoothed validity (0-1), which says
    how much of each pixel's neighbourhood had data. Where it is 0, the
    heights are 0 and meaningless.
    """
    from scipy.ndimage import gaussian_filter
    m = valid.astype(np.float32)
    num = gaussian_filter(np.where(valid, h, 0).astype(np.float32), sigma, mode='nearest')
    den = gaussian_filter(m, sigma, mode='nearest')
    out = np.divide(num, den, out=np.zeros_like(num), where=den > 1e-3)
    return out, den


def blend_weight(w: np.ndarray) -> np.ndarray:
    """Feather from fill (0) to source (1) as the smoothed validity rises.

    A coastline sits at w ~ 0.5; ramping over 0.25-0.75 keeps the source's
    heights a sigma inland and fades to the fill a sigma out to sea.
    """
    t = np.clip((w - 0.25) / 0.5, 0, 1)
    return (t * t * (3 - 2 * t)).astype(np.float32)


# --- Fill from Mapterhorn ----------------------------------------------------------

class Filler:
    """Mapterhorn's heights for a tile, from its nearest ancestor if it 404s."""

    def __init__(self, url: str = FILL_URL):
        import requests
        self.url = url
        self.s = requests.Session()
        self.cache: dict[tuple[int, int, int], np.ndarray | None] = {}

    def __call__(self, z: int, x: int, y: int) -> np.ndarray:
        key = (z, x, y)
        if key not in self.cache:
            got = None
            for attempt in range(4):
                try:
                    r = self.s.get(self.url.format(z=z, x=x, y=y), timeout=60)
                    if r.status_code == 404 and z > 0:
                        got = upsample_quadrant(self(z - 1, x >> 1, y >> 1), x & 1, y & 1)
                    else:
                        r.raise_for_status()
                        got = decode_image(r.content)
                    break
                except Exception:
                    if attempt == 3:
                        raise
                    time.sleep(2 ** attempt)
            if len(self.cache) > 256:
                self.cache.pop(next(iter(self.cache)))
            self.cache[key] = got
        return self.cache[key]


# --- Rendering ---------------------------------------------------------------------

class Source:
    """The NL GeoTIFF, read a window at a time."""

    def __init__(self, path: str):
        import rasterio
        self.ds = rasterio.open(path)

    def render(self, z: int, x: int, y: int) -> tuple[np.ndarray, np.ndarray]:
        """A tile's smoothed heights and blend weight (0 = no source data)."""
        from rasterio.warp import reproject, transform_bounds
        from rasterio.windows import from_bounds
        from rasterio.transform import from_bounds as tr_from_bounds
        from rasterio.enums import Resampling
        ds = self.ds
        left, bottom, right, top = tile_bounds(z, x, y)
        w, s, e, n = transform_bounds('EPSG:3857', ds.crs, left, bottom, right, top)
        lat = (s + n) / 2
        px_y = abs(ds.res[1]) * 111320.0
        px_x = abs(ds.res[0]) * 111320.0 * math.cos(math.radians(lat))
        sigma = (SIGMA_M / px_y, SIGMA_M / px_x)
        pad = 4 * max(sigma) * max(abs(ds.res[0]), abs(ds.res[1]))
        win = from_bounds(w - pad, s - pad, e + pad, n + pad, ds.transform).round_offsets().round_lengths()
        raw = ds.read(1, window=win, boundless=True, fill_value=0).astype(np.float32)
        valid = raw != 0
        if not valid.any():
            return np.zeros((TILE, TILE), np.float32), np.zeros((TILE, TILE), np.float32)
        h, wgt = masked_gaussian(raw, valid, sigma)
        src_tr = ds.window_transform(win)
        dst_tr = tr_from_bounds(left, bottom, right, top, TILE, TILE)
        out_h = np.zeros((TILE, TILE), np.float32)
        out_w = np.zeros((TILE, TILE), np.float32)
        for a, b in ((h, out_h), (wgt, out_w)):
            reproject(a, b, src_transform=src_tr, src_crs=ds.crs, dst_transform=dst_tr, dst_crs='EPSG:3857',
                      resampling=Resampling.bilinear)
        return out_h, blend_weight(out_w)


def render_z13(src: Source, fill: Filler, z: int, x: int, y: int) -> tuple[np.ndarray, bool]:
    """A complete max-zoom tile, and whether it has any source data."""
    h, w = src.render(z, x, y)
    if not (w > 0).any():
        return fill(z, x, y), False
    if (w < 1).any():
        h = w * h + (1 - w) * fill(z, x, y)
    return h.astype(np.float32), True


# --- Tile set ------------------------------------------------------------------------

def coverage_tiles(path: str, z: int, bbox: tuple[float, float, float, float] | None = None,
                   factor: int = 32) -> set[tuple[int, int]]:
    """Tiles at zoom z that touch source data, padded by one tile.

    Reads the GeoTIFF decimated by `factor` (its .ovr overviews make this
    quick; without them it reads every pixel once).
    """
    import rasterio
    from rasterio.enums import Resampling
    with rasterio.open(path) as ds:
        a = ds.read(1, out_shape=(ds.height // factor, ds.width // factor), resampling=Resampling.nearest)
        tr = ds.transform @ ds.transform.scale(ds.width / a.shape[1], ds.height / a.shape[0])
    rows, cols = np.nonzero(a != 0)
    lons = tr.c + (cols + 0.5) * tr.a
    lats = tr.f + (rows + 0.5) * tr.e
    if bbox:
        keep = (lons >= bbox[0]) & (lons <= bbox[2]) & (lats >= bbox[1]) & (lats <= bbox[3])
        lons, lats = lons[keep], lats[keep]
    found = {lonlat_to_tile(lo, la, z) for lo, la in zip(lons, lats)}
    return {(x + dx, y + dy) for x, y in found for dx in (-1, 0, 1) for dy in (-1, 0, 1)}


# --- Build -------------------------------------------------------------------------

_worker: dict = {}


def _init(path: str, outdir: str, zmin: int, zmax: int, wanted: set[tuple[int, int]], filler=None):
    _worker.update(src=Source(path), fill=(filler or Filler)(), outdir=outdir, zmin=zmin, zmax=zmax, wanted=wanted)


def _build_root(root: tuple[int, int]) -> list[tuple[int, int, int, str]]:
    """Render the pyramid under one zmin tile; returns written (z, x, y, path)."""
    src, fill, outdir = _worker['src'], _worker['fill'], _worker['outdir']
    zmin, zmax, wanted = _worker['zmin'], _worker['zmax'], _worker['wanted']
    written = []

    # Returns (heights, has source data). Tiles without source data return
    # None heights: a parent with some data fetches Mapterhorn's for them.
    def tile(z: int, x: int, y: int) -> tuple[np.ndarray | None, bool]:
        if z == zmax:
            if (x, y) not in wanted:
                return None, False
            h, has = render_z13(src, fill, z, x, y)
            if not has:
                return None, False
        else:
            xys = [(2 * x + dx, 2 * y + dy) for dy in (0, 1) for dx in (0, 1)]
            kids = [tile(z + 1, cx, cy) for cx, cy in xys]
            if not any(k[1] for k in kids):
                return None, False
            h = downsample([k[0] if k[1] else fill(z + 1, cx, cy) for k, (cx, cy) in zip(kids, xys)])
        p = os.path.join(outdir, f'{z}-{x}-{y}.webp')
        with open(p, 'wb') as f:
            f.write(encode_webp(h))
        written.append((z, x, y, p))
        return h, True

    tile(zmin, *root)
    return written


def build(path: str, out: str, zmin: int = ZMIN, zmax: int = ZMAX, bbox=None, jobs: int | None = None,
          filler=None) -> int:
    """Build the archive; `filler` (default Filler) makes the fill source, per worker."""
    from pmtiles.tile import zxy_to_tileid, TileType, Compression
    from pmtiles.writer import Writer
    t0 = time.time()
    wanted = coverage_tiles(path, zmax, bbox)
    roots = sorted({(x >> (zmax - zmin), y >> (zmax - zmin)) for x, y in wanted})
    print(f'{len(wanted)} z{zmax} tiles under {len(roots)} z{zmin} roots', flush=True)
    written = []
    with tempfile.TemporaryDirectory(dir=os.path.dirname(os.path.abspath(out))) as tmp:
        args = (path, tmp, zmin, zmax, wanted, filler)
        if jobs == 1:
            _init(*args)
            results = map(_build_root, roots)
            ex = None
        else:
            ex = ProcessPoolExecutor(jobs, initializer=_init, initargs=args)
            results = ex.map(_build_root, roots)
        try:
            for i, w in enumerate(results):
                written += w
                if (i + 1) % 10 == 0 or i + 1 == len(roots):
                    print(f'{i + 1}/{len(roots)} roots, {len(written)} tiles, {time.time() - t0:.0f}s', flush=True)
        finally:
            if ex:
                ex.shutdown()
            elif _worker:
                _worker['src'].ds.close()
                _worker.clear()
        written.sort(key=lambda t: zxy_to_tileid(t[0], t[1], t[2]))
        lons, lats = [], []
        with open(out, 'wb') as f:
            wr = Writer(f)
            for z, x, y, p in written:
                with open(p, 'rb') as g:
                    wr.write_tile(zxy_to_tileid(z, x, y), g.read())
                if z == zmax:
                    l, b, r, t = tile_bounds(z, x, y)
                    lons += [l, r]; lats += [b, t]
            to_ll = lambda mx, my: (mx / EARTH * 360, math.degrees(math.atan(math.sinh(my / EARTH * 2 * math.pi))))
            w_, s_ = to_ll(min(lons), min(lats)); e_, n_ = to_ll(max(lons), max(lats))
            header = {
                'tile_type': TileType.WEBP, 'tile_compression': Compression.NONE,
                'min_zoom': zmin, 'max_zoom': zmax,
                'min_lon_e7': int(w_ * 1e7), 'min_lat_e7': int(s_ * 1e7),
                'max_lon_e7': int(e_ * 1e7), 'max_lat_e7': int(n_ * 1e7),
                'center_zoom': zmin, 'center_lon_e7': int((w_ + e_) / 2 * 1e7), 'center_lat_e7': int((s_ + n_) / 2 * 1e7),
            }
            wr.finalize(header, {
                'name': 'nl-dem', 'encoding': 'terrarium', 'tileSize': TILE,
                'attribution': ATTRIBUTION,
                'description': "Newfoundland's 5 m DEM, smoothed, as Terrarium tiles; Mapterhorn outside it.",
                'source': SOURCE_ZIP, 'sigma_m': SIGMA_M,
            })
    print(f'wrote {out}: {len(written)} tiles, {os.path.getsize(out) / 1e6:.0f} MB, {time.time() - t0:.0f}s', flush=True)
    return len(written)


# --- Fetch ----------------------------------------------------------------------------

def fetch(out: str, member: str = SOURCE_MEMBER, url: str = SOURCE_ZIP) -> None:
    """Stream one member out of the remote zip, inflating as it downloads.

    The zip is 4 GB and serves byte ranges, so read its central directory with
    ranged requests, then make one ranged GET for the member's data.
    """
    import requests
    import zipfile

    class Remote(io.RawIOBase):
        def __init__(self):
            self.s = requests.Session()
            r = self.s.get(url, headers={'Range': 'bytes=0-0'})
            r.raise_for_status()
            self.url, self.size, self.pos = r.url, int(r.headers['Content-Range'].split('/')[1]), 0
        def readable(self): return True
        def seekable(self): return True
        def tell(self): return self.pos
        def seek(self, off, whence=0):
            self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
            return self.pos
        def readinto(self, b):
            if self.pos >= self.size: return 0
            end = min(self.size, self.pos + len(b)) - 1
            d = self.s.get(self.url, headers={'Range': f'bytes={self.pos}-{end}'}).content
            b[:len(d)] = d; self.pos += len(d)
            return len(d)

    remote = Remote()
    info = zipfile.ZipFile(io.BufferedReader(remote, 1 << 20)).getinfo(member)
    hdr = remote.s.get(remote.url, headers={'Range': f'bytes={info.header_offset}-{info.header_offset + 29}'}).content
    n, m = struct.unpack('<HH', hdr[26:30])
    start = info.header_offset + 30 + n + m
    if info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
        sys.exit(f'unsupported compression {info.compress_type}')
    d = zlib.decompressobj(-15) if info.compress_type == zipfile.ZIP_DEFLATED else None
    t0, got, crc = time.time(), 0, 0
    with remote.s.get(remote.url, headers={'Range': f'bytes={start}-{start + info.compress_size - 1}'}, stream=True) as r, \
            open(out + '.part', 'wb') as f:
        r.raise_for_status()
        for chunk in r.iter_content(1 << 22):
            data = d.decompress(chunk) if d else chunk
            crc = zlib.crc32(data, crc); f.write(data); got += len(chunk)
        if d:
            tail = d.flush(); crc = zlib.crc32(tail, crc); f.write(tail)
    if crc != info.CRC:
        sys.exit(f'CRC mismatch for {member}')
    os.replace(out + '.part', out)
    print(f'fetched {member}: {os.path.getsize(out) / 1e6:.0f} MB in {time.time() - t0:.0f}s', flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    f = sub.add_parser('fetch', help='stream the GeoTIFF (and its .ovr overviews) out of the source zip')
    f.add_argument('out')
    b = sub.add_parser('build', help='build the PMTiles archive')
    b.add_argument('src'); b.add_argument('out')
    b.add_argument('--zmin', type=int, default=ZMIN); b.add_argument('--zmax', type=int, default=ZMAX)
    b.add_argument('--bbox', help='W,S,E,N in degrees: only tiles here (for testing)')
    b.add_argument('--jobs', type=int)
    a = p.parse_args(argv)
    if a.cmd == 'fetch':
        fetch(a.out)
        fetch(a.out + '.ovr', SOURCE_MEMBER + '.ovr')
    else:
        bbox = tuple(map(float, a.bbox.split(','))) if a.bbox else None
        build(a.src, a.out, a.zmin, a.zmax, bbox, a.jobs)


if __name__ == '__main__':
    main()
