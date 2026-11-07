import io
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import nl_dem as nd


def test_terrarium_round_trip():
    h = np.array([[-32.5, 0.0, 0.25], [1.0, 812.75, 2345.5]], dtype=np.float32)
    assert np.allclose(nd.terrarium_decode(nd.terrarium_encode(h)), h, atol=1 / 256)


def test_terrarium_matches_spec():
    # h = R*256 + G + B/256 - 32768: 0 m is (128, 0, 0).
    assert nd.terrarium_encode(np.array([0.0])).tolist() == [[128, 0, 0]]
    assert nd.terrarium_encode(np.array([1.5])).tolist() == [[128, 1, 128]]


def test_webp_is_lossless():
    rng = np.random.default_rng(1)
    h = (rng.random((nd.TILE, nd.TILE)) * 900).astype(np.float32)
    back = nd.decode_image(nd.encode_webp(h))
    assert np.allclose(back, h, atol=1 / 256)


def test_tile_bounds_and_lookup_agree():
    z, lon, lat = 13, -57.78, 49.59
    x, y = nd.lonlat_to_tile(lon, lat, z)
    left, bottom, right, top = nd.tile_bounds(z, x, y)
    mx = lon / 360 * nd.EARTH
    my = math.asinh(math.tan(math.radians(lat))) / (2 * math.pi) * nd.EARTH
    assert left <= mx < right and bottom < my <= top
    assert math.isclose(right - left, nd.EARTH / 2**z)


def test_downsample_averages_children():
    t = nd.TILE
    kids = [np.full((t, t), v, np.float32) for v in (0, 4, 8, 12)]
    p = nd.downsample(kids)
    assert p.shape == (t, t)
    assert p[0, 0] == 0 and p[0, -1] == 4 and p[-1, 0] == 8 and p[-1, -1] == 12


def test_upsample_quadrant_of_a_ramp():
    n = 8
    ramp = np.tile(np.arange(n, dtype=np.float32), (n, 1))  # h = column
    q = nd.upsample_quadrant(ramp, 1, 0)  # the east half
    # Child pixel i sits at parent column n/2 + (i + 0.5)/2 - 0.5.
    expect = np.clip(n / 2 + (np.arange(n) + 0.5) / 2 - 0.5, 0, n - 1)
    assert np.allclose(q[0], expect)


def test_masked_gaussian_ignores_invalid_pixels():
    h = np.full((40, 40), 100.0, np.float32)
    valid = np.ones_like(h, bool)
    valid[:, 20:] = False
    h[~valid] = 0  # the source's "no data"
    out, w = nd.masked_gaussian(h, valid, (2.0, 2.0))
    # The coast isn't dragged towards the 0 next to it.
    assert np.allclose(out[:, :20], 100, atol=1e-3)
    assert w[:, 0].min() > 0.99 and w[:, -1].max() < 0.01
    assert 0.3 < w[20, 19] < 0.7


def test_blend_weight_ramps_smoothly():
    w = np.linspace(0, 1, 101, dtype=np.float32)
    b = nd.blend_weight(w)
    assert b[0] == 0 and b[25] == 0 and b[75] == 1 and b[-1] == 1
    assert np.all(np.diff(b) >= 0)
    assert math.isclose(float(b[50]), 0.5, abs_tol=1e-6)


def _geotiff(path, data, west, north, res):
    import rasterio
    from rasterio.transform import from_origin
    with rasterio.open(path, 'w', driver='GTiff', width=data.shape[1], height=data.shape[0], count=1,
                       dtype='int32', crs='EPSG:4617', transform=from_origin(west, north, res, res),
                       tiled=True, blockxsize=128, blockysize=128) as ds:
        ds.write(data, 1)


class FlatFill:
    """A Mapterhorn stand-in: every tile is flat at `level` metres."""
    def __init__(self, level):
        self.level, self.calls = level, []
    def __call__(self, z, x, y):
        self.calls.append((z, x, y))
        return np.full((nd.TILE, nd.TILE), self.level, np.float32)


@pytest.fixture
def plateau(tmp_path):
    """A ~1.1 km square, 300 m high, in a 0 ("no data") sea, near Gros Morne."""
    res = 0.0000449158
    n = 600
    data = np.zeros((n, n), np.int32)
    data[150:450, 150:450] = 300
    path = str(tmp_path / 'dem.tif')
    _geotiff(path, data, -57.80, 49.60, res)
    return path


def test_render_z13_fills_outside_the_source(plateau):
    src = nd.Source(plateau)
    x, y = nd.lonlat_to_tile(-57.80 + 300 * 0.0000449158, 49.60 - 300 * 0.0000449158, 13)
    fill = FlatFill(-5.0)
    h, has = nd.render_z13(src, fill, 13, x, y)
    assert has and h.shape == (nd.TILE, nd.TILE)
    # The plateau keeps its height; far from it, the tile is Mapterhorn's.
    assert np.isclose(h.max(), 300, atol=0.5)
    assert np.isclose(h.min(), -5, atol=0.5)
    assert fill.calls == [(13, x, y)]


class SeaFill(FlatFill):
    def __init__(self):
        super().__init__(0.0)


def test_build_writes_a_readable_archive(plateau, tmp_path):
    out = str(tmp_path / 'out.pmtiles')
    n = nd.build(plateau, out, zmin=11, zmax=13, jobs=1, filler=SeaFill)
    from pmtiles.reader import Reader, MmapSource
    with open(out, 'rb') as f:
        rd = Reader(MmapSource(f))
        hdr, meta = rd.header(), rd.metadata()
        assert (hdr['min_zoom'], hdr['max_zoom']) == (11, 13)
        assert meta['encoding'] == 'terrarium' and 'Newfoundland' in meta['attribution']
        x, y = nd.lonlat_to_tile(-57.80 + 300 * 0.0000449158, 49.60 - 300 * 0.0000449158, 13)
        h13 = nd.decode_image(rd.get(13, x, y))
        assert np.isclose(h13.max(), 300, atol=0.5)
        h11 = nd.decode_image(rd.get(11, x >> 2, y >> 2))
        assert 0 < h11.max() <= 300.5
        # Only tiles over (or next to) the source are written.
        assert rd.get(13, x + 40, y) is None
    assert n > 0
