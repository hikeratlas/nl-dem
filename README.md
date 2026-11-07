# nl-dem

Republish Newfoundland and Labrador's 5 m DEM in a MapLibre-friendly format: [Terrarium](https://github.com/tilezen/joerd/blob/master/docs/formats.md#terrarium)-encoded 512 px WebP tiles in a [PMTiles](https://github.com/protomaps/PMTiles) archive.

The tiles use the same encoding and tile grid as [Mapterhorn](https://mapterhorn.com/), so a map can use them instead of Mapterhorn's over the island of Newfoundland. There, Mapterhorn only has 30 m Copernicus data, and its z14+ tiles are blurrier than its z13 ([hikeratlas/basemap#181](https://github.com/hikeratlas/basemap/issues/181)).

[Hiker Atlas](https://www.hikeratlas.com/map) uses it for hillshade and contours ([hikeratlas/basemap](https://github.com/hikeratlas/basemap)).

## The source

[Newfoundland and Labrador 5m Digital Elevation Model](https://geohub-gnl.hub.arcgis.com/datasets/newfoundland-and-labrador-5m-digital-elevation-model), from the Government of Newfoundland and Labrador:
- Island of Newfoundland only; no Labrador.
- One GeoTIFF of whole-metre heights (int32), 172,110 × 116,527 px at ~0.0000449° (~5 m), in NAD83(CSRS) geographic coordinates (EPSG:4617).
- Inside a 4 GB zip (`NL_DEM_5m_NAD83_CSRS.zip`).
- **0 means no data:** sea, and the parts of Labrador, Quebec and Saint-Pierre inside its bounds.
- The province describes it as built from 1:30,000 air photos (mass points on a 50 m grid, plus breaklines and drainage), ±2 m.

Against HRDEM lidar near Corner Brook it has an RMSE of 1.75 m, against 4.4 m for Copernicus GLO-30 and 5.6 m for CDEM. Near Grand Falls it's 2.1 m, against 2.8 m and 5.5 m. See [basemap#181](https://github.com/hikeratlas/basemap/issues/181#issuecomment-6711024224).

## What the build does

- **Smooths** heights with a Gaussian (σ = 7.5 m, i.e. 1.5 source pixels north–south), counting only valid pixels.
  - This removes the 1 m terracing that whole-metre heights leave on gentle slopes, while keeping cliffs and ridges.
  - Coasts aren't dragged down towards the 0 beside them.
- **Fills** where the source has no data with Mapterhorn's heights for the same tile, feathered across the edge (`blend_weight`).
  - Every tile is complete, and the archive's edges, which are open sea, match Mapterhorn's tiles.
  - Using these tiles inside the archive's coverage and Mapterhorn's outside it gives no seams.
- **z13** (~6 m/px at this latitude, close to the source's 5 m) is rendered from the source. **z12–z10** average their four children; a child with no source data is Mapterhorn's tile. Past z13, overzoom.
- Writes tiles only over the island, padded by one tile. A missing tile means "use Mapterhorn".

The whole island is 16,891 tiles (3.4 GB). That took 18 minutes on 4 cores, plus about 2 minutes to fetch the source.

## Usage

Python 3.10 or later:

```sh
python3 -m venv venv && venv/bin/pip install -r requirements.txt

# Stream DEM_5m.tif (3.6 GB) and its .ovr overviews (0.9 GB) out of the
# province's zip, without storing the zip.
venv/bin/python nl_dem.py fetch DEM_5m.tif

venv/bin/python nl_dem.py build DEM_5m.tif nl-dem.pmtiles
```

`build` options:
- `--bbox=W,S,E,N` builds only the tiles over part of the island, e.g. `--bbox=-58.0,49.4,-57.6,49.7` for Gros Morne.
- `--zmin` / `--zmax` change the zoom range.
- `--jobs` sets the number of worker processes.

The archive's metadata carries `encoding: terrarium`, `tileSize: 512` and the attribution.

## Publishing

[`.github/workflows/publish.yml`](.github/workflows/publish.yml) runs `fetch` and `build` on a GitHub runner and uploads the archive to Hiker Atlas's R2 bucket, as https://public.hikeratlas.com/nl-dem/nl-dem.pmtiles.
- It runs by hand (Actions → Publish → Run workflow), since the province rarely updates the DEM.
- It needs the `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` secrets: R2 keys with write access to `hikes-public`.

## Tests

```sh
venv/bin/pip install pytest && venv/bin/pytest -q tests
```

## Licence

Code: MIT (see [LICENSE](LICENSE)).

Output: contains information licensed under the [Open Government Licence – Newfoundland and Labrador](https://www.gov.nl.ca/exec/tbs/working-with-us/open-government-licence/), with fill from [Mapterhorn](https://mapterhorn.com/attribution) at the edges. Credit both, e.g. "Contains information licensed under the Open Government Licence – Newfoundland and Labrador; © Mapterhorn".
