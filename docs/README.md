# waveform-forecast documentation

Multi-station continuous earthquake forecasting from waveform-derived features,
scored against the same label and the same floor as the catalogue-only model in
`../forecast` (`catalog_mlp`).

The question, in one line: **given several stations' continuous waveforms, does
an M≥4.5 event occur within the next 14 days?** — or, in the other two modes,
**how many days until the next one**, and **will the seismicity rate rise?**

The stations span two tectonic settings (Aegean extensional, Marmara/NAF), so
labels can be region-wide or **region-local**, and one zone can be **held out**
to test whether anything transfers across fault zones.

## Contents

| document | what it covers |
|---|---|
| [architecture.md](architecture.md) | the model: per-station encoder → masked pool → LSTM+attention → head, and why each piece is shaped that way |
| [data-pipeline.md](data-pipeline.md) | `waveform-forecast features`: extractor output → one aligned hourly multi-station table; windowing and standardization |
| [labels-and-splits.md](labels-and-splits.md) | the two labels, the walk-forward folds, the embargo, and the per-hour label-span purge |
| [evaluation.md](evaluation.md) | floors, metrics, and why no number is reported without the bar it had to clear |
| [cli.md](cli.md) | every command and flag, with defaults |
| [performance.md](performance.md) | **measured results** on the four-station archive, per fold, against the floor |
| [runs/](runs/) | the raw console logs those results were read off |

## Module map

```
src/waveform_forecast/
  cli.py         dispatch: `waveform-forecast <command>` → that module's main()
  features.py    CMD. extractor parquet/CSV → one aligned hourly table
  waveforms.py   CMD. miniSEED archives → hourly 5 Hz waveform tensor (--arm raw)
  train.py       CMD. train the ensemble, walk-forward, and score it
  data.py        windowing, per-(station,feature) standardization, Dataset
  regions.py     station coordinates, zones, region-local event loading
  model.py       MaskedStationPool, MultiStationForecaster
  blocks.py      LSTMAttentionBranch, RawWaveformEncoder  (ported unchanged)
  catalog.py     catalogue → hourly labels                (ported unchanged)
  splits.py      walk-forward CV, embargo, label-span purge (ported unchanged)
  metrics.py     binary + regression metric reports       (ported unchanged)
  evaluate.py    the floors, and FoldResult/summarise
  seeding.py     seeds python/numpy/torch together        (ported unchanged)
```

`catalog.py`, `splits.py`, `metrics.py`, `evaluate.py` and `seeding.py` are ports
from `../forecast`/`cnn_earthquake`. That is deliberate: "the same labelling" is
only a meaningful claim if both projects run the same code to produce it.

## Reproducing

```bash
# 1. aggregate the extractors' output onto one hourly clock
waveform-forecast features \
    --station MANT=feats/MANT/MANT_features.parquet \
    --station DEMI=feats/DEMI/DEMI_features.parquet \
    --drop-dev --out hourly.parquet

# 2. classify: does an M>=4.5 event occur in the next 14 days
waveform-forecast train --features hourly.parquet --stations MANT DEMI \
    --catalog-path cnn_earthquake/catalogs/catalog_current.csv \
    --keep-features Z_STA_LTA_Max_max EN_CROSS_CORR_mean \
    --horizon-days 14 --cv-folds 5

# 3. region-local labels, and a fault-zone holdout
waveform-forecast train --features hourly.parquet \
    --stations aegean --test-stations marmara --label-radius-km 150 \
    --catalog-path cnn_earthquake/catalogs/catalog_current.csv \
    --mode rate --rate-threshold 3.0 \
    --keep-features Z_STA_LTA_Max_max EN_CROSS_CORR_mean --cv-folds 5

# 4. the raw arm: build the waveform tensor first, then train on it
waveform-forecast waveforms --station MANT=../mseed/MANT \
    --station DEMI=../mseed/DEMI --rate 5 --out wav5
waveform-forecast train --arm raw --waveforms wav5 --stations aegean \
    --catalog-path cnn_earthquake/catalogs/catalog_current.csv \
    --label-radius-km 150 --seq-hours 8 --batch-size 16 --cv-folds 5

# 5. regress: how many days until the next one
waveform-forecast train --mode regress --features hourly.parquet \
    --stations MANT DEMI \
    --catalog-path cnn_earthquake/catalogs/catalog_current.csv \
    --keep-features Z_STA_LTA_Max_max EN_CROSS_CORR_mean --cv-folds 5

uv run pytest        # 188 tests
```

## Read this expecting a negative

The waveform arms are a documented negative in this project's earlier work: on
the corrected catalogue neither raw-waveform CNNs nor hand-crafted continuous
features beat persistence — 0 of 10 chaos-sweep cells, all three sequence
architectures below floor. Multi-station pooling is a genuine difference from
those runs, and so is the raw arm, which had never actually run before.

Thirteen walk-forward runs later, only two sit above their own floor — and both
by less than their fold spread. The most useful finding is about the model
rather than the data: **the recurrent branch is harmful here**, and deleting it
(214k of 279k parameters) improved every summary statistic. See
[performance.md](performance.md).
