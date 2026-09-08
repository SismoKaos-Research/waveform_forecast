"""Multi-station continuous forecasting from waveform-derived features.

Predicts the same label `../forecast`'s catalog_mlp predicts -- does an
M>=threshold event occur within the horizon -- from several stations' waveforms
instead of from the catalogue. `waveform_forecast.cli.main` is the entry point.
"""
