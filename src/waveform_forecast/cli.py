"""`waveform-forecast` -- multi-station continuous forecasting.

    waveform-forecast                     # this listing
    waveform-forecast features --help     # each command has its own flags

Arguments after the command are passed through untouched, and each command is
exactly the underlying module's `main()`, so a recorded command and the real one
cannot diverge. Every command also runs standalone as
`python -m waveform_forecast.features`.
"""
import importlib
import sys

COMMANDS = {
    "features": ("waveform_forecast.features",
                 "aggregate extractor output into one aligned hourly table"),
    "waveforms": ("waveform_forecast.waveforms",
                  "decimate miniSEED archives into an hourly waveform tensor"),
    "train":    ("waveform_forecast.train",
                 "train the multi-station forecaster against its floor"),
}


def usage():
    """Prints the command listing."""
    print("waveform-forecast -- forecasting from several stations at once\n")
    print("usage: waveform-forecast <command> [args...]"
          "   (each command has its own --help)\n")
    for name, (_, summary) in COMMANDS.items():
        print(f"  {name:<9} {summary}")
    print("\nFeatures come from sismokaos-cli or Sismokaos-featureExtract; this")
    print("project consumes their output rather than defining a second version")
    print("of the same features. The labels are catalog_mlp's, unchanged, so a")
    print("waveform result can be read against a catalogue one.")
    print("\nStations are pooled over a presence mask: they do not share a span")
    print("(all three Aegean stations overlap on 80 days of ~700), so an")
    print("intersection-only model would have almost nothing to train on.")
    return 0


def main():
    """Dispatches to one command's `main()`, leaving its arguments untouched."""
    argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help", "help"):
        return usage()
    name = argv[0]
    if name not in COMMANDS:
        print(f"waveform-forecast: unknown command {name!r}", file=sys.stderr)
        print("run `waveform-forecast` for the list", file=sys.stderr)
        return 2
    sys.argv = [f"waveform-forecast {name}"] + argv[1:]
    return importlib.import_module(COMMANDS[name][0]).main() or 0


if __name__ == "__main__":
    sys.exit(main())
