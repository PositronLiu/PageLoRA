"""Rebuild portable experiment reports from a run directory."""
import argparse
from scripts.benchmark_support import summarize

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory')
    args = parser.parse_args()
    rows = summarize(args.directory)
    print(f"Wrote summary.md/json/csv and measurements.csv: {len(rows)} jobs")

if __name__ == '__main__':
    main()
