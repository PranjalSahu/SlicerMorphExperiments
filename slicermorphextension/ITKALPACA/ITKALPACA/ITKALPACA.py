#!/usr/bin/env python-real

import sys

import script_FPFH_RANSAC_Deform as alpaca

def main(input, sigma, output):
    alpaca.process(input, output)

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: ITKALPACA <input> <sigma> <output>")
        sys.exit(1)
    main(sys.argv[1], float(sys.argv[2]), sys.argv[3])
