"""Evaluates precomputed transforms on the 3DMatch/3DLoMatch benchmark.

By default, this loads the transforms in ../logdev , which is the directory
used for saving the transforms if you use our test.py.
"""
import argparse
import os
from pathlib import Path

from benchmark.benchmark_predator import benchmark as benchmark_predator
from benchmark.benchmark_3dmatch import benchmark_dgr

parser = argparse.ArgumentParser()
parser.add_argument('--results_dir', type=str, default='../logdev',
                    help='Path to results (estimated transforms)')
parser.add_argument('--benchmark', choices=['3DMatch', '3DLoMatch'],
                    default='3DMatch')
parser.add_argument('--use_dgr', action='store_true',
                    help='If set, will evaluate using DGR success metrics (<15deg, 30cm)')
opt = parser.parse_args()


def evaluate_log_files():
    benchmark_dir = Path('datasets/3dmatch/benchmarks') / opt.benchmark
    results_dir = Path(opt.results_dir) / opt.benchmark

    if opt.use_dgr:
        out = benchmark_dgr(results_dir, benchmark_dir, require_individual_errors=True)
    else:
        out, recall, indiv_errs = benchmark_predator(results_dir, benchmark_dir,
                                                     require_individual_errors=True)
        indiv_errs.to_excel(os.path.join(opt.results_dir, 'individual_errors.xlsx'))

    print(out)


if __name__ == '__main__':
    evaluate_log_files()
