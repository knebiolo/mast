"""
Run unsupervised overlap removal against a MAST HDF5 project database.

Usage (PowerShell):
    python scripts\run_overlap.py --db "C:\path\to\project.h5" --nodes R1,R2 --edges R1:R2 --method posterior --confidence 0.1

This script expects the project HDF5 to contain the usual keys:
- /presence (columns: freq_code, epoch, time_stamp, power, rec_id, bout_no)
- /classified or /recaptures (columns: freq_code, epoch, time_stamp, power, rec_id, posterior_T)

The script will run the overlap_reduction and write results to the project's /overlapping key.
"""
import argparse
import logging
import sys
import os
from pprint import pprint

# Ensure repo root is on sys.path so the local `pymast` package can be imported
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from pymast.overlap_removal import overlap_reduction


class DummyProject:
    def __init__(self, db_path):
        self.db = db_path


def parse_edges(edges_str):
    # edges_str: R1:R2,R2:R3
    pairs = []
    if not edges_str:
        return pairs
    for token in edges_str.split(','):
        token = token.strip()
        if ':' not in token:
            continue
        a, b = token.split(':', 1)
        pairs.append((a.strip(), b.strip()))
    return pairs


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--db', required=True, help='Path to project HDF5 database')
    p.add_argument('--nodes', required=False, help='Comma-separated list of node rec_ids (e.g. R1,R2,R3). If omitted, caller should provide edges.')
    p.add_argument('--edges', required=False, help='Comma-separated parent:child pairs (e.g. R1:R2,R2:R3). If omitted, all pairwise edges from nodes will be used.')
    p.add_argument('--method', choices=['posterior','power'], default='posterior')
    p.add_argument('--confidence', type=float, default=0.03, help='Confidence threshold (smaller = more aggressive)')
    p.add_argument('--power-threshold', type=float, default=0.2)
    p.add_argument('--min-detections', type=int, default=1, help='Minimum detections required on each receiver to make a decision')
    p.add_argument('--log-level', default='INFO')
    args = p.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format='%(asctime)s %(levelname)s %(message)s')

    nodes = []
    edges = []
    if args.nodes:
        nodes = [n.strip() for n in args.nodes.split(',') if n.strip()]
    if args.edges:
        edges = parse_edges(args.edges)

    # If edges not provided but nodes were, create all pairwise non-self edges
    if not edges and nodes:
        edges = [(i, j) for i in nodes for j in nodes if i != j]

    logging.info('Nodes: %s', nodes)
    logging.info('Edges: %s', edges)
    logging.info('DB: %s', args.db)
    logging.info('Method: %s', args.method)

    project = DummyProject(args.db)

    ov = overlap_reduction(nodes, edges, project)
    ov.unsupervised_removal(method=args.method, confidence_threshold=args.confidence, power_threshold=args.power_threshold, min_detections=args.min_detections)

    logging.info('Done')


if __name__ == '__main__':
    main()
