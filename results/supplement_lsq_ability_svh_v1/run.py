#!/usr/bin/env python3
"""Rebuild LSQ teachers, retrain Ability/SVH K8, then evaluate three methods.

Run stages: check -> teacher -> train -> warm -> search -> recheck -> summarize.
All new physics/labels use LSQ controller-target projection; legacy paths are
read-only. Long search and CPU recheck jobs resume from verified output files.
"""
from __future__ import annotations

import argparse

import evaluate as e
import pipeline as p


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("check", "teacher", "train", "warm", "search", "recheck",
                                           "summarize", "worker"))
    parser.add_argument("--hand", choices=(*p.HANDS, "all"), default="all")
    parser.add_argument("--gen", choices=(*p.GENERATORS, "all"), default="all")
    parser.add_argument("--method", choices=(*p.METHODS, "all"), default="all")
    parser.add_argument("--object", help="A single training object for teacher or test object for search/recheck")
    parser.add_argument("--arm", choices=p.ARMS, help="One PD branch for a search pilot")
    parser.add_argument("--part", type=int, choices=range(8), help="One eight-source search part")
    parser.add_argument("--smoke", action="store_true", help="Independent two-source teacher or 2-epoch training smoke")
    parser.add_argument("--max-worker-groups", type=int, help="Limit physical recheck to this many short workers")
    parser.add_argument("--batch-start", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--batch-count", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--cache-sha", help=argparse.SUPPRESS)
    args = parser.parse_args()
    hands = p.HANDS if args.hand == "all" else (args.hand,)
    gens = p.GENERATORS if args.gen == "all" else (args.gen,)
    methods = p.METHODS if args.method == "all" else (args.method,)
    if args.stage == "worker":
        if (args.hand == "all" or args.gen == "all" or args.method == "all"
                or args.object not in p.TEST10 or args.batch_start is None
                or args.batch_count is None or not args.cache_sha):
            parser.error("worker requires specific hand/gen/method/test object and frozen batch arguments")
        e.worker(args.hand, args.gen, args.method, args.object,
                 args.batch_start, args.batch_count, args.cache_sha)
        return
    if args.batch_start is not None or args.batch_count is not None or args.cache_sha is not None:
        parser.error("Batch/cache arguments only apply to internal worker")
    if args.smoke and args.stage not in ("teacher", "train"):
        parser.error("--smoke is for teacher or train only")
    if args.object is not None:
        wanted = p.TRAIN40 if args.stage == "teacher" else p.TEST10
        if args.stage not in ("teacher", "search", "recheck") or args.object not in wanted:
            parser.error("--object must belong to the selected training/test stage")
    if args.arm is not None and (args.stage != "search" or args.method != "pd9"):
        parser.error("--arm requires search --method pd9")
    if args.part is not None and (args.stage != "search" or args.object is None or
                                  args.hand == "all" or args.gen == "all" or args.method == "all"):
        parser.error("--part requires search with one hand, generator, method and test object")
    if args.max_worker_groups is not None and (args.stage != "recheck" or args.max_worker_groups < 1):
        parser.error("--max-worker-groups is for recheck and must be positive")
    if args.stage == "check":
        p.check_inputs()
    elif args.stage == "teacher":
        for hand in hands:
            for gen in gens:
                p.teachers(hand, gen, (args.object,) if args.object else p.TRAIN40, smoke=args.smoke)
    elif args.stage == "train":
        for hand in hands:
            p.train(hand, smoke=args.smoke)
    elif args.stage == "warm":
        for hand in hands:
            for gen in gens:
                p.warm(hand, gen)
    elif args.stage == "search":
        for hand in hands:
            for gen in gens:
                for method in methods:
                    p.search(hand, gen, method, (args.object,) if args.object else p.TEST10,
                             chosen_arm=args.arm, chosen_part=args.part)
    elif args.stage == "recheck":
        for hand in hands:
            for gen in gens:
                for method in methods:
                    e.recheck(hand, gen, method, (args.object,) if args.object else p.TEST10,
                              worker_limit=args.max_worker_groups)
    else:
        for hand in hands:
            for gen in gens:
                e.summarize(hand, gen, methods=None if args.method == "all" else (args.method,))


if __name__ == "__main__":
    main()
