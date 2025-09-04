#!/usr/bin/env python
'''
Simple HTML stripper to get rid of tags (and optionally, embedded contents).

By "simple" I really mean "for instructional purposes only." It doesn't do some
very basic things, like allow HTML attributes in tags.

DATA 470D3
Stephen Davies, University of Mary Washington, fall 2025
'''
import os
import re
import argparse
import sys

parser = argparse.ArgumentParser("Strip HTML tags.")
parser.add_argument(
    "source",
    type=str,
    help="Path to source file (must end in '.html')."
)
parser.add_argument(
    "-f", "--full-content",
    action="store_true",
    help="Strip all content embedded in tags, not just the tags? (Careful!)"
)

args = parser.parse_args()

if not os.path.isfile(args.source):
    sys.exit(f"No such file {args.source}.")
if not args.source.endswith(".html"):
    sys.exit(f"Source file must end in .html! ({args.source}).")

dest_file = re.sub(r"\.html$", r".stripped", args.source)
print(f"Stripping {args.source} into {dest_file}...")

if args.full_content:
    stripper = re.compile(r"<([^>]*)>.*</\1>", re.DOTALL)
else:
    stripper = re.compile(r"<[^>]*>")

with open(args.source, "r") as s, open(dest_file, "w") as d:
    d.write(stripper.sub("", s.read()))
