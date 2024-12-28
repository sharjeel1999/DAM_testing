import argparse

from models import build_model

parser = argparse.ArgumentParser(description='Network Details')

# Architecture details
parser.add_argument('--model', default = 'classical', type = str)
parser.add_argument('--pattern_size', default=36, type = int)

args = parser.parse_args()


model = build_model(args)

