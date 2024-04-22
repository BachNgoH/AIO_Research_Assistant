import argparse
import logging
from time import time
from data_generation.generate_relationship import generate_relationships


def main(args):
    logging.info("Generating relationships")
    logging.info(f"Using service: {args.service}")
    start = time()
    generate_relationships(args)
    end = time()
    logging.info(f"Time taken: {end - start} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='BachNgoHParsedArxivPapers_12k')
    parser.add_argument('--load_local', type=bool, default=False)
    parser.add_argument('--service', type=str, default='hf')
    parser.add_argument('--model_name', type=str, default='Gemma_7b_Citation_v2')
    parser.add_argument('--output_path', type=str, default='../outputs')
    args = parser.parse_args()
    main(args)