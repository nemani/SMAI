# -*- coding: utf-8 -*-

import argparse
import sys
import logging

from smai import __version__
# from .language_model import LanguageModel
from .lstm_model import LstmLanguageModel
from .ngram_model import NgramLanguageModel

__author__ = "Arjun Nemani"
__copyright__ = "Arjun Nemani"
__license__ = "mit"

_logger = logging.getLogger(__name__)


def parse_args(args):
    """Parse command line parameters

    Args:
      args ([str]): command line parameters as list of strings

    Returns:
      :obj:`argparse.Namespace`: command line parameters namespace
    """
    parser = argparse.ArgumentParser(
        description="SMAI: Social Media API for IRE")

    parser.add_argument(
        "--version",
        action="version",
        version="SMAI {ver}".format(ver=__version__))
    parser.add_argument(
        dest="mode",
        help="Eval or Train",
        # action="store_const",
    )
    parser.add_argument(
        dest="type",
        help="LSTM or Ngram",
        # action="store_const",
    )
    parser.add_argument(
        dest="slug",
        help="Name of Social Media Channel",
        # action="store_const",
    )
    parser.add_argument(
        '-d',
        "--base_dir",
        required=('train' in sys.argv),
        help="path to dir relative to current or absolute where we will save the models and the data given",
        # action="store_const",
    )
    parser.add_argument(
        '-f',
        "--file",
        required=('train' in sys.argv),
        help="File to use for training",
        # action="store_const",
    )
    parser.add_argument(
        '-s',
        "--seed_text",
        required=False,
        help="Seed Text to use for conditional generation",
        # action="store_const",
    )
    parser.add_argument(
        "--epochs",
        required=('train' in sys.argv and 'LSTM' in sys.argv),
        type=int,
        help="Number of Epochs to train for",
        # action="store_const",
    )

    return parser.parse_args(args)


def setup_logging(loglevel):
    """Setup basic logging

    Args:
      loglevel (int): minimum loglevel for emitting messages
    """
    logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(level=loglevel, stream=sys.stdout,
                        format=logformat, datefmt="%Y-%m-%d %H:%M:%S")


def main(args):
    """Main entry point allowing external calls

    Args:
      args ([str]): command line parameter list
    """
    args = parse_args(args)

    if args.type == "LSTM":
        LM = LstmLanguageModel(
            mode=args.mode, slug=args.slug, base_dir=args.base_dir)
    else:
        LM = NgramLanguageModel(
            mode=args.mode, slug=args.slug, base_dir=args.base_dir)

    if args.mode == 'train':
        with open(args.file, 'r') as file:
            data = file.readlines()

            if args.type == "LSTM":
              LM.train(data, epochs=args.epochs)
            else:
              LM.train(data)
            LM.save_model()

    if args.mode == 'eval':
        LM.load_model()
        LM.start_prompt()


def run():
    """Entry point for console_scripts"""
    main(sys.argv[1:])


if __name__ == "__main__":
    run()
