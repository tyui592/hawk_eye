"""Catalog outputs; default to a dry run."""

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="apply the migration")
    args = parser.parse_args()
    from hawk_eye.experiments.organize import migrate
    migrate(args.apply)


if __name__ == "__main__":
    main()
