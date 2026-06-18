from argparse import ArgumentParser

def run_file(filename: str) -> None:
    print('TODO: File input not yet implemented.')


def run_repl() -> None:
    print('TODO: REPL not yet implemented.')


def main():
    parser = ArgumentParser(
            prog='heddle',
            description='A DSL for image and video transformation.',
            epilog='Sources in `./` are available as global variables (without extensions).')
    parser.add_argument('filename', nargs='?', help='input file; otherwise reads from stdin')
    args = parser.parse_args()

    if args.filename:
        run_file(args.filename)
    else:
        run_repl()


if __name__ == '__main__':
    main()
