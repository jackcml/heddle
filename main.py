from argparse import ArgumentParser

from lexer import LexError, Token, tokenize


def _print_tokens(tokens: list[Token]) -> None:
    for tok in tokens:
        loc = f"{tok.line}:{tok.col}"
        print(f"  {loc:>6}  {tok}")


def run_file(filename: str) -> None:
    try:
        with open(filename, "r") as f:
            source = f.read()
    except OSError as e:
        print(f"heddle: cannot read {filename!r}: {e}")
        return
    try:
        tokens = tokenize(source)
    except LexError as e:
        print(f"heddle: {filename}:{e}")
        return
    _print_tokens(tokens)


def run_repl() -> None:
    print("heddle REPL — enter a line to tokenize it; Ctrl-D to exit.")
    while True:
        try:
            line = input("> ")
        except EOFError:
            print()
            break
        except KeyboardInterrupt:
            print()
            continue
        if not line.strip():
            continue
        try:
            tokens = tokenize(line)
        except LexError as e:
            print(f"  error: {e}")
            continue
        _print_tokens(tokens)


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
