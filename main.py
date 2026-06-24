from argparse import ArgumentParser

import interpreter
from errors import HeddleError
from lexer import LexError
from parser import (
    Binding,
    Call,
    Concat,
    Index,
    Name,
    Num,
    Overlay,
    ParseError,
    Pipe,
    Pipeline,
    Program,
    Sink,
    Slice,
    Speed,
    Stack,
    Str,
    parse,
)


def format_ast(node, indent: int = 0) -> str:
    """Render an AST node as an indented multi-line tree (for 'show our work')."""
    pad = "  " * indent
    at = f"@{node.line}:{node.col}"

    def wrap(header, *children):
        return "\n".join([f"{pad}{header}  {at}", *children])

    def child(n):
        return format_ast(n, indent + 1)

    def label(text):
        return "  " * (indent + 1) + text

    match node:
        case Program(statements=stmts):
            return wrap("Program", *(child(s) for s in stmts))
        case Binding(name=name):
            return wrap(f"Binding {name!r}", child(node.pipeline))
        case Pipeline(sink=sink):
            kids = [child(node.expr)]
            if sink is not None:
                kids.append(child(sink))
            return wrap("Pipeline", *kids)
        case Sink():
            return wrap("Sink", child(node.target))
        case Pipe(stages=xs):
            return wrap("Pipe", *(child(x) for x in xs))
        case Overlay(layers=xs):
            return wrap("Overlay", *(child(x) for x in xs))
        case Concat(clips=xs):
            return wrap("Concat", *(child(x) for x in xs))
        case Stack(items=items, joins=joins):
            kids = [child(items[0])]
            for j, it in zip(joins, items[1:]):
                sym = "&" if j.axis == "h" else "/"
                kids.append(label(f"{sym}{j.mode or ''}  @{j.line}:{j.col}"))
                kids.append(child(it))
            return wrap("Stack", *kids)
        case Speed():
            return wrap(
                "Speed",
                child(node.base),
                label("factor:"),
                format_ast(node.factor, indent + 2),
            )
        case Call(args=args):
            kids = [child(node.func)]
            for a in args:
                kids.append(label(f"arg {a.name}=" if a.name else "arg (positional)"))
                kids.append(format_ast(a.value, indent + 2))
            return wrap("Call", *kids)
        case Index(axes=axes):
            return wrap("Index", child(node.base), *(child(ax) for ax in axes))
        case Slice(start=start, stop=stop, step=step):
            kids = []
            for part_name, part in (("start", start), ("stop", stop), ("step", step)):
                if part is None:
                    kids.append(label(f"{part_name}: (none)"))
                else:
                    kids.append(label(f"{part_name}:"))
                    kids.append(format_ast(part, indent + 2))
            return wrap("Slice", *kids)
        case Name(ident=ident):
            return f"{pad}Name {ident!r}  {at}"
        case Num(value=value, unit=unit):
            return f"{pad}Num {value}{unit or ''}  {at}"
        case Str(value=value):
            return f"{pad}Str {value!r}  {at}"
        case _:
            return f"{pad}{node!r}  {at}"


def run_file(filename: str, show_ast: bool = False) -> None:
    try:
        with open(filename, "r") as f:
            source = f.read()
    except OSError as e:
        print(f"heddle: cannot read {filename!r}: {e}")
        return
    try:
        program = parse(source)
    except (LexError, ParseError) as e:
        print(f"heddle: {filename}:{e}")
        return
    if show_ast:
        print(format_ast(program))
        return
    try:
        interpreter.run_program(program, cwd=".")
    except (HeddleError, NotImplementedError) as e:
        print(f"heddle: {filename}: {e}")


def run_repl(show_ast: bool = False) -> None:
    action = "parse" if show_ast else "run"
    print(f"heddle REPL — enter a line to {action} it; Ctrl-D to exit.")
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
            program = parse(line)
        except (LexError, ParseError) as e:
            print(f"  error: {e}")
            continue
        if show_ast:
            print(format_ast(program))
            continue
        try:
            interpreter.run_program(program, cwd=".")
        except (HeddleError, NotImplementedError) as e:
            print(f"  error: {e}")


def main():
    parser = ArgumentParser(
        prog="heddle",
        description="A DSL for image and video transformation.",
        epilog="Sources in `./` are available as global variables (without extensions).",
    )
    parser.add_argument(
        "filename", nargs="?", help="input file; otherwise reads from stdin"
    )
    parser.add_argument(
        "--ast",
        action="store_true",
        help="print the parsed AST instead of running the program",
    )
    args = parser.parse_args()

    if args.filename:
        run_file(args.filename, show_ast=args.ast)
    else:
        run_repl(show_ast=args.ast)


if __name__ == "__main__":
    main()
