import sys

from im_copilot.graph.pipeline import run_pipeline


USAGE = 'Usage: python -m im_copilot.main "<message>"'


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if not args:
        print(USAGE)
        return 1

    message = " ".join(args)
    result = run_pipeline(message)
    print(result.get("summary", ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
