import argparse
import sys

from langgraph.types import Command
from langgraph.checkpoint.memory import InMemorySaver

from im_copilot.graph.pipeline import build_pipeline, run_pipeline


USAGE = 'Usage: python -m im_copilot.main "<message>"'


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="IM Copilot CLI")
    parser.add_argument("message", nargs="*", help="User message")
    parser.add_argument("--thread-id", default="cli", help="Conversation thread ID")
    parser.add_argument("--resume", default=None, help="Resume from interrupt with JSON decision")
    args = parser.parse_args(argv)

    if not args.message:
        print(USAGE)
        return 1

    message = " ".join(args.message)

    # If resuming from interrupt
    if args.resume:
        import json
        decision = json.loads(args.resume)
        checkpointer = InMemorySaver()
        graph = build_pipeline(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": args.thread_id}}
        result = graph.invoke(Command(resume=decision), config=config)
    else:
        result = run_pipeline(message, thread_id=args.thread_id)

    # Handle interrupt output
    if "__interrupt__" in result:
        interrupt_data = result["__interrupt__"][0]
        print("=" * 60)
        print(f"[INTERRUPT] {interrupt_data.value['gate']}")
        print("=" * 60)
        print(interrupt_data.value.get("message", ""))
        if "plan" in interrupt_data.value:
            print("\n计划:")
            for step in interrupt_data.value["plan"]:
                print(f"  - {step}")
        if "questions" in interrupt_data.value:
            print("\n问题:")
            for i, q in enumerate(interrupt_data.value["questions"], 1):
                print(f"  {i}. {q}")
        print("\n恢复命令:")
        print(f'  python -m im_copilot.main --thread-id {args.thread_id} --resume \'{{"approved": true}}\' "{message}"')
        return 0

    print(result.get("summary", ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
