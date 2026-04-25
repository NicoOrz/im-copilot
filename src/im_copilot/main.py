import argparse
import os
import sys

from langgraph.types import Command

from im_copilot.checkpointer import get_checkpointer
from im_copilot.graph.pipeline import build_pipeline


USAGE = 'Usage: python -m im_copilot.main "<message>"'


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="IM Copilot CLI")
    parser.add_argument("message", nargs="*", help="User message")
    parser.add_argument("--thread-id", default="cli", help="Conversation thread ID")
    parser.add_argument("--resume", default=None, help="Resume from interrupt with JSON decision")
    parser.add_argument("--web", action="store_true", help="Launch web UI server")
    parser.add_argument("--host", default="0.0.0.0", help="Web server host")
    parser.add_argument("--port", type=int, default=8000, help="Web server port")
    args = parser.parse_args(argv)

    if args.web:
        return _run_web_server(args.host, args.port)

    if not args.message:
        print(USAGE)
        return 1

    message = " ".join(args.message)

    # Use the same persistent checkpointer for both initial and resume runs
    cp_type = os.getenv("CHECKPOINTER_TYPE", "sqlite")
    with get_checkpointer(cp_type) as checkpointer:
        graph = build_pipeline(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": args.thread_id}}

        if args.resume:
            import json
            decision = json.loads(args.resume)
            result = graph.invoke(Command(resume=decision), config=config)
        else:
            initial_state = {
                "raw_message": message,
                "chat_id": "cli",
                "message_id": "cli",
                "source": "cli",
                "errors": [],
                "checks": [],
                "reflection_iteration": 0,
            }
            result = graph.invoke(initial_state, config=config)

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


def _run_web_server(host: str, port: int) -> int:
    import uvicorn
    from im_copilot.web.app import app

    print(f"Starting IM Copilot web server at http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
