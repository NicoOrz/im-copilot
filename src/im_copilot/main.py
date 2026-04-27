import argparse
import logging
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
    parser.add_argument("--lark-bot", action="store_true", help="Launch Lark bot WebSocket client")
    parser.add_argument("--host", default="0.0.0.0", help="Web server host")
    parser.add_argument("--port", type=int, default=8000, help="Web server port")
    parser.add_argument("--debug", action="store_true", help="Enable verbose debug logs")
    args = parser.parse_args(argv)
    _configure_logging(args.debug)

    if args.web:
        return _run_web_server(args.host, args.port)

    if args.lark_bot:
        return _run_lark_bot(debug=args.debug)

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


def _configure_logging(debug: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


def _run_web_server(host: str, port: int) -> int:
    import uvicorn
    from fastapi import FastAPI
    from im_copilot.oauth_handler import router as oauth_router

    try:
        from im_copilot.web.app import app
    except ImportError:
        app = FastAPI()

    app.include_router(oauth_router)
    print(f"Starting IM Copilot web server at http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")
    return 0


def _run_lark_bot(debug: bool = False) -> int:
    from im_copilot.lark_bot import LarkBot
    from im_copilot.lark_handlers import build_event_handler

    app_id = os.getenv("LARK_APP_ID")
    app_secret = os.getenv("LARK_APP_SECRET")
    if not app_id or not app_secret:
        print("Missing LARK_APP_ID or LARK_APP_SECRET", file=sys.stderr)
        return 1

    kwargs = {}
    domain = os.getenv("LARK_DOMAIN")
    if domain:
        kwargs["domain"] = domain

    bot = LarkBot(
        app_id=app_id,
        app_secret=app_secret,
        encrypt_key=os.getenv("LARK_ENCRYPT_KEY"),
        verification_token=os.getenv("LARK_VERIFICATION_TOKEN"),
        debug=debug,
        **kwargs,
    )
    bot.start_ws(build_event_handler(bot))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
