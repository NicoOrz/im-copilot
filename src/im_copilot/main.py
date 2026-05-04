import argparse
import logging
import os
import sys
import uuid

from im_copilot.deep_agent.service import run_agent


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

    if args.debug:
        os.environ["IM_COPILOT_DEBUG"] = "1"

    if args.web and args.lark_bot:
        return _run_web_server(args.host, args.port, with_lark_bot=True)

    if args.web:
        return _run_web_server(args.host, args.port)

    if args.lark_bot:
        return _run_lark_bot(debug=args.debug)

    if args.resume:
        print("无待处理任务。")
        return 0

    if not args.message:
        print(USAGE)
        return 1

    message = " ".join(args.message)
    result = run_agent(
        message,
        thread_id=args.thread_id,
        source="cli",
        chat_id="cli",
        message_id=str(uuid.uuid4()),
    )
    if result.status == "error":
        print(result.error, file=sys.stderr)
        return 1

    print(result.summary)
    for artifact in result.artifacts.values():
        url = artifact.get("url", "")
        if url:
            print(f"{artifact.get('title', artifact.get('kind', 'artifact'))}: {url}")
    return 0


def _configure_logging(debug: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


def _run_web_server(host: str, port: int, with_lark_bot: bool = False) -> int:
    import uvicorn
    from fastapi import FastAPI
    from im_copilot.oauth_handler import router as oauth_router

    if with_lark_bot:
        os.environ["LARK_BOT_ENABLED"] = "1"
        if os.getenv("LARK_APP_ID") and os.getenv("LARK_APP_SECRET"):
            print("Lark bot WebSocket will be started alongside the web server")
        else:
            print("Warning: --lark-bot set but LARK_APP_ID or LARK_APP_SECRET missing")

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
