#!/usr/bin/env python3
"""
Enhanced Ludo Game Launcher

A beautiful and interactive Ludo game with:
- Multiple AI strategies to choose from
- Human player support
- Enhanced graphics with token stacking
- Real-time game statistics
- Tournament simulation mode

Usage:
    python main.py [--port PORT] [--host HOST] [--share]

Options:
    --port PORT    Port to run the server on (default: 7860)
    --host HOST    Host to bind to (default: 0.0.0.0)
    --share        Create a public shareable link
    --help         Show this help message
"""

import argparse
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(
        description="Launch the Enhanced Ludo Game Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py                    # Launch on default port 7860
    python main.py --port 8080        # Launch on port 8080
    python main.py --share            # Create shareable link
    python main.py --host localhost   # Only local access

Game Features:
    🎮 Human Player Mode: Select "human" strategy to play yourself
    🤖 AI Strategies: Choose from 12+ different AI personalities
    🎨 Enhanced Graphics: Beautiful board with smooth animations
    📊 Statistics: Track wins and game performance
    🏆 Tournament Mode: Simulate multiple games for strategy analysis
        """,
    )

    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run the server on (default: 7860)",
    )

    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)"
    )

    parser.add_argument(
        "--share", action="store_true", help="Create a public shareable link via Gradio"
    )

    args = parser.parse_args()

    try:
        # Import and launch the app
        from ludo_interface.app import launch_app

        print("🎲 Enhanced Ludo Game")
        print("=" * 50)
        print("🚀 Starting the game interface...")
        print(f"📡 Server: http://{args.host}:{args.port}")
        if args.share:
            print("🌐 Public link will be generated...")
        print("=" * 50)
        print("\n🎮 Game Features:")
        print("• 👤 Human players - select 'human' strategy")
        print("• 🤖 12+ AI strategies to choose from")
        print("• 🎨 Enhanced graphics with token stacking")
        print("• 📊 Real-time statistics and move history")
        print("• 🏆 Tournament simulation mode")
        print("\n💡 Tips:")
        print("• Set Red player to 'human' to play yourself")
        print("• Use 'Random Strategies' for quick AI setup")
        print("• Enable 'Show Token IDs' for easier tracking")
        print("• Try different AI combinations in tournament mode")
        print("\n✨ Enjoy playing Enhanced Ludo!")
        print("=" * 50)

        # Launch the app
        demo = launch_app()
        demo.launch(
            server_name=args.host,
            server_port=args.port,
            share=args.share,
            inbrowser=False,
            show_error=True,
            quiet=False,
        )

    except ImportError as e:
        print(f"❌ Error: Missing dependencies - {e}")
        print("\n🔧 To install dependencies, run:")
        print("pip install gradio pillow")
        print("\nOr if using uv:")
        print("uv add gradio pillow")
        return 1

    except Exception as e:
        print(f"❌ Error launching game: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())