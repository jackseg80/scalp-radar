"""Permet de lancer via : uv run python -m scripts.fetch_history"""

from scripts.fetch_history import main
import asyncio

asyncio.run(main())
