"""Permet de lancer le backend via : uv run python -m backend"""

from backend.main import main
import asyncio

asyncio.run(main())
