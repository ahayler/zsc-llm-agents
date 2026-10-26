from langfuse import get_client

from dotenv import load_dotenv
from pathlib import Path
import os

# https://github.com/langfuse/langfuse/issues/7740
os.environ['LANGFUSE_FLUSH_AT'] = "32"

load_dotenv(Path(__file__).parent.parent.parent / ".env")

langfuse_client = get_client()

def disable_tracing():
    os.environ['LANGFUSE_TRACING_ENABLED'] = "False"