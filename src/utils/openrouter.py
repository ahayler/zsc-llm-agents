import requests
import json
import os
from dotenv import load_dotenv
from utils.log_utils import get_logger

logger = get_logger(__name__)
load_dotenv()

def get_openrouter_key_metadata():

    if os.getenv('OPENROUTER_API_KEY') is None:
        logger.warning("OPENROUTER_API_KEY is not set")
        return None

    response = requests.get(
    url="https://openrouter.ai/api/v1/key",
    headers={
        "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}"
    }
    )

    try: 
        return dict(response.json())['data']
    except Exception as e:
        logger.warning(f"Error getting openrouter key metadata: {e}")
        return None

def get_remaining_openrouter_balance():
    key_metadata = get_openrouter_key_metadata()

    if isinstance(key_metadata, dict):
        return key_metadata.get('limit_remaining', None)
    else:
        return 0


if __name__ == "__main__":
    print(get_remaining_openrouter_balance())