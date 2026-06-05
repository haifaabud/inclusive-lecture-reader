# API keys, settings

import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("This is where we are going to put our API key")
