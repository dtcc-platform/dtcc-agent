import os

# The chatbot's startup Bedrock probe (chatbot/runtime/pai.py) makes a real
# request; tests that start the app must never reach the network.
os.environ.setdefault("DTCC_AGENT_BEDROCK_PROBE", "0")
