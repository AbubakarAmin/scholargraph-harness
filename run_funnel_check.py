"""Run discover_topics() once and report the rejection_funnel breakdown."""
import json
import logging
import sys

logging.basicConfig(level=logging.INFO, stream=sys.stdout)

from core.config import config
from core.utils import log_agent_action

funnel_data = {}
_orig_log = log_agent_action

def capture_log(agent, action, data=None):
    if action == "rejection_funnel":
        funnel_data.update(data or {})
    _orig_log(agent, action, data)

import core.utils
core.utils.log_agent_action = capture_log

from agents.topic_hunter import TopicHunterAgent

hunter = TopicHunterAgent()
try:
    topics = hunter.discover_topics(domain="computer_science", n_parallel=3)
    print("\n=== REJECTION FUNNEL ===")
    print(json.dumps(funnel_data, indent=2))
    print(f"\nTopics found: {len(topics)}")
except Exception as e:
    print(f"\nError: {e}")
    print("Partial funnel data:", json.dumps(funnel_data, indent=2))
