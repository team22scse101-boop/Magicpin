"""
Keep-alive pinger for Render free tier.
Run this in a terminal BEFORE submitting your URL to magicpin.
It pings your bot every 4 minutes to prevent Render from sleeping.

Usage: python keep_alive.py
"""
import time
import json
from urllib.request import urlopen, Request
from datetime import datetime

BOT_URL = "https://magicpin-ye58.onrender.com"
PING_INTERVAL = 240  # 4 minutes (Render sleeps after ~15 min)

def ping():
    try:
        r = urlopen(Request(BOT_URL + "/v1/healthz"), timeout=60)
        data = json.loads(r.read())
        status = data.get("status", "?")
        uptime = data.get("uptime_seconds", 0)
        now = datetime.now().strftime("%H:%M:%S")
        print(f"[{now}] ALIVE - status={status}, uptime={uptime}s")
        return True
    except Exception as e:
        now = datetime.now().strftime("%H:%M:%S")
        print(f"[{now}] WAKE-UP - {e}")
        # Try again after cold start
        try:
            time.sleep(5)
            r = urlopen(Request(BOT_URL + "/v1/healthz"), timeout=60)
            print(f"[{now}] RECOVERED - bot is awake now")
            return True
        except:
            print(f"[{now}] FAILED - bot may be down")
            return False

if __name__ == "__main__":
    print(f"Keep-alive pinger for: {BOT_URL}")
    print(f"Pinging every {PING_INTERVAL}s. Press Ctrl+C to stop.\n")

    # Initial wake-up
    print("Waking up bot...")
    ping()
    print()

    try:
        while True:
            time.sleep(PING_INTERVAL)
            ping()
    except KeyboardInterrupt:
        print("\nStopped.")
