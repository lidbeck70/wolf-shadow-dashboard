#!/usr/bin/env python3
"""discord_probe.py — diagnostik: vart pekar Actions-webhooken, och kommer
ett testmeddelande fram?

Skriver ut webhookens namn, kanal-id och server-id (aldrig URL:en eller
token) och postar ett testmeddelande via samma kod som larmen
(alerts.channels.discord.send). Ändrar ingenting."""
import json
import os
import sys
import urllib.request
from datetime import datetime, timezone


def main() -> int:
    url = (os.environ.get("DISCORD_WEBHOOK_URL") or "").strip()
    if not url:
        print("DISCORD_WEBHOOK_URL saknas i Actions-secrets.")
        return 1
    print(f"URL-form: {'OK' if url.startswith('https://discord.com/api/webhooks/') else 'FEL'} "
          f"· längd {len(url)} · id-del {url.split('/')[5] if url.count('/') >= 6 else '?'}")

    # Webhookens metadata (GET utan auth — svarar med namn, kanal och server)
    req = urllib.request.Request(url, headers={"User-Agent": "wolf-shadow-dashboard-alerts/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            info = json.loads(r.read().decode("utf-8"))
        print(f"Webhook: namn='{info.get('name')}' · kanal-id={info.get('channel_id')} · "
              f"server-id={info.get('guild_id')} · typ={info.get('type')}")
    except Exception as exc:
        print(f"Kunde inte läsa webhookens metadata: {exc}")

    from alerts.channels import discord
    stamp = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    ok = discord.send(f"🔧 Testmeddelande från GitHub Actions ({stamp}). Ser du det här "
                      f"är larmkanalen rätt.", {"username": "wolf-shadow-probe"})
    print(f"Utskick: {'OK' if ok else 'MISSLYCKADES — ' + discord.last_error}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
