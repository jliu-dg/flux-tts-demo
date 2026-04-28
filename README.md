# Vapi Support Voice Demo

Realtime voice support demo built on Deepgram Voice Agent + Aura sandbox TTS.

## Features

- Vapi-style customer support flow
- Sandbox voice only (`aura-3-canopy-en`)
- Optional progressive TTS (`/speak` + `/add_text`)
- Expressive mode toggle in UI
- Debug panel with latency and flow events

## Local Run

1. Create `.env` from `.env.example`
2. Set `DEEPGRAM_API_KEY`
3. Install deps and run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python client.py
```

Open `http://127.0.0.1:5051`.

## Deploy (Fly.io)

1. Login:

```bash
flyctl auth login
```

2. Pick a unique app name and create app:

```bash
flyctl apps create <your-unique-app-name>
```

3. Update `fly.toml`:

- Set `app = '<your-unique-app-name>'`

4. Set required secret:

```bash
flyctl secrets set DEEPGRAM_API_KEY=<your_key>
```

5. Deploy:

```bash
flyctl deploy
```

6. Open:

```bash
flyctl open
```

Your public URL will be:

`https://<your-unique-app-name>.fly.dev`
