# Mosa Chat UI

Browser UI for the local Mosa RAG server.

This frontend talks to `serve_mosa_rag.py` and renders:

- grounded answers
- retrieval confidence
- abstentions for low-confidence queries
- the source records returned by the backend

## Run it

From the repo root, start the API:

```bash
python serve_mosa_rag.py --host 127.0.0.1 --port 8000
```

Then start the frontend:

```bash
cd chat-ui
npm install
npm run dev
```

By default the UI calls `http://127.0.0.1:8000`.

To point it somewhere else:

```bash
VITE_RAG_API_BASE_URL=http://127.0.0.1:8128 npm run dev
```

## Backend contract

The UI uses:

- `GET /health`
- `POST /answer`

`/answer` should return:

- `answer`
- `abstained`
- `retrieval_confidence`
- `results`

If `abstained` is `true`, the UI shows the abstention and lets the user retry with `allow_low_confidence: true`.
