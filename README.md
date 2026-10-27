# Learnly Backend

FastAPI backend that crawls a website, turns it into a searchable knowledge base, and builds learning tools on top of it: RAG chat, auto-generated MCQ quizzes, and AI learning-path graphs.

**Stack:** FastAPI · Socket.IO · DynamoDB · Qdrant (vectors) · Redis (cache/pub-sub, optional) · Groq LLM via LangChain · HuggingFace `all-MiniLM-L6-v2` embeddings

## Prerequisites

- Python **3.11** (see `.python-version`)
- Docker + Docker Compose (for Qdrant and Redis, or the full stack)
- An AWS account with DynamoDB access (or [DynamoDB Local](#using-dynamodb-local))
- A [Groq API key](https://console.groq.com) (free). Chat, quizzes and learning paths need it.

## 1. Configure environment

```bash
cp .env.example .env
```

Fill in at least:

| Variable | Purpose |
| --- | --- |
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION` | DynamoDB credentials. Leave the keys blank to use the default AWS credential chain. |
| `GROQ_API_KEY` | LLM provider key |
| `SECRET_KEY` | JWT signing key. Generate one with `python -c "import secrets; print(secrets.token_hex(32))"`. |
| `QDRANT_URL` | `http://localhost:6333` locally (Compose overrides it for the backend container) |
| `REDIS_URL` | Optional. Leave empty to run without Redis caching. |

Optional: `DYNAMODB_ENDPOINT_URL` for [DynamoDB Local](#using-dynamodb-local), `LLM_MODEL` (default `llama-3.3-70b-versatile`), `DYNAMODB_*_TABLE` to rename tables, `COLLECTIONS_GLOBAL=false` to list only a user's own collections.

`.env` is git-ignored and is not copied into the Docker image.

## 2. Create the DynamoDB tables

Every table uses a single string partition key named `PK`. Chat and quiz sessions share the `quiz_sessions` table.

```bash
for t in users collections scrape_jobs chunk_question learning_node_summaries quiz_sessions; do
  aws dynamodb create-table \
    --table-name "$t" \
    --attribute-definitions AttributeName=PK,AttributeType=S \
    --key-schema AttributeName=PK,KeyType=HASH \
    --billing-mode PAY_PER_REQUEST \
    --region "${AWS_REGION:-ap-south-1}"
done
```

If you changed any `DYNAMODB_*_TABLE` names in `.env`, use those names instead.

## 3a. Run with Docker Compose (recommended)

Starts Redis, Qdrant and the backend:

```bash
docker compose up -d --build
docker compose logs -f backend
```

The API is served at **http://localhost:3000**. The first build is slow because it downloads the PyTorch and sentence-transformers wheels.

To stop it, run `docker compose down`. Qdrant data persists in the `qdrant_data` volume.

## 3b. Run locally (for development)

Start only the infrastructure in Docker:

```bash
docker compose up -d redis qdrant
```

Then run the API with hot reload:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 3000
```

On first start the embedding model (~90 MB) is downloaded from HuggingFace.

## 4. Verify

- Swagger UI: http://localhost:3000/docs
- The startup logs should show `Connected to DynamoDB` and, if configured, `Connected to Redis`.

Quick smoke test:

```bash
# Register a user (returns a JWT)
curl -X POST localhost:3000/users/register -H 'Content-Type: application/json' \
  -d '{"email":"me@example.com","username":"me","password":"secret123"}'

# Crawl a site into a new collection (runs in the background)
curl -X POST localhost:3000/collections/ -H 'Content-Type: application/json' \
  -d '{"name":"fastapi-docs","url":"https://fastapi.tiangolo.com"}'
```

The crawler is currently capped at 5 pages per collection (`CRAWL_MAX_PAGES_LIMIT` in `app/crawler.py`).

## API overview

| Prefix | What it does |
| --- | --- |
| `/users` | Register, login (JWT), profile, the user's jobs and collections |
| `/collections` | Create (crawl + embed + generate quiz questions), list, rename, delete; crawl routes and logs; learning-path generation and node summaries |
| `/chat` | Chat sessions over a collection (RAG over Qdrant + Groq) |
| `/quizzes` | List quiz-ready collections, start a quiz, submit answers and get feedback |
| `/scrape` | Scrape a single page into a temporary in-memory chat session |

### Real-time updates (Socket.IO)

The Socket.IO server is mounted at `/socket.io`.

Client → server events:
- `join_job` with `{job_id}`
- `join_collection` with `{collection_id}`
- `join_chat` with `{collection_id, session_id}`
- `send_message` with `{collection_id, session_id?, message, top_k?}`

Server → client events:
- `job_update`, `progress`, `collection_update`
- `session_created`, `new_message`, `typing`, `typing_stopped`
- `chat_error`, `error`

## Using DynamoDB Local

```bash
docker run -d -p 8000:8000 amazon/dynamodb-local
```

1. In `.env`, set:
   - `DYNAMODB_ENDPOINT_URL=http://localhost:8000`
   - any non-empty dummy `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`
2. Create the tables with the same loop as in step 2, adding `--endpoint-url http://localhost:8000` to the command.

If the backend runs inside Docker Compose, use `http://host.docker.internal:8000` instead (on Linux this needs `extra_hosts: ["host.docker.internal:host-gateway"]` on the `backend` service).

## Deployment

`.github/workflows/backend-ci.yml` runs syntax and dependency checks on every push. On a push to `main` it SSHes into EC2 and runs `docker compose up -d --build`. It needs these repository secrets:
- `EC2_HOST`, `EC2_USER`, `EC2_SSH_KEY`, `EC2_APP_DIR`
- `EC2_PORT` (optional)

The server must have its own `.env` in `EC2_APP_DIR`.
