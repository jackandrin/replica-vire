# ReplicaVire

## Vercel deployment

The repository's `vercel.json` routes `/api/*` requests to the Python API service and all other requests to the frontend. The API reads market data from TigerData; its connection string must be configured as `TIGER_DATABASE_URL` in the **API service's** Vercel Environment Variables for each deployment environment you use. Do not set it as a `VITE_` variable or expose it to the frontend. Redeploy the API after adding or changing the variable.

The database must also have the tables initialized and populated by the pipeline. See [`hackathon-trader/README.md`](hackathon-trader/README.md) for database setup and pipeline instructions.

For local development, start the API from `hackathon-trader/` with `python -m uvicorn src.web_api:app --host 127.0.0.1 --port 8000`; the Vite development server proxies `/api` to it.
