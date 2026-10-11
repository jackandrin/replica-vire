# Training the v2 models on USC Theia

Three stages: **download** (needs internet + keys), **sentiment backfill** (GPU job), **train** (CPU or GPU job).
Training reads only parquet files, so compute nodes don't need internet or database access.

```
 laptop / server / Theia login node            Theia compute nodes                      server running the bot
 ─────────────────────────────────            ───────────────────                      ──────────────────────
 scripts/backfill.sh  ─► data/bundle/  ─────► sentiment.sbatch ─► sentiment_events.parquet
                                              theia.sbatch     ─► models/v2/, results/ ──► predict / autotrade
```

## 1. What data the models need

| Bundle file | Source | Used for | Rough size (55 stocks + 13 ETFs, 2019→now) |
|---|---|---|---|
| `bars_15m.parquet` | Alpaca (SIP history, IEX fallback) | prices, intraday features, all labels | 3–9 M rows (more with extended hours), 0.2–0.6 GB |
| `bars_1d.parquet` | Alpaca | daily features (MAs, 52-week, volatility, beta) | ~130 k rows |
| `events.parquet` | SEC EDGAR (8-K Item 2.02, Form 4), Finnhub (analyst trends), FINRA (daily short volume) | earnings, insider, analyst, short-selling regressors | ~200–400 k rows |
| `documents.parquet` | Alpaca news (full text), Finnhub news (~1 year), EDGAR 8-K/10-Q/10-K text | sentiment agent input | 100–400 k docs, 0.5–2 GB |
| `sentiment_events.parquet` | sentiment agent (this guide, step 3) | 4 sentiment features | one row per scored doc |
| `symbols.parquet`, `manifest.json` | sectors.py | sector ETF mapping, row counts | tiny |

Why this much: features need ~1 year of warm-up (52-week high, 63-day momentum), and the walk-forward training
needs several years after that so that the held-out test period (last 15%) is ~1 year. Starting in 2019 gives
~6 usable years ≈ 55 stocks × 1,500 days × 7 decision times ≈ **575 k training rows**.

Known gaps (the pipeline handles them; the affected features are just weaker):
- Finnhub's free tier returns only recent analyst recommendation trends, so the **analyst** regressor has
  little history. Paid Finnhub or another ratings source would fix this.
- Finnhub company news only goes back ~1 year; Alpaca news (Benzinga) covers earlier years.
- Product reviews have no free API, so they aren't ingested. A source can be added later as `documents`
  rows with `doc_type='review'`.
- If Alpaca refuses SIP, bars fall back to IEX (a few % of market volume). Volume features are ratios, so they
  still work, but they're noisier.

## 2. Download (any machine with internet: your laptop, the bot server, or a Theia login node)

```bash
cd hackathon-trader
./setup.sh --skip-apt                       # or setup.ps1 on Windows
nano .env                                   # ALPACA_*, FINNHUB_API_KEY, SEC_USER_AGENT, TIGER_DATABASE_URL
bash scripts/backfill.sh --bg               # resumable; tail -f logs/backfill.log
```
It creates the tables, checks every API, downloads bars → events → documents into Tiger Data, and exports
`data/bundle/`. Documents are the slow part (hours). For a faster first pass: `SINCE=2022-01-01 bash scripts/backfill.sh`.

Check size and cost of the sentiment backfill before running it:
```bash
python -m src.sentiment_agent --source bundle --sink bundle --estimate
```

## 3. Theia setup (once)

```bash
ssh <you>@<theia-login>
git clone <your GitHub repo URL> ~/replica-vire && ln -s ~/replica-vire/hackathon-trader ~/hackathon-trader
cd ~/hackathon-trader
module load python            # whatever RC's Python 3.10+ module is called
python -m venv .venv && . .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu121     # CUDA wheel (bundles its own runtime)
pip install -r requirements-pipeline.txt -r requirements-sentiment-hf.txt
pytest -q                                                                  # offline tests, ~30 s
```
Code comes from git (a Linux clone keeps the scripts' LF line endings; `sbatch` rejects Windows CRLF). The data is
not in git, so copy the bundle and your `.env` from the machine that ran the backfill:
```bash
rsync -av hackathon-trader/data/bundle/ <you>@<theia-login>:~/hackathon-trader/data/bundle/
scp hackathon-trader/.env <you>@<theia-login>:~/hackathon-trader/.env     # only needed for step 6 (publishing)
```
Download the sentiment models on the **login node**; compute nodes may have no internet:
```bash
export HF_HOME=$HOME/.cache/huggingface
huggingface-cli login                       # Llama 3.1 is gated: accept Meta's license on Hugging Face first
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct --exclude "original/*"
huggingface-cli download Snowflake/snowflake-arctic-embed-m-v1.5
```
Fill in the `PLACEHOLDER` partition/account lines in `deploy/*.sbatch`, or pass them on the command line
(`sbatch -p <partition> -A <account> ...`).

## 4. Run

```bash
sbatch --export=ALL,MODE=quick deploy/theia.sbatch      # smoke test: whole pipeline, few epochs (~5 min)
SENT=$(sbatch --parsable deploy/sentiment.sbatch)       # GPU sentiment backfill (resumable: resubmit if it times out)
sbatch --dependency=afterok:$SENT deploy/theia.sbatch   # full training once sentiment finishes
squeue -u $USER ; tail -f logs/slurm-train-*.out
```
Why it's the same model for backfill and live: the backfill runs `meta-llama/Llama-3.1-8B-Instruct` and
`snowflake-arctic-embed-m-v1.5` locally, and the live agent calls the same models through Cortex
(`llama3.1-8b`, `snowflake-arctic-embed-m-v1.5`). The sentiment features the network trains on therefore match
what it sees in production.

You can skip the sentiment job; training still runs and the sentiment features stay neutral. You can also run the
sentiment agent with Cortex instead of the GPU: `python -m src.sentiment_agent --backend cortex --source bundle --sink bundle`
(costs Snowflake credits; check `--estimate` first).

## 5. Read the results

`results/metrics.json` (also shown on the dashboard's Performance page):

| Field | Meaning | Healthy range |
|---|---|---|
| `horizons.<h>.nn.ic_mean` | mean cross-sectional rank correlation of predicted vs realised return | 0.00–0.05 is realistic; **> 0.10 means suspect a bug** |
| `horizons.<h>.nn.hit_rate` | sign accuracy | 0.50–0.54 |
| `horizons.<h>.nn.coverage_1sigma` | share of outcomes within ±1σ | ≈ 0.68 if σ is calibrated |
| `horizons.<h>.baseline` | one ridge on all raw features | the network should match or beat it |
| `regressors.<family>.ic_mean` | each regressor alone, next-day | small; many near 0 |
| `redundant_pairs` | regressors whose OOF predictions correlate > 0.8 | candidates to merge |
| `backtest` | daily long-short top/bottom 5 at 10:00, 1-day hold, costs included | compare `nn` vs `baseline` vs `spy` |

If the network does not beat the baseline on the held-out period, that is a real finding, not a failure.

## 6. Publish to Tiger Data (this is what puts it on the website)

Run this on the Theia **login node** once training has finished. It needs `.env` with `TIGER_DATABASE_URL`, and
login nodes normally have internet:
```bash
cd ~/hackathon-trader && . .venv/bin/activate
python -m src.init_db --tiger          # creates the model_runs / model_artifacts tables (safe to repeat)
python -m src.publish_models           # models + metrics + sentiment history -> Tiger Data
python -m src.publish_models --list    # check it arrived
```
| What it uploads | Table | Who reads it |
|---|---|---|
| `results/metrics.json` + run metadata | `model_runs` | the website's **Performance** page (Vercel API) |
| `models/v2/*.joblib`, `meta.json` | `model_artifacts` | `src.predict` on the bot machine downloads the newest version automatically |
| `data/bundle/sentiment_events.parquet` | `sentiment_events` | sentiment charts on the website; the live agent continues from it |

Nothing has to be committed to GitHub or copied by hand. The next `run_cycle` / `predict` on your bot machine picks up
the new model, writes `signals`, and the website shows the new forecasts, regressors and explanations.

If the login node can't reach Tiger Data (some clusters block outbound database ports), copy `models/v2/`, `results/`
and `data/bundle/sentiment_events.parquet` to your laptop and run `python -m src.publish_models` there instead.

Retrain monthly or when results drift: re-run `scripts/backfill.sh` (incremental) and copy the new bundle, then
`sbatch deploy/theia.sbatch` and `python -m src.publish_models`. The feature cache rebuilds automatically when the
bundle changes.
