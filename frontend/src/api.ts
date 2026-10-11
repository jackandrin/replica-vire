// Read-only research data contract. The Vite dev server proxies /api to Python.
// Never put TigerData/Snowflake/Alpaca credentials in frontend code.
export type PricePoint = { date: string; close: number };
export type Stock = {
  ticker: string;
  company: string;
  sector: string;
  price: number | null;
  price_date: string | null;
  day_change_pct: number | null;
  sentiment: number | null;
  sentiment_1d: number | null;
  news_count_5d: number | null;
  feature_date: string | null;
  prediction_date: string | null;
  ridge_excess_pct: number | null;
  mlp_excess_pct: number | null;
  mean_excess_pct: number | null;
  spread_pp: number | null;
  price_history: PricePoint[];
};
export type Overview = {
  stocks: Stock[];
  benchmark: string;
  sentiment_source: string;
  prediction_target: string;
  review_data_connected: boolean;
};
export type Article = {
  event_time: string;
  ingested_at: string;
  source_api: string;
  headline: string | null;
  summary: string | null;
  publisher: string | null;
  url: string;
  sentiment: number | null;
};
export type Bar = { ts: string; close: number | null; open: number | null };
export type SentimentPoint = { d: string; sent_1d: number | null; sent_5d: number | null; news_cnt_5d: number | null };
export type StockDetail = {
  stock: Stock;
  history: Bar[];
  articles: Article[];
  sentiment_history: SentimentPoint[];
  source_note: string;
};
export type Evaluation = {
  name: string;
  hit_rate: number | null;
  ic_mean: number | null;
  n_rows: number | null;
  n_days: number | null;
};
export type Performance = {
  status: string;
  models: Evaluation[];
  reason?: string;
  meta?: { train_dates?: string[]; test_dates?: string[]; caveat?: string; trained_at?: string };
  curve?: { date: string; ridge: number; mlp: number; agreement: number; spy: number }[];
  backtest?: Record<string, { total_return: number | null; sharpe: number | null; max_drawdown: number | null; n_periods: number | null }>;
};
export type Replay = {
  ticker: string;
  selected_date: string;
  status: 'retrospective_only' | 'insufficient_data';
  latest_research_date: string | null;
  verified_forecast: null;
  realized_return: null;
  reason: string;
};

export async function apiGet<T>(route: string, signal?: AbortSignal): Promise<T> {
  // Only relative /api paths, preventing accidental credential-bearing third-party requests.
  if (!route.startsWith('/api/')) throw new Error('Invalid API path');
  const response = await fetch(route, { signal });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(typeof body.detail === 'string' ? body.detail : `API returned ${response.status}`);
  }
  return response.json() as Promise<T>;
}
