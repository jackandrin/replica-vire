// Read-only research data contract. The Vite dev server proxies /api to Python.
// Never put TigerData/Snowflake/Alpaca credentials in frontend code.
export type PricePoint = { date: string; close: number };
// Fusion neural network output for one horizon: raw return in percent (mu) and its uncertainty (sigma).
export type HorizonForecast = { mu: number | null; sigma: number | null };
export type Horizons = Record<string, HorizonForecast>;
export type Stock = {
  ticker: string;
  company: string;
  sector: string;
  price: number | null;
  price_date: string | null;
  day_change_pct: number | null;
  sentiment_level: number | null;   // LLM sentiment agent, 0-100
  sentiment_trend: number | null;   // latest level minus the level ~24h earlier
  sentiment_date: string | null;
  news_count_5d: number | null;
  price_history: PricePoint[];
  direction: Direction;
  expected_pct: number | null;
  timeframe: string | null;
  height: number | null;
  suggested_usd: number | null;
  reason: string | null;
  signal_date: string | null;
  signal_ts: string | null;
  horizons: Horizons | null;
};
export type Direction = 'long' | 'short' | 'flat';
export type Overview = {
  stocks: Stock[];
  sectors: string[];
  benchmark: string;
  sentiment_source: string;
  prediction_target: string;
  horizon_order?: string[];
  review_data_connected: boolean;
};
export type Article = {
  published_at: string | null;
  source: string | null;      // alpaca / finnhub / edgar
  doc_type: string | null;    // news / 8-K / 10-Q / 10-K
  title: string | null;
  body: string | null;
  url: string | null;
  publisher: string | null;
};
export type Bar = { ts: string; close: number | null; open: number | null };
export type RegressorOutput = { key: string; label: string; yhat_pct: number | null };
export type Driver = { name: string; label: string; value: number | null; attribution: number | null };
export type SentimentDirection = 'improving' | 'deteriorating' | 'stable';
export type SentimentEvent = {
  published_at: string;
  direction: SentimentDirection | null;
  delta: number | null;
  level: number | null;
  novelty: number | null;
  materiality: number | null;
  drivers: string[];
  summary: string | null;
  doc_title: string | null;
  doc_url: string | null;
  doc_source: string | null;
  doc_type: string | null;
};
export type SentimentLevelPoint = { published_at: string; level: number | null };
export type StockDetail = {
  stock: Stock;
  history: Bar[];
  horizons: Horizons | null;
  regressors: RegressorOutput[] | null;
  drivers: Driver[] | null;
  sentiment_events: SentimentEvent[];
  sentiment_history: SentimentLevelPoint[];
  articles: Article[];
  source_note: string;
};
export type HorizonMetrics = {
  ic_mean?: number | null;
  hit_rate?: number | null;
  coverage_1sigma?: number | null;
  n_rows?: number | null;
  n_days?: number | null;
};
export type HorizonEvaluation = { horizon: string; nn: HorizonMetrics; baseline: HorizonMetrics };
export type RegressorEvaluation = { key: string; label: string; ic_mean: number | null; hit_rate: number | null };
export type RedundantPair = { a: string; b: string; a_label: string; b_label: string; corr: number | null };
export type BacktestSummary = { total_return?: number | null; sharpe?: number | null; max_drawdown?: number | null; n_periods?: number | null };
export type Performance = {
  status: string;
  reason?: string;
  horizons: HorizonEvaluation[];
  regressors: RegressorEvaluation[];
  redundant_pairs?: RedundantPair[];
  backtest?: Record<string, BacktestSummary>;
  // Cumulative returns (fractions) as stored by src.train.
  curve?: { date: string; nn: number | null; baseline: number | null; spy: number | null }[];
  meta?: { train_dates?: string[]; test_dates?: string[]; caveat?: string; trained_at?: string; n_train_rows?: number; n_test_rows?: number; folds?: number };
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

export type Session = { authenticated: boolean };
export type DirectionMode = 'both' | 'long_only' | 'short_only';
export type Settings = {
  auto_trading: boolean;
  sectors: string[];
  min_usd: number;
  max_usd: number;
  min_height: number;
  direction_mode: DirectionMode;
  max_positions: number;
  day_trade_min_height: number;
};
export type Alert = {
  id: number;
  created_at: string;
  symbol: string;
  action: 'BUY' | 'SELL' | 'SHORT' | 'COVER';
  expected_pct: number | null;
  timeframe: string | null;
  act_by: string | null;
  size_usd: number | null;
  height: number | null;
  reason: string;
  mode: 'recommend' | 'auto';
  status: 'new' | 'executed' | 'dismissed' | 'requested';
};

export type PortfolioProfile = { budget: number; risk: string; horizon: string; sectors: string[] };
export type PortfolioTarget = {
  symbol: string; company: string; sector: string; price: number | null;
  weight: number;       // percent of portfolio value
  target_usd: number; score: number; reason: string;
};
export type PortfolioPosition = {
  symbol: string; company: string; sector: string; qty: number; avg_cost: number | null; price: number | null;
  market_value: number | null; pnl: number | null; pnl_pct: number | null; weight: number | null;
};
export type PortfolioOrder = {
  symbol: string; side: 'buy' | 'sell'; notional: number | null; filled_qty: number | null;
  filled_avg_price: number | null; status: string; reason: string | null; created_at: string;
};
export type PortfolioPoint = { date: string; value: number; return_pct: number; spy_return_pct: number | null };
// A live paper-trading portfolio, generated and managed server-side (Alpaca paper account + virtual ledger).
export type LivePortfolio = {
  id: string;
  status: 'active' | 'closed';
  created_at: string;
  last_rebalance: string | null;
  profile: PortfolioProfile;
  summary: {
    value: number; cash: number; invested: number; pnl: number; pnl_pct: number;
    spy_return_pct: number | null; pending_orders: number;
  };
  targets: PortfolioTarget[];
  positions: PortfolioPosition[];
  history: PortfolioPoint[];
  notes: string[];
  orders: PortfolioOrder[];
};
export type CreatedPortfolio = LivePortfolio & { token: string };

export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) { super(message); this.status = status; }
}
async function failure(response: Response): Promise<never> {
  const body = await response.json().catch(() => ({}));
  throw new ApiError(typeof body.detail === 'string' ? body.detail : `API returned ${response.status}`, response.status);
}

export async function apiGet<T>(route: string, signal?: AbortSignal): Promise<T> {
  // Only relative /api paths, preventing accidental credential-bearing third-party requests.
  if (!route.startsWith('/api/')) throw new Error('Invalid API path');
  const response = await fetch(route, { signal, credentials: 'same-origin' });
  if (!response.ok) return failure(response);
  return response.json() as Promise<T>;
}

const STORE_KEY = 'rv.portfolio';
export type StoredPortfolio = { id: string; token: string };
export function storedPortfolio(): StoredPortfolio | null {
  try {
    const v = JSON.parse(localStorage.getItem(STORE_KEY) ?? 'null');
    return v && typeof v.id === 'string' && typeof v.token === 'string' ? v : null;
  } catch { return null; }
}
export function rememberPortfolio(p: StoredPortfolio | null): void {
  try { p ? localStorage.setItem(STORE_KEY, JSON.stringify(p)) : localStorage.removeItem(STORE_KEY); } catch { /* storage blocked */ }
}

async function portfolioRequest<T>(route: string, method: 'GET' | 'POST', token?: string, body?: unknown, signal?: AbortSignal): Promise<T> {
  if (!route.startsWith('/api/portfolios')) throw new Error('Invalid API path');
  const headers: Record<string, string> = {};
  if (token) headers.Authorization = `Bearer ${token}`;
  if (body !== undefined) headers['Content-Type'] = 'application/json';
  const response = await fetch(route, { method, headers, signal, body: body === undefined ? undefined : JSON.stringify(body) });
  if (!response.ok) return failure(response);
  return response.json() as Promise<T>;
}

/** Submit the setup profile; the server builds the portfolio and places the paper orders. */
export const createPortfolio = (profile: PortfolioProfile) =>
  portfolioRequest<CreatedPortfolio>('/api/portfolios', 'POST', undefined, profile);
export const fetchPortfolio = (id: string, token: string, signal?: AbortSignal) =>
  portfolioRequest<LivePortfolio>(`/api/portfolios/${encodeURIComponent(id)}`, 'GET', token, undefined, signal);
export const closePortfolio = (id: string, token: string) =>
  portfolioRequest<LivePortfolio>(`/api/portfolios/${encodeURIComponent(id)}/close`, 'POST', token);

export async function apiSend<T>(route: string, method: 'POST' | 'PUT' | 'DELETE', body?: unknown): Promise<T> {
  if (!route.startsWith('/api/')) throw new Error('Invalid API path');
  const response = await fetch(route, {
    method,
    credentials: 'same-origin',
    headers: body === undefined ? undefined : { 'Content-Type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!response.ok) return failure(response);
  return response.json().catch(() => ({})) as Promise<T>;
}
