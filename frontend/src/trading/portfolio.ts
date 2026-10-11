import type { Stock } from '../api';
import type { Profile } from './parts';

export type Holding = {
  stock: Stock;
  weight: number; // percent of budget
  amount: number; // USD
  shares: number;
  color: string;
  spark: string;
  reason: string;
  score: number; // percent-rank of the ranking score within the eligible universe, 0-100
};

export type Portfolio = {
  holdings: Holding[];
  sectorFallback: boolean;
  eligible: number;
};

const COLORS = ['#23684f', '#769b7f', '#c27b68', '#b79b5f', '#668490', '#9aa49e'];
const COUNT: Record<string, number> = { Conservative: 6, Balanced: 5, Growth: 4 };
// How strongly forecast uncertainty (the network's 1-sigma, in percent) is penalised.
const DISAGREEMENT_PENALTY: Record<string, number> = { Conservative: 1, Balanced: 0.5, Growth: 0 };

export function sparkPath(closes: number[]): string {
  if (closes.length < 2) return '';
  const min = Math.min(...closes);
  const span = Math.max(...closes) - min || 1;
  return closes
    .map((v, i) => `${i ? 'L' : 'M'}${(2 + (92 * i) / (closes.length - 1)).toFixed(1)} ${(36 - (30 * (v - min)) / span).toFixed(1)}`)
    .join(' ');
}

const fmt = (v: number | null, digits = 2) => (v == null ? '—' : `${v > 0 ? '+' : ''}${v.toFixed(digits)}`);
/** 1-sigma uncertainty (percent) of the fusion network's forecast at the stock's chosen timeframe. */
export const sigmaOf = (s: Stock): number | null => (s.timeframe ? s.horizons?.[s.timeframe]?.sigma ?? null : null);
/** Conviction with direction: + for long signals, - for short, 0 when the model sees no trade. */
const signedHeight = (s: Stock) => (s.direction === 'short' ? -1 : s.direction === 'long' ? 1 : 0) * (s.height ?? 0);

export function buildPortfolio(stocks: Stock[], profile: Profile): Portfolio {
  const usable = stocks.filter(s => s.price != null && s.price > 0 && s.expected_pct != null);
  const wanted = profile.sectors.filter(s => s !== 'No preference');
  const inSector = wanted.length ? usable.filter(s => wanted.includes(s.sector)) : usable;
  const sectorFallback = wanted.length > 0 && inSector.length < 3;
  const pool = sectorFallback ? usable : inSector;
  const penalty = DISAGREEMENT_PENALTY[profile.risk] ?? 0.5;
  const scoreOf = (s: Stock) => signedHeight(s) - penalty * (sigmaOf(s) ?? 0);  // riskier profiles tolerate more uncertainty
  const ranked = [...pool].sort((a, b) => scoreOf(b) - scoreOf(a));
  const picks = ranked.slice(0, Math.min(COUNT[profile.risk] ?? 5, ranked.length));
  // Conservative: equal weight. Balanced: rank-linear. Growth: rank-squared (concentrates in top picks).
  const raw = picks.map((_, i) => {
    const r = picks.length - i;
    return profile.risk === 'Conservative' ? 1 : profile.risk === 'Growth' ? r * r : r;
  });
  const total = raw.reduce((a, b) => a + b, 0) || 1;
  const holdings = picks.map((stock, i) => {
    const weight = (100 * raw[i]) / total;
    const amount = (profile.budget * weight) / 100;
    const rank = ranked.indexOf(stock);
    return {
      stock,
      weight,
      amount,
      shares: amount / (stock.price as number),
      color: COLORS[i % COLORS.length],
      spark: sparkPath(stock.price_history.map(p => p.close)),
      score: ranked.length > 1 ? Math.round(100 * (1 - rank / (ranked.length - 1))) : 100,
      reason: stock.reason ?? `The fusion network expects ${fmt(stock.expected_pct)}% over ${stock.timeframe ?? '—'} (±${sigmaOf(stock)?.toFixed(2) ?? '—'}%); ${stock.sector} sector; news sentiment ${stock.sentiment_level?.toFixed(0) ?? '—'}/100.`,
    };
  });
  return { holdings, sectorFallback, eligible: usable.length };
}
