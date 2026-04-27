"use client";

import { useSearchParams, useRouter } from "next/navigation";
import { useState, useEffect, useMemo, useRef, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useQueries, useQueryClient } from "@tanstack/react-query";
import {
  Search, X, Plus, BarChart2, TrendingUp, TrendingDown,
  Info, Activity, Newspaper, Zap, Trophy, RefreshCcw,
  Bell, Share2, Bookmark, ChevronDown, ChevronUp,
  ArrowUpRight, ArrowDownRight, Shield, Target,
  Minus, Eye, Filter, Sparkles, CheckCircle2, Clock,
  AlertTriangle, TrendingUp as Trend, BarChart,
} from "lucide-react";

// ============================================================
// CONSTANTS & CONFIG
// ============================================================

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

const CHART_COLORS = ["#7C3AED", "#2563EB", "#059669", "#D97706"];
const CHART_RANGES = ["1W", "1M", "3M", "1Y"] as const;
type ChartRange = typeof CHART_RANGES[number];

// ============================================================
// TYPES
// ============================================================

interface PricePoint {
  date: string;
  close: number;
}

interface Fundamentals {
  roe: number;
  debt_equity: number;
  revenue_growth: number;
  profit_margin: number;
  pe_ratio?: number;
  eps?: number;
  market_cap?: number;
}

interface StockData {
  symbol: string;
  prices: PricePoint[];
  final_score: number;
  sentiment_score: number;
  risk_score: number;
  technical_score: number;
  fundamental_score: number;
  fundamentals: Fundamentals;
  explanation: string;
}

interface NewsItem {
  title: string;
  link: string;
  source?: string;
  sentiment?: "bullish" | "bearish" | "neutral";
  symbol: string;
  publishedAt?: string;
}

interface SearchSuggestion {
  symbol: string;
  name: string;
}

interface AlertConfig {
  id: string;
  symbol: string;
  type: "price_above" | "price_below" | "score_change" | "risk_spike";
  value: number;
  createdAt: string;
  active: boolean;
  triggered?: boolean;
  triggeredAt?: string;
}

interface SavedComparison {
  id: string;
  name: string;
  symbols: string[];
  savedAt: string;
}

// ============================================================
// FORMATTERS
// ============================================================

const fmt = {
  currency: (v?: number) =>
    v != null
      ? new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", notation: "compact" }).format(v)
      : "—",
  percent: (v?: number) =>
    v != null ? `${v > 0 ? "+" : ""}${v.toFixed(2)}%` : "—",
  number: (v?: number) => (v != null ? v.toFixed(2) : "—"),
  score: (v?: number) => (v != null ? `${(v * 100).toFixed(0)}` : "—"),
  scoreLabel: (v: number) =>
    v >= 0.75 ? "Strong" : v >= 0.5 ? "Moderate" : v >= 0.25 ? "Weak" : "Poor",
};

// ============================================================
// HELPERS
// ============================================================

function getAccessorValue(data: StockData, key: string): number | undefined {
  if (key.includes(".")) {
    const [, field] = key.split(".");
    return (data.fundamentals as Record<string, number | undefined>)[field];
  }
  return (data as Record<string, unknown>)[key] as number | undefined;
}

function getSentimentColor(sentiment?: string) {
  if (sentiment === "bullish") return "text-emerald-400 bg-emerald-500/10 border-emerald-500/20";
  if (sentiment === "bearish") return "text-rose-400 bg-rose-500/10 border-rose-500/20";
  return "text-gray-400 bg-white/5 border-white/10";
}

function getSentimentIcon(sentiment?: string) {
  if (sentiment === "bullish") return <TrendingUp size={10} />;
  if (sentiment === "bearish") return <TrendingDown size={10} />;
  return <Minus size={10} />;
}

function timeAgo(dateStr?: string): string {
  if (!dateStr) return "";
  const diff = Date.now() - new Date(dateStr).getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  return `${Math.floor(hrs / 24)}d ago`;
}

// ============================================================
// COMPANY AVATAR
// ============================================================

function CompanyAvatar({ symbol, className = "w-10 h-10" }: { symbol: string; className?: string }) {
  const [error, setError] = useState(false);
  const clean = symbol.split(".")[0];
  if (error) {
    return (
      <div
        className={`rounded-full bg-gradient-to-br from-violet-500/20 to-indigo-500/20 border border-white/10 flex items-center justify-center text-white font-bold shrink-0 ${className}`}
      >
        {clean.charAt(0)}
      </div>
    );
  }
  return (
    <img
      src={`https://logo.clearbit.com/${clean.toLowerCase()}.com`}
      alt={symbol}
      onError={() => setError(true)}
      className={`rounded-full bg-white/5 object-cover shrink-0 shadow-md ${className}`}
    />
  );
}

// ============================================================
// SCORE BAR
// ============================================================

function ScoreBar({ value, color = "#7C3AED", label }: { value: number; color?: string; label?: string }) {
  const pct = Math.max(0, Math.min(1, value)) * 100;
  return (
    <div className="space-y-1">
      {label && <div className="text-[10px] text-gray-500 uppercase tracking-wider">{label}</div>}
      <div className="relative h-1.5 bg-white/5 rounded-full overflow-hidden">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${pct}%` }}
          transition={{ duration: 0.8, ease: "easeOut" }}
          className="absolute inset-y-0 left-0 rounded-full"
          style={{ backgroundColor: color }}
        />
      </div>
    </div>
  );
}

// ============================================================
// METRIC TOOLTIP
// ============================================================

const METRIC_EXPLANATIONS: Record<string, string> = {
  final_score: "Composite AI score blending technical, sentiment, risk, and fundamental signals. Higher is better.",
  technical_score: "Derived from moving averages, RSI, MACD, and volume trends. Reflects short-term price momentum.",
  risk_score: "Volatility and drawdown-based measure. Lower values indicate a safer, more stable investment.",
  "fundamentals.market_cap": "Total market capitalization — price × shares outstanding.",
  "fundamentals.pe_ratio": "Price-to-Earnings ratio. Lower can signal undervaluation, but context matters by sector.",
  "fundamentals.eps": "Earnings Per Share — net income divided by shares outstanding.",
  "fundamentals.revenue_growth": "Year-over-year revenue growth rate. Higher indicates stronger business expansion.",
  "fundamentals.roe": "Return on Equity — measures how efficiently management generates profit from shareholders' equity.",
  "fundamentals.debt_equity": "Debt-to-Equity ratio. Higher values indicate more leverage and financial risk.",
};

function MetricTooltip({ metricKey, children }: { metricKey: string; children: React.ReactNode }) {
  const [show, setShow] = useState(false);
  const explanation = METRIC_EXPLANATIONS[metricKey];
  if (!explanation) return <>{children}</>;
  return (
    <div className="relative inline-flex items-center gap-1 group" onMouseEnter={() => setShow(true)} onMouseLeave={() => setShow(false)}>
      {children}
      <Info size={12} className="text-gray-600 group-hover:text-gray-400 cursor-help transition-colors" />
      <AnimatePresence>
        {show && (
          <motion.div
            initial={{ opacity: 0, y: 4 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 4 }}
            className="absolute bottom-full left-0 mb-2 w-64 bg-[#1a1a2e] border border-white/10 rounded-xl p-3 text-xs text-gray-300 leading-relaxed shadow-2xl z-50 pointer-events-none"
          >
            {explanation}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

// ============================================================
// COMPARISON CHART
// ============================================================

interface ChartProps {
  stocks: StockData[];
  range: ChartRange;
  normalize: boolean;
}

function ComparisonChart({ stocks, range, normalize }: ChartProps) {
  const [hoverIdx, setHoverIdx] = useState<number | null>(null);
  const [tooltipX, setTooltipX] = useState(0);
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const rangeDays: Record<ChartRange, number> = { "1W": 7, "1M": 30, "3M": 90, "1Y": 365 };

  const filtered = useMemo(
    () =>
      stocks.map((s) => ({ ...s, prices: s.prices.slice(-rangeDays[range]) })),
    [stocks, range]
  );

  const maxLen = Math.max(...filtered.map((s) => s.prices.length), 1);

  const chartData = useMemo(
    () =>
      filtered
        .map((stock, si) => {
          const prices = stock.prices;
          if (prices.length < 2) return null;
          const baseline = prices[0].close;
          const points = prices.map((p, i) => ({
            x: (i / (maxLen - 1)) * 100,
            y: normalize ? ((p.close - baseline) / baseline) * 100 : p.close,
            raw: p.close,
            date: p.date,
          }));
          return {
            symbol: stock.symbol,
            color: CHART_COLORS[si],
            points,
            yMin: Math.min(...points.map((p) => p.y)),
            yMax: Math.max(...points.map((p) => p.y)),
          };
        })
        .filter(Boolean) as {
        symbol: string;
        color: string;
        points: { x: number; y: number; raw: number; date: string }[];
        yMin: number;
        yMax: number;
      }[],
    [filtered, normalize, maxLen]
  );

  const globalYMin = Math.min(...chartData.map((d) => d.yMin));
  const globalYMax = Math.max(...chartData.map((d) => d.yMax));
  const yRange = (globalYMax - globalYMin) || 1;

  const toSVGY = (y: number) => 100 - ((y - globalYMin) / yRange) * 100;

  const yTicks = useMemo(() => {
    const count = 5;
    return Array.from({ length: count }, (_, i) => globalYMin + (yRange * i) / (count - 1));
  }, [globalYMin, yRange]);

  const handleMouseMove = useCallback(
    (e: React.MouseEvent) => {
      if (!containerRef.current) return;
      const rect = containerRef.current.getBoundingClientRect();
      const ratio = (e.clientX - rect.left) / rect.width;
      const idx = Math.max(0, Math.min(maxLen - 1, Math.round(ratio * (maxLen - 1))));
      setHoverIdx(idx);
      setTooltipX(ratio * 100);
    },
    [maxLen]
  );

  if (filtered.length === 0 || filtered[0]?.prices.length < 2) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center gap-3 text-gray-600">
        <BarChart size={32} className="opacity-30" />
        <p className="text-sm">Not enough price data for this range</p>
      </div>
    );
  }

  const crosshairX = hoverIdx !== null ? (hoverIdx / (maxLen - 1)) * 100 : null;

  return (
    <div
      className="relative w-full h-full"
      ref={containerRef}
      onMouseMove={handleMouseMove}
      onMouseLeave={() => setHoverIdx(null)}
    >
      {/* Y-axis labels */}
      <div className="absolute left-0 top-0 bottom-0 w-14 flex flex-col justify-between pointer-events-none">
        {[...yTicks].reverse().map((tick, i) => (
          <div key={i} className="text-[10px] text-gray-600 font-mono text-right pr-2">
            {normalize ? `${tick >= 0 ? "+" : ""}${tick.toFixed(1)}%` : fmt.currency(tick)}
          </div>
        ))}
      </div>

      {/* Chart area */}
      <div className="absolute left-16 right-0 top-0 bottom-6">
        <svg
          ref={svgRef}
          viewBox="0 0 100 100"
          preserveAspectRatio="none"
          className="w-full h-full overflow-visible"
        >
          {/* Grid lines */}
          {yTicks.map((tick, i) => (
            <line
              key={i}
              x1="0" y1={toSVGY(tick)}
              x2="100" y2={toSVGY(tick)}
              stroke="rgba(255,255,255,0.04)"
              strokeWidth="0.5"
              vectorEffect="non-scaling-stroke"
            />
          ))}

          {/* Zero line for normalized */}
          {normalize && (
            <line
              x1="0" y1={toSVGY(0)}
              x2="100" y2={toSVGY(0)}
              stroke="rgba(255,255,255,0.12)"
              strokeWidth="1"
              strokeDasharray="3 3"
              vectorEffect="non-scaling-stroke"
            />
          )}

          {/* Area fills */}
          {chartData.map((cd) => {
            const pts = cd.points
              .map((p) => `${p.x},${toSVGY(p.y)}`)
              .join(" ");
            const lastY = toSVGY(cd.points[cd.points.length - 1]?.y ?? globalYMin);
            const fillPts = `${cd.points[0].x},100 ${pts} ${cd.points[cd.points.length - 1].x},100`;
            return (
              <polygon
                key={`fill-${cd.symbol}`}
                points={fillPts}
                fill={cd.color}
                fillOpacity="0.06"
              />
            );
          })}

          {/* Lines */}
          {chartData.map((cd, si) => {
            const pts = cd.points.map((p) => `${p.x},${toSVGY(p.y)}`).join(" ");
            return (
              <motion.polyline
                key={cd.symbol}
                initial={{ pathLength: 0, opacity: 0 }}
                animate={{ pathLength: 1, opacity: 1 }}
                transition={{ duration: 0.9, delay: si * 0.1, ease: "easeOut" }}
                points={pts}
                fill="none"
                stroke={cd.color}
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
                vectorEffect="non-scaling-stroke"
              />
            );
          })}

          {/* Hover crosshair */}
          {crosshairX !== null && (
            <>
              <line
                x1={crosshairX} y1="0"
                x2={crosshairX} y2="100"
                stroke="rgba(255,255,255,0.15)"
                strokeWidth="1"
                strokeDasharray="2 2"
                vectorEffect="non-scaling-stroke"
              />
              {chartData.map((cd) => {
                const pt = cd.points[hoverIdx!] ?? cd.points[cd.points.length - 1];
                if (!pt) return null;
                return (
                  <circle
                    key={`dot-${cd.symbol}`}
                    cx={pt.x}
                    cy={toSVGY(pt.y)}
                    r="3"
                    fill={cd.color}
                    stroke="#050505"
                    strokeWidth="1.5"
                    vectorEffect="non-scaling-stroke"
                  />
                );
              })}
            </>
          )}
        </svg>

        {/* Cursor-following tooltip */}
        <AnimatePresence>
          {hoverIdx !== null && chartData[0]?.points[hoverIdx] && (
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.95 }}
              transition={{ duration: 0.1 }}
              className="absolute top-2 pointer-events-none z-20"
              style={{
                left: `${Math.min(tooltipX, 70)}%`,
                transform: tooltipX > 60 ? "translateX(-110%)" : "translateX(8px)",
              }}
            >
              <div className="bg-[#0d0d1a]/95 backdrop-blur-md border border-white/10 p-3 rounded-xl shadow-2xl min-w-[150px]">
                <div className="text-[10px] text-gray-500 font-mono mb-2 flex items-center gap-1">
                  <Clock size={9} />
                  {chartData[0].points[hoverIdx]?.date}
                </div>
                <div className="space-y-1.5">
                  {chartData.map((cd) => {
                    const pt = cd.points[hoverIdx] ?? cd.points[cd.points.length - 1];
                    if (!pt) return null;
                    return (
                      <div key={cd.symbol} className="flex items-center justify-between gap-4">
                        <div className="flex items-center gap-1.5">
                          <div className="w-2 h-2 rounded-full" style={{ backgroundColor: cd.color }} />
                          <span className="text-xs font-bold">{cd.symbol}</span>
                        </div>
                        <span className="text-xs font-mono">
                          {normalize ? fmt.percent(pt.y) : fmt.currency(pt.raw)}
                        </span>
                      </div>
                    );
                  })}
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* X-axis date labels */}
      <div className="absolute left-16 right-0 bottom-0 flex justify-between">
        {[0, Math.floor(maxLen / 3), Math.floor((maxLen * 2) / 3), maxLen - 1].map((idx) => {
          const date = filtered[0]?.prices[idx]?.date;
          return (
            <div key={idx} className="text-[9px] text-gray-600 font-mono">
              {date ? new Date(date).toLocaleDateString("en-US", { month: "short", day: "numeric" }) : ""}
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ============================================================
// METRICS TABLE
// ============================================================

const METRIC_GROUPS = [
  {
    label: "AI Intelligence",
    icon: <Sparkles size={12} />,
    metrics: [
      { label: "Overall Score", key: "final_score", format: (v: number) => fmt.score(v), isBar: true },
      { label: "Technical Score", key: "technical_score", format: (v: number) => fmt.score(v), isBar: true },
      { label: "Sentiment Score", key: "sentiment_score", format: (v: number) => fmt.score(v), isBar: true },
    ],
  },
  {
    label: "Fundamentals",
    icon: <BarChart size={12} />,
    metrics: [
      { label: "Market Cap", key: "fundamentals.market_cap", format: fmt.currency },
      { label: "P/E Ratio", key: "fundamentals.pe_ratio", format: fmt.number },
      { label: "EPS", key: "fundamentals.eps", format: fmt.currency },
      { label: "Revenue Growth", key: "fundamentals.revenue_growth", format: fmt.percent },
      { label: "Return on Equity", key: "fundamentals.roe", format: fmt.percent },
    ],
  },
  {
    label: "Risk",
    icon: <Shield size={12} />,
    metrics: [
      { label: "Risk Score", key: "risk_score", format: (v: number) => fmt.score(v), isBar: true, invertWinner: true },
      { label: "Debt / Equity", key: "fundamentals.debt_equity", format: fmt.number, invertWinner: true },
    ],
  },
];

function MetricsTable({ stocks, queries }: { stocks: StockData[]; queries: { isLoading: boolean }[] }) {
  const [expandedGroups, setExpandedGroups] = useState<Record<string, boolean>>({ "AI Intelligence": true, Fundamentals: true, Risk: true });

  function getWinnerAnalysis(key: string, value: number | undefined, invertWinner = false) {
    if (stocks.length < 2 || value == null) return { color: "text-gray-200", isWinner: false, isLoser: false };
    const vals = stocks.map((d) => getAccessorValue(d, key)).filter((v): v is number => v != null);
    if (vals.length < 2) return { color: "text-gray-200", isWinner: false, isLoser: false };
    const best = invertWinner ? Math.min(...vals) : Math.max(...vals);
    const worst = invertWinner ? Math.max(...vals) : Math.min(...vals);
    if (value === best) return { color: "text-emerald-400", isWinner: true, isLoser: false };
    if (value === worst) return { color: "text-rose-400", isWinner: false, isLoser: true };
    return { color: "text-gray-300", isWinner: false, isLoser: false };
  }

  return (
    <div className="bg-[#0d0d1a] border border-white/5 rounded-2xl overflow-hidden">
      <div className="px-6 py-4 border-b border-white/5 flex items-center justify-between">
        <h3 className="text-sm font-semibold flex items-center gap-2 text-white">
          <BarChart2 size={15} className="text-violet-400" /> Metrics Matrix
        </h3>
        <div className="flex items-center gap-1 text-[10px] text-gray-500">
          <div className="w-2 h-2 rounded-full bg-emerald-400" /> Best
          <div className="w-2 h-2 rounded-full bg-rose-400 ml-2" /> Worst
        </div>
      </div>

      {/* Stock Headers */}
      <div className="grid border-b border-white/5" style={{ gridTemplateColumns: `180px repeat(${stocks.length}, 1fr)` }}>
        <div className="px-4 py-3" />
        {stocks.map((s, i) => (
          <div key={s.symbol} className="px-4 py-3 flex items-center gap-2">
            <CompanyAvatar symbol={s.symbol} className="w-7 h-7" />
            <span className="text-sm font-bold" style={{ color: CHART_COLORS[i] }}>{s.symbol}</span>
          </div>
        ))}
      </div>

      {METRIC_GROUPS.map((group) => (
        <div key={group.label}>
          <button
            onClick={() => setExpandedGroups((prev) => ({ ...prev, [group.label]: !prev[group.label] }))}
            className="w-full px-4 py-2.5 flex items-center justify-between bg-white/[0.02] hover:bg-white/[0.04] transition-colors border-b border-white/5"
          >
            <div className="flex items-center gap-2 text-[11px] text-gray-400 font-semibold uppercase tracking-wider">
              <span className="text-violet-400">{group.icon}</span>
              {group.label}
            </div>
            {expandedGroups[group.label] ? <ChevronUp size={12} className="text-gray-600" /> : <ChevronDown size={12} className="text-gray-600" />}
          </button>

          <AnimatePresence>
            {expandedGroups[group.label] && (
              <motion.div
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: "auto", opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                transition={{ duration: 0.2 }}
                className="overflow-hidden"
              >
                {group.metrics.map((metric, mi) => (
                  <div
                    key={metric.key}
                    className="grid hover:bg-white/[0.02] transition-colors border-b border-white/[0.03]"
                    style={{ gridTemplateColumns: `180px repeat(${Math.max(stocks.length, 1)}, 1fr)` }}
                  >
                    <div className="px-4 py-3">
                      <MetricTooltip metricKey={metric.key}>
                        <span className="text-xs text-gray-500">{metric.label}</span>
                      </MetricTooltip>
                    </div>

                    {stocks.length === 0
                      ? Array.from({ length: 2 }).map((_, i) => (
                          <div key={i} className="px-4 py-3">
                            <div className="h-3 w-12 bg-white/5 rounded animate-pulse" />
                          </div>
                        ))
                      : stocks.map((s, si) => {
                          const isLoading = queries[
                            (queries as { isLoading: boolean; data?: StockData }[]).findIndex(
                              (q) => (q as { data?: StockData }).data?.symbol === s.symbol
                            )
                          ]?.isLoading;

                          if (isLoading) {
                            return (
                              <div key={s.symbol} className="px-4 py-3">
                                <div className="h-3 w-12 bg-white/5 rounded animate-pulse" />
                              </div>
                            );
                          }

                          const val = getAccessorValue(s, metric.key);
                          const { color, isWinner, isLoser } = getWinnerAnalysis(
                            metric.key,
                            val,
                            (metric as { invertWinner?: boolean }).invertWinner
                          );

                          return (
                            <div key={s.symbol} className="px-4 py-3 space-y-1.5">
                              <div className="flex items-center gap-1.5">
                                <span className={`text-sm font-mono font-semibold ${color}`}>
                                  {val != null ? metric.format(val) : "—"}
                                </span>
                                {isWinner && <Trophy size={11} className="text-yellow-400 shrink-0" />}
                                {isLoser && <AlertTriangle size={11} className="text-rose-500 shrink-0" />}
                              </div>
                              {(metric as { isBar?: boolean }).isBar && val != null && (
                                <ScoreBar
                                  value={val}
                                  color={
                                    isWinner
                                      ? "#34d399"
                                      : isLoser
                                      ? "#f87171"
                                      : CHART_COLORS[si]
                                  }
                                />
                              )}
                            </div>
                          );
                        })}
                  </div>
                ))}
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      ))}
    </div>
  );
}

// ============================================================
// AI SUMMARY CARDS
// ============================================================

function AISummary({ stocks }: { stocks: StockData[] }) {
  const summary = useMemo(() => {
    if (stocks.length < 2) return null;
    const best = [...stocks].sort((a, b) => b.final_score - a.final_score)[0];
    const safest = [...stocks].sort((a, b) => a.risk_score - b.risk_score)[0];
    const riskiest = [...stocks].sort((a, b) => b.risk_score - a.risk_score)[0];
    const valuePick = [...stocks]
      .filter((d) => d.fundamentals?.pe_ratio != null)
      .sort((a, b) => a.fundamentals.pe_ratio! - b.fundamentals.pe_ratio!)[0];
    const growth = [...stocks]
      .filter((d) => d.fundamentals?.revenue_growth != null)
      .sort((a, b) => b.fundamentals.revenue_growth - a.fundamentals.revenue_growth)[0];

    return { best, safest, riskiest, valuePick, growth };
  }, [stocks]);

  const cards = summary
    ? [
        {
          icon: <Trophy size={16} className="text-yellow-400" />,
          label: "Best Performer",
          symbol: summary.best.symbol,
          color: CHART_COLORS[stocks.indexOf(summary.best)],
          desc: `Leads with a ${fmt.score(summary.best.final_score)}/100 AI score and strong technical momentum.`,
          confidence: summary.best.final_score,
          badge: "Outperformer",
          badgeColor: "text-yellow-400 bg-yellow-500/10 border-yellow-500/20",
        },
        {
          icon: <Shield size={16} className="text-blue-400" />,
          label: "Lowest Risk",
          symbol: summary.safest.symbol,
          color: CHART_COLORS[stocks.indexOf(summary.safest)],
          desc: `Risk score of ${fmt.score(summary.safest.risk_score)}/100 — the most stable choice in this group.`,
          confidence: 1 - summary.safest.risk_score,
          badge: "Safe Haven",
          badgeColor: "text-blue-400 bg-blue-500/10 border-blue-500/20",
        },
        ...(summary.valuePick
          ? [
              {
                icon: <Target size={16} className="text-emerald-400" />,
                label: "Best Value",
                symbol: summary.valuePick.symbol,
                color: CHART_COLORS[stocks.indexOf(summary.valuePick)],
                desc: `Trades at ${fmt.number(summary.valuePick.fundamentals.pe_ratio)}x P/E — the most attractively priced on earnings.`,
                confidence: 0.72,
                badge: "Value Play",
                badgeColor: "text-emerald-400 bg-emerald-500/10 border-emerald-500/20",
              },
            ]
          : []),
        ...(summary.growth
          ? [
              {
                icon: <Trend size={16} className="text-violet-400" />,
                label: "Top Growth",
                symbol: summary.growth.symbol,
                color: CHART_COLORS[stocks.indexOf(summary.growth)],
                desc: `Revenue growing at ${fmt.percent(summary.growth.fundamentals.revenue_growth)} YoY — highest expansion rate.`,
                confidence: Math.min(summary.growth.fundamentals.revenue_growth / 50, 1),
                badge: "High Growth",
                badgeColor: "text-violet-400 bg-violet-500/10 border-violet-500/20",
              },
            ]
          : []),
      ]
    : [];

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2 mb-1">
        <Sparkles size={14} className="text-violet-400" />
        <h3 className="text-sm font-semibold text-white">AI Synthesis</h3>
      </div>

      {stocks.length < 2 ? (
        <div className="p-4 bg-white/[0.02] border border-white/5 rounded-xl text-sm text-gray-500 text-center">
          Add a second stock to unlock comparative AI insights
        </div>
      ) : (
        <div className="grid grid-cols-2 gap-3">
          {cards.map((card, i) => (
            <motion.div
              key={card.label}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.08 }}
              className="bg-[#0d0d1a] border border-white/5 rounded-xl p-4 space-y-3 hover:border-white/10 transition-colors"
            >
              <div className="flex items-start justify-between">
                <div className="flex items-center gap-1.5">
                  {card.icon}
                  <span className="text-[10px] text-gray-500 uppercase tracking-wider font-semibold">{card.label}</span>
                </div>
                <span className={`text-[9px] font-bold px-1.5 py-0.5 rounded-md border ${card.badgeColor}`}>{card.badge}</span>
              </div>

              <div className="flex items-center gap-2">
                <CompanyAvatar symbol={card.symbol} className="w-7 h-7" />
                <span className="text-lg font-bold" style={{ color: card.color }}>{card.symbol}</span>
              </div>

              <p className="text-[11px] text-gray-400 leading-relaxed">{card.desc}</p>

              <div className="space-y-1">
                <div className="flex justify-between text-[10px] text-gray-600">
                  <span>Confidence</span>
                  <span>{Math.round(card.confidence * 100)}%</span>
                </div>
                <ScoreBar value={card.confidence} color={card.color} />
              </div>
            </motion.div>
          ))}
        </div>
      )}
    </div>
  );
}

// ============================================================
// NEWS PANEL
// ============================================================

function NewsPanel({ newsQueries, symbols }: { newsQueries: { data?: NewsItem[]; isLoading: boolean }[]; symbols: string[] }) {
  const [filterSymbol, setFilterSymbol] = useState<string>("All");

  const allNews: NewsItem[] = newsQueries
    .flatMap((q) => q.data || [])
    .sort((a, b) => {
      if (a.publishedAt && b.publishedAt) return new Date(b.publishedAt).getTime() - new Date(a.publishedAt).getTime();
      return 0;
    });

  const filtered = filterSymbol === "All" ? allNews : allNews.filter((n) => n.symbol === filterSymbol);
  const featured = filtered[0];
  const rest = filtered.slice(1, 8);
  const isLoading = newsQueries.some((q) => q.isLoading);

  return (
    <div className="bg-[#0d0d1a] border border-white/5 rounded-2xl overflow-hidden">
      <div className="px-5 py-4 border-b border-white/5 flex items-center justify-between">
        <h3 className="text-sm font-semibold flex items-center gap-2">
          <Newspaper size={14} className="text-blue-400" /> News Feed
        </h3>
        <div className="flex items-center gap-1.5">
          <Filter size={11} className="text-gray-600" />
          <div className="flex gap-1">
            {["All", ...symbols].map((sym) => (
              <button
                key={sym}
                onClick={() => setFilterSymbol(sym)}
                className={`text-[10px] px-2 py-1 rounded-md font-bold transition-all ${
                  filterSymbol === sym
                    ? "bg-blue-500/20 text-blue-300 border border-blue-500/30"
                    : "text-gray-600 hover:text-gray-400"
                }`}
              >
                {sym}
              </button>
            ))}
          </div>
        </div>
      </div>

      {isLoading ? (
        <div className="flex items-center justify-center gap-2 text-gray-600 py-10">
          <RefreshCcw size={16} className="animate-spin" />
          <span className="text-sm">Fetching headlines…</span>
        </div>
      ) : filtered.length === 0 ? (
        <div className="py-10 text-center text-sm text-gray-600">No news found for this filter.</div>
      ) : (
        <div className="p-4 space-y-3">
          {/* Featured */}
          {featured && (
            <a
              href={featured.link}
              target="_blank"
              rel="noreferrer"
              className="block p-4 bg-white/[0.03] hover:bg-white/[0.06] border border-white/8 rounded-xl transition-colors group"
            >
              <div className="flex items-center gap-2 mb-2">
                <span className="text-[10px] font-bold text-blue-300 bg-blue-500/10 border border-blue-500/20 px-2 py-0.5 rounded uppercase">{featured.symbol}</span>
                <span className={`text-[10px] font-bold px-2 py-0.5 rounded border flex items-center gap-1 ${getSentimentColor(featured.sentiment)}`}>
                  {getSentimentIcon(featured.sentiment)}
                  {featured.sentiment ?? "neutral"}
                </span>
                {featured.publishedAt && (
                  <span className="text-[10px] text-gray-600 ml-auto">{timeAgo(featured.publishedAt)}</span>
                )}
              </div>
              <p className="text-sm font-semibold text-gray-200 group-hover:text-white transition-colors leading-snug">{featured.title}</p>
              {featured.source && <p className="text-[10px] text-gray-600 mt-1.5">{featured.source}</p>}
            </a>
          )}

          {/* Rest */}
          <div className="space-y-2">
            {rest.map((news, i) => (
              <a
                key={i}
                href={news.link}
                target="_blank"
                rel="noreferrer"
                className="flex items-start gap-3 p-3 bg-white/[0.015] hover:bg-white/[0.04] rounded-xl border border-white/[0.04] transition-colors group"
              >
                <div className="shrink-0 mt-0.5">
                  <span className={`text-[9px] font-bold px-1.5 py-0.5 rounded border flex items-center gap-0.5 ${getSentimentColor(news.sentiment)}`}>
                    {getSentimentIcon(news.sentiment)}
                  </span>
                </div>
                <div className="min-w-0">
                  <p className="text-xs text-gray-400 group-hover:text-gray-200 leading-snug line-clamp-2 transition-colors">{news.title}</p>
                  <div className="flex items-center gap-2 mt-1">
                    <span className="text-[9px] text-gray-600 font-bold">{news.symbol}</span>
                    {news.publishedAt && <span className="text-[9px] text-gray-700">{timeAgo(news.publishedAt)}</span>}
                  </div>
                </div>
              </a>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// ============================================================
// ALERT MODAL
// ============================================================

function AlertModal({
  symbol,
  onClose,
  alerts,
  onSave,
}: {
  symbol: string;
  onClose: () => void;
  alerts: AlertConfig[];
  onSave: (alert: AlertConfig) => void;
}) {
  const [type, setType] = useState<AlertConfig["type"]>("price_above");
  const [value, setValue] = useState("");
  const [saved, setSaved] = useState(false);

  const existing = alerts.filter((a) => a.symbol === symbol && a.active);

  const ALERT_TYPES: { value: AlertConfig["type"]; label: string; desc: string; unit: string }[] = [
    { value: "price_above", label: "Price rises above", desc: "Triggers when price exceeds your target", unit: "$" },
    { value: "price_below", label: "Price drops below", desc: "Triggers when price falls under your target", unit: "$" },
    { value: "score_change", label: "AI score drops below", desc: "Alert when AI score falls under threshold", unit: "pts" },
    { value: "risk_spike", label: "Risk score exceeds", desc: "Notifies on elevated risk detection", unit: "%" },
  ];

  function handleSave() {
    if (!value || isNaN(Number(value))) return;
    onSave({
      id: `alert-${Date.now()}`,
      symbol,
      type,
      value: Number(value),
      createdAt: new Date().toISOString(),
      active: true,
    });
    setSaved(true);
    setTimeout(() => { setSaved(false); onClose(); }, 1500);
  }

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-4"
      onClick={onClose}
    >
      <motion.div
        initial={{ scale: 0.95, y: 10 }}
        animate={{ scale: 1, y: 0 }}
        exit={{ scale: 0.95, y: 10 }}
        className="bg-[#0d0d1a] border border-white/10 rounded-2xl p-6 w-full max-w-md shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-violet-500/10 border border-violet-500/20 flex items-center justify-center">
              <Bell size={16} className="text-violet-400" />
            </div>
            <div>
              <h3 className="font-bold text-white">Set Price Alert</h3>
              <p className="text-xs text-gray-500">{symbol}</p>
            </div>
          </div>
          <button onClick={onClose} className="w-8 h-8 rounded-lg bg-white/5 hover:bg-white/10 flex items-center justify-center transition-colors">
            <X size={14} className="text-gray-400" />
          </button>
        </div>

        {/* Alert type selector */}
        <div className="space-y-2 mb-5">
          {ALERT_TYPES.map((at) => (
            <button
              key={at.value}
              onClick={() => setType(at.value)}
              className={`w-full flex items-center gap-3 p-3 rounded-xl border transition-all text-left ${
                type === at.value
                  ? "border-violet-500/40 bg-violet-500/10"
                  : "border-white/5 bg-white/[0.02] hover:bg-white/[0.04]"
              }`}
            >
              <div className={`w-4 h-4 rounded-full border-2 flex items-center justify-center transition-colors ${type === at.value ? "border-violet-400" : "border-gray-600"}`}>
                {type === at.value && <div className="w-2 h-2 rounded-full bg-violet-400" />}
              </div>
              <div>
                <div className="text-sm font-medium text-white">{at.label}</div>
                <div className="text-[10px] text-gray-500">{at.desc}</div>
              </div>
            </button>
          ))}
        </div>

        {/* Value input */}
        <div className="mb-5">
          <label className="text-xs text-gray-500 mb-2 block">
            {ALERT_TYPES.find((a) => a.value === type)?.label} ({ALERT_TYPES.find((a) => a.value === type)?.unit})
          </label>
          <div className="flex items-center gap-2 bg-white/[0.03] border border-white/10 rounded-xl px-4 py-3 focus-within:border-violet-500/50 transition-colors">
            <span className="text-gray-500 text-sm">{ALERT_TYPES.find((a) => a.value === type)?.unit}</span>
            <input
              type="number"
              value={value}
              onChange={(e) => setValue(e.target.value)}
              placeholder="0.00"
              className="bg-transparent outline-none flex-1 text-white text-sm font-mono"
            />
          </div>
        </div>

        {/* Existing alerts */}
        {existing.length > 0 && (
          <div className="mb-5 space-y-2">
            <p className="text-[10px] text-gray-500 uppercase tracking-wider font-semibold">Active Alerts</p>
            {existing.map((a) => (
              <div key={a.id} className="flex items-center justify-between bg-white/[0.02] border border-white/5 rounded-lg px-3 py-2">
                <div className="text-xs text-gray-400">
                  {ALERT_TYPES.find((t) => t.value === a.type)?.label} {ALERT_TYPES.find((t) => t.value === a.type)?.unit}{a.value}
                </div>
                <CheckCircle2 size={13} className="text-emerald-400" />
              </div>
            ))}
          </div>
        )}

        <button
          onClick={handleSave}
          disabled={!value || saved}
          className={`w-full py-3 rounded-xl font-semibold text-sm transition-all ${
            saved
              ? "bg-emerald-500 text-white"
              : "bg-violet-600 hover:bg-violet-500 text-white disabled:opacity-40 disabled:cursor-not-allowed"
          }`}
        >
          {saved ? (
            <span className="flex items-center justify-center gap-2">
              <CheckCircle2 size={15} /> Alert Saved!
            </span>
          ) : (
            "Set Alert"
          )}
        </button>

        <p className="text-center text-[10px] text-gray-600 mt-3">
          Alerts are stored in your browser and will notify you when conditions are met.
        </p>
      </motion.div>
    </motion.div>
  );
}

// ============================================================
// SHARE / SAVE MODAL
// ============================================================

function ShareModal({ symbols, onClose }: { symbols: string[]; onClose: () => void }) {
  const url = typeof window !== "undefined" ? `${window.location.origin}/compare?symbols=${symbols.join(",")}` : "";
  const [copied, setCopied] = useState(false);

  function copy() {
    navigator.clipboard.writeText(url);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-4"
      onClick={onClose}
    >
      <motion.div
        initial={{ scale: 0.95, y: 10 }}
        animate={{ scale: 1, y: 0 }}
        exit={{ scale: 0.95, y: 10 }}
        className="bg-[#0d0d1a] border border-white/10 rounded-2xl p-6 w-full max-w-sm shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between mb-4">
          <h3 className="font-bold text-white flex items-center gap-2">
            <Share2 size={16} className="text-blue-400" /> Share Comparison
          </h3>
          <button onClick={onClose} className="w-7 h-7 rounded-lg bg-white/5 flex items-center justify-center">
            <X size={13} className="text-gray-400" />
          </button>
        </div>
        <p className="text-xs text-gray-500 mb-4">Anyone with this link can view this exact comparison.</p>
        <div className="flex items-center gap-2 bg-white/[0.03] border border-white/10 rounded-xl p-3 mb-4">
          <p className="text-xs text-gray-400 truncate flex-1 font-mono">{url}</p>
          <button onClick={copy} className={`text-xs px-3 py-1.5 rounded-lg font-bold transition-all ${copied ? "bg-emerald-500/20 text-emerald-400" : "bg-white/10 text-white hover:bg-white/20"}`}>
            {copied ? "Copied!" : "Copy"}
          </button>
        </div>
      </motion.div>
    </motion.div>
  );
}

// ============================================================
// STOCK CHIP
// ============================================================

function StockChip({
  symbol,
  index,
  onRemove,
  onAlertClick,
  isLoading,
  data,
}: {
  symbol: string;
  index: number;
  onRemove: () => void;
  onAlertClick: () => void;
  isLoading: boolean;
  data?: StockData;
}) {
  const color = CHART_COLORS[index];
  const price = data?.prices?.[data.prices.length - 1]?.close;
  const prevPrice = data?.prices?.[data.prices.length - 2]?.close;
  const change = price && prevPrice ? ((price - prevPrice) / prevPrice) * 100 : null;

  return (
    <motion.div
      layout
      initial={{ opacity: 0, scale: 0.85 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.85 }}
      className="flex items-center gap-2 px-3 py-2 rounded-xl border bg-[#0d0d1a] hover:border-white/15 transition-all"
      style={{ borderColor: `${color}40` }}
    >
      {isLoading ? (
        <div className="w-5 h-5 rounded-full bg-white/10 animate-pulse" />
      ) : (
        <CompanyAvatar symbol={symbol} className="w-5 h-5" />
      )}

      <div className="flex items-center gap-1.5">
        <span className="text-sm font-bold" style={{ color }}>{symbol}</span>
        {price && !isLoading && (
          <>
            <span className="text-xs text-gray-500 font-mono">{fmt.currency(price)}</span>
            {change !== null && (
              <span className={`text-[10px] font-bold ${change >= 0 ? "text-emerald-400" : "text-rose-400"}`}>
                {change >= 0 ? <ArrowUpRight size={10} className="inline" /> : <ArrowDownRight size={10} className="inline" />}
                {Math.abs(change).toFixed(2)}%
              </span>
            )}
          </>
        )}
      </div>

      <button
        onClick={onAlertClick}
        className="w-5 h-5 rounded flex items-center justify-center text-gray-600 hover:text-violet-400 transition-colors"
        title="Set alert"
      >
        <Bell size={11} />
      </button>
      <button
        onClick={onRemove}
        className="w-5 h-5 rounded flex items-center justify-center text-gray-600 hover:text-rose-400 transition-colors"
        title="Remove"
      >
        <X size={11} />
      </button>
    </motion.div>
  );
}

// ============================================================
// SEARCH & ADD BAR
// ============================================================

function SearchAddBar({
  activeSymbols,
  onAdd,
  stockQueries,
}: {
  activeSymbols: string[];
  onAdd: (sym: string) => void;
  stockQueries: { data?: StockData; isLoading: boolean }[];
}) {
  const [input, setInput] = useState("");
  const [suggestions, setSuggestions] = useState<SearchSuggestion[]>([]);
  const [searching, setSearching] = useState(false);
  const queryClient = useQueryClient();

  useEffect(() => {
    if (input.length < 2) { setSuggestions([]); return; }
    const t = setTimeout(async () => {
      try {
        setSearching(true);
        const res = await fetch(`${API_BASE}/search/${encodeURIComponent(input)}`);
        const data = await res.json();
        setSuggestions(Array.isArray(data) ? data : []);
      } catch { setSuggestions([]); }
      finally { setSearching(false); }
    }, 300);
    return () => clearTimeout(t);
  }, [input]);

  const atMax = activeSymbols.length >= 4;

  return (
    <div className="relative">
      <div className={`flex items-center gap-3 p-3 rounded-xl border bg-[#0d0d1a] transition-all ${atMax ? "opacity-50 border-white/5" : "border-white/10 focus-within:border-violet-500/50"}`}>
        <Search size={16} className="text-gray-500 shrink-0" />
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && suggestions.length > 0) {
              const s = suggestions[0];
              if (!activeSymbols.includes(s.symbol)) onAdd(s.symbol);
              setInput("");
              setSuggestions([]);
            }
          }}
          placeholder={atMax ? "Max 4 stocks reached" : "Search by symbol or name…"}
          disabled={atMax}
          className="bg-transparent outline-none flex-1 text-sm placeholder:text-gray-600"
        />
        {input && (
          <button onClick={() => { setInput(""); setSuggestions([]); }} className="text-gray-600 hover:text-gray-400">
            <X size={14} />
          </button>
        )}
      </div>

      <AnimatePresence>
        {input.length >= 2 && (
          <motion.div
            initial={{ opacity: 0, y: 4 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 4 }}
            className="absolute z-50 w-full mt-1.5 bg-[#0d0d1a] border border-white/10 rounded-xl overflow-hidden shadow-2xl"
          >
            {searching && <div className="p-4 text-sm text-gray-500 flex items-center gap-2"><RefreshCcw size={12} className="animate-spin" /> Searching…</div>}
            {!searching && suggestions.length === 0 && input.length >= 2 && (
              <div className="p-4 text-sm text-gray-600">No results for "{input}"</div>
            )}
            {!searching && suggestions.map((s) => {
              const alreadyAdded = activeSymbols.includes(s.symbol);
              return (
                <div
                  key={s.symbol}
                  onClick={() => {
                    if (!alreadyAdded && !atMax) { onAdd(s.symbol); }
                    setInput("");
                    setSuggestions([]);
                  }}
                  onMouseEnter={() =>
                    queryClient.prefetchQuery({
                      queryKey: ["stock", s.symbol],
                      queryFn: () => fetch(`${API_BASE}/stock/${s.symbol}`).then((r) => r.json()),
                    })
                  }
                  className={`flex items-center justify-between px-4 py-3 transition-colors ${
                    alreadyAdded ? "opacity-40 cursor-default" : "hover:bg-white/5 cursor-pointer"
                  }`}
                >
                  <div className="flex items-center gap-3">
                    <CompanyAvatar symbol={s.symbol} className="w-7 h-7" />
                    <div>
                      <span className="text-sm font-bold text-white">{s.symbol}</span>
                      <span className="text-xs text-gray-500 ml-2">{s.name}</span>
                    </div>
                  </div>
                  {alreadyAdded ? (
                    <CheckCircle2 size={14} className="text-emerald-500" />
                  ) : (
                    <Plus size={14} className="text-gray-500" />
                  )}
                </div>
              );
            })}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

// ============================================================
// MAIN PAGE
// ============================================================

export default function ComparePage() {
  const router = useRouter();
  const searchParams = useSearchParams();

  const urlSymbols = useMemo(() => searchParams.get("symbols")?.split(",").filter(Boolean) ?? [], [searchParams]);
  const [activeSymbols, setActiveSymbols] = useState<string[]>(urlSymbols);
  const [chartRange, setChartRange] = useState<ChartRange>("1M");
  const [normalizeChart, setNormalizeChart] = useState(true);
  const [alertModal, setAlertModal] = useState<string | null>(null);
  const [showShareModal, setShowShareModal] = useState(false);
  const [alerts, setAlerts] = useState<AlertConfig[]>(() => {
    if (typeof window === "undefined") return [];
    try { return JSON.parse(localStorage.getItem("stock-alerts") || "[]"); } catch { return []; }
  });
  const [savedComparisons, setSavedComparisons] = useState<SavedComparison[]>(() => {
    if (typeof window === "undefined") return [];
    try { return JSON.parse(localStorage.getItem("saved-comparisons") || "[]"); } catch { return []; }
  });
  const [savedBanner, setSavedBanner] = useState(false);

  // Sync URL
  useEffect(() => {
    const current = searchParams.get("symbols") || "";
    const next = activeSymbols.join(",");
    if (current !== next) {
      const p = new URLSearchParams(searchParams.toString());
      if (activeSymbols.length > 0) p.set("symbols", next);
      else p.delete("symbols");
      router.replace(`/compare?${p.toString()}`, { scroll: false });
    }
  }, [activeSymbols, router, searchParams]);

  // Alert checker (polls every 30s)
  useEffect(() => {
    if (alerts.length === 0) return;
    const interval = setInterval(() => {
      const updated = alerts.map((alert) => {
        if (!alert.active || alert.triggered) return alert;
        const stockData = stockQueries.find((q) => (q as { data?: StockData }).data?.symbol === alert.symbol);
        const data = (stockData as { data?: StockData })?.data;
        if (!data) return alert;
        const price = data.prices[data.prices.length - 1]?.close;
        let triggered = false;
        if (alert.type === "price_above" && price > alert.value) triggered = true;
        if (alert.type === "price_below" && price < alert.value) triggered = true;
        if (alert.type === "score_change" && data.final_score * 100 < alert.value) triggered = true;
        if (alert.type === "risk_spike" && data.risk_score * 100 > alert.value) triggered = true;
        if (triggered) {
          // Browser notification
          if (typeof Notification !== "undefined" && Notification.permission === "granted") {
            new Notification(`📊 ${alert.symbol} Alert Triggered`, {
              body: `Condition met: ${alert.type.replace("_", " ")} ${alert.value}`,
              icon: `https://logo.clearbit.com/${alert.symbol.toLowerCase()}.com`,
            });
          }
          return { ...alert, triggered: true, triggeredAt: new Date().toISOString() };
        }
        return alert;
      });
      setAlerts(updated);
      localStorage.setItem("stock-alerts", JSON.stringify(updated));
    }, 30000);
    return () => clearInterval(interval);
  }, [alerts]);

  // Request notification permission on first alert set
  function requestNotifPermission() {
    if (typeof Notification !== "undefined" && Notification.permission === "default") {
      Notification.requestPermission();
    }
  }

  function addSymbol(sym: string) {
    if (!activeSymbols.includes(sym) && activeSymbols.length < 4) {
      setActiveSymbols((prev) => [...prev, sym]);
    }
  }

  function removeSymbol(sym: string) {
    setActiveSymbols((prev) => prev.filter((s) => s !== sym));
  }

  function saveAlert(alert: AlertConfig) {
    requestNotifPermission();
    const next = [...alerts, alert];
    setAlerts(next);
    localStorage.setItem("stock-alerts", JSON.stringify(next));
  }

  function saveComparison() {
    const name = activeSymbols.join(" vs ");
    const comp: SavedComparison = { id: `comp-${Date.now()}`, name, symbols: activeSymbols, savedAt: new Date().toISOString() };
    const next = [comp, ...savedComparisons].slice(0, 10);
    setSavedComparisons(next);
    localStorage.setItem("saved-comparisons", JSON.stringify(next));
    setSavedBanner(true);
    setTimeout(() => setSavedBanner(false), 2500);
  }

  // Data fetching
  const stockQueries = useQueries({
    queries: activeSymbols.map((sym) => ({
      queryKey: ["stock", sym],
      queryFn: async (): Promise<StockData> => {
        const res = await fetch(`${API_BASE}/stock/${sym}`);
        if (!res.ok) throw new Error(`Failed: ${sym}`);
        return res.json();
      },
      staleTime: 5 * 60 * 1000,
    })),
  });

  const newsQueries = useQueries({
    queries: activeSymbols.map((sym) => ({
      queryKey: ["news", sym],
      queryFn: async (): Promise<NewsItem[]> => {
        const res = await fetch(`${API_BASE}/news/${sym}`);
        if (!res.ok) return [];
        const data = await res.json();
        return (data.news || []).map((n: NewsItem) => ({ ...n, symbol: sym }));
      },
      staleTime: 5 * 60 * 1000,
    })),
  });

  const fetchedData = stockQueries.map((q) => q.data).filter(Boolean) as StockData[];
  const activeAlerts = alerts.filter((a) => activeSymbols.includes(a.symbol) && a.active && !a.triggered);
  const triggeredAlerts = alerts.filter((a) => activeSymbols.includes(a.symbol) && a.triggered);

  return (
    <main className="min-h-screen bg-[#050508] text-white font-sans pb-20">
      <div className="max-w-7xl mx-auto px-4 md:px-6 pt-28">

        {/* ── HEADER ── */}
        <div className="flex flex-col md:flex-row md:items-start justify-between gap-6 mb-8">
          <div>
            <h1 className="text-3xl font-bold tracking-tight flex items-center gap-3 mb-1">
              <BarChart2 size={28} className="text-violet-400" /> Market Compare
            </h1>
            <p className="text-sm text-gray-500">Side-by-side AI analysis. Up to 4 assets.</p>
          </div>

          {activeSymbols.length > 0 && (
            <div className="flex items-center gap-2 shrink-0">
              {activeAlerts.length > 0 && (
                <div className="flex items-center gap-1.5 text-xs text-violet-300 bg-violet-500/10 border border-violet-500/20 px-3 py-1.5 rounded-lg">
                  <Bell size={12} /> {activeAlerts.length} active alert{activeAlerts.length > 1 ? "s" : ""}
                </div>
              )}
              {triggeredAlerts.length > 0 && (
                <div className="flex items-center gap-1.5 text-xs text-yellow-300 bg-yellow-500/10 border border-yellow-500/20 px-3 py-1.5 rounded-lg animate-pulse">
                  <AlertTriangle size={12} /> {triggeredAlerts.length} triggered
                </div>
              )}
              <button onClick={saveComparison} className="flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 transition-all">
                <Bookmark size={12} /> Save
              </button>
              <button onClick={() => setShowShareModal(true)} className="flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 transition-all">
                <Share2 size={12} /> Share
              </button>
            </div>
          )}
        </div>

        {/* ── SAVED BANNER ── */}
        <AnimatePresence>
          {savedBanner && (
            <motion.div initial={{ opacity: 0, y: -8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -8 }}
              className="mb-4 p-3 bg-emerald-500/10 border border-emerald-500/20 rounded-xl text-emerald-300 text-sm flex items-center gap-2">
              <CheckCircle2 size={15} /> Comparison saved to your browser!
            </motion.div>
          )}
        </AnimatePresence>

        {/* ── STOCK CHIPS + SEARCH ── */}
        <div className="bg-[#0d0d1a] border border-white/5 rounded-2xl p-4 mb-6 space-y-3">
          {/* Chips */}
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-xs text-gray-600 font-medium shrink-0">Comparing:</span>
            <AnimatePresence>
              {activeSymbols.length === 0 && (
                <motion.span initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="text-xs text-gray-700 italic">
                  No stocks selected yet
                </motion.span>
              )}
              {activeSymbols.map((sym, i) => (
                <StockChip
                  key={sym}
                  symbol={sym}
                  index={i}
                  onRemove={() => removeSymbol(sym)}
                  onAlertClick={() => setAlertModal(sym)}
                  isLoading={stockQueries[i]?.isLoading ?? false}
                  data={stockQueries[i]?.data}
                />
              ))}
            </AnimatePresence>
            {activeSymbols.length < 4 && (
              <div className="text-[10px] text-gray-700 ml-auto">
                {4 - activeSymbols.length} slot{4 - activeSymbols.length !== 1 ? "s" : ""} remaining
              </div>
            )}
          </div>

          {/* Search */}
          <SearchAddBar activeSymbols={activeSymbols} onAdd={addSymbol} stockQueries={stockQueries} />
          <p className="text-[10px] text-gray-700">Add up to 4 stocks • Search by ticker or company name • Press Enter to add first result</p>
        </div>

        {/* ── TRIGGERED ALERT BANNER ── */}
        <AnimatePresence>
          {triggeredAlerts.map((a) => (
            <motion.div key={a.id} initial={{ opacity: 0, y: -8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
              className="mb-3 p-3 bg-yellow-500/10 border border-yellow-500/20 rounded-xl text-yellow-300 text-sm flex items-center justify-between">
              <div className="flex items-center gap-2">
                <AlertTriangle size={14} />
                <strong>{a.symbol}</strong> — {a.type.replace("_", " ")} {a.value} was triggered {timeAgo(a.triggeredAt)}
              </div>
              <button onClick={() => {
                const next = alerts.map((al) => al.id === a.id ? { ...al, active: false } : al);
                setAlerts(next);
                localStorage.setItem("stock-alerts", JSON.stringify(next));
              }} className="text-[10px] text-yellow-500 hover:text-yellow-300 underline">Dismiss</button>
            </motion.div>
          ))}
        </AnimatePresence>

        {/* ── EMPTY STATE ── */}
        {activeSymbols.length === 0 && (
          <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}
            className="text-center py-24 border border-white/5 border-dashed rounded-2xl bg-white/[0.01]">
            <div className="w-14 h-14 rounded-2xl bg-violet-500/10 flex items-center justify-center mx-auto mb-4">
              <BarChart2 className="text-violet-400" size={28} />
            </div>
            <h3 className="text-lg font-semibold mb-2">Start your comparison</h3>
            <p className="text-sm text-gray-500 max-w-xs mx-auto">Search for up to 4 stocks above to see side-by-side AI analysis, charts, and insights.</p>

            {savedComparisons.length > 0 && (
              <div className="mt-8">
                <p className="text-xs text-gray-600 mb-3">Or load a saved comparison:</p>
                <div className="flex flex-wrap gap-2 justify-center">
                  {savedComparisons.slice(0, 5).map((c) => (
                    <button key={c.id} onClick={() => setActiveSymbols(c.symbols)}
                      className="text-xs px-3 py-1.5 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 text-gray-300 transition-all">
                      {c.name}
                    </button>
                  ))}
                </div>
              </div>
            )}
          </motion.div>
        )}

        {/* ── ONE STOCK HINT ── */}
        {activeSymbols.length === 1 && (
          <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}
            className="mb-6 p-4 bg-blue-500/5 border border-blue-500/15 rounded-xl flex items-center gap-3">
            <Eye size={16} className="text-blue-400 shrink-0" />
            <div>
              <p className="text-sm text-blue-200 font-medium">Add a second stock to unlock the comparison</p>
              <p className="text-xs text-gray-500 mt-0.5">Charts, AI synthesis, and metric battles all activate when you have 2+ assets.</p>
            </div>
          </motion.div>
        )}

        {/* ── MAIN DASHBOARD ── */}
        {activeSymbols.length > 0 && (
          <div className="space-y-6">

            {/* CHART */}
            {fetchedData.length > 0 && (
              <div className="bg-[#0d0d1a] border border-white/5 rounded-2xl overflow-hidden">
                <div className="px-5 py-4 border-b border-white/5 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                  <div className="flex items-center gap-2">
                    <Activity size={15} className="text-violet-400" />
                    <h3 className="text-sm font-semibold">Price Performance</h3>
                    {activeSymbols.length < 2 && (
                      <span className="text-[10px] text-gray-600 bg-white/[0.02] border border-white/5 px-2 py-0.5 rounded-md">Add more stocks to compare</span>
                    )}
                  </div>
                  <div className="flex items-center gap-3 flex-wrap">
                    <button
                      onClick={() => setNormalizeChart(!normalizeChart)}
                      className={`text-[11px] font-bold px-3 py-1.5 rounded-lg border transition-all ${normalizeChart ? "bg-violet-500/15 border-violet-500/40 text-violet-300" : "bg-white/[0.03] border-white/10 text-gray-500 hover:text-white"}`}
                    >
                      {normalizeChart ? "% Relative" : "$ Absolute"}
                    </button>
                    <div className="flex gap-1 bg-white/[0.03] p-1 rounded-lg border border-white/5">
                      {CHART_RANGES.map((r) => (
                        <button
                          key={r}
                          onClick={() => setChartRange(r)}
                          className={`px-3 py-1 rounded-md text-[11px] font-bold transition-all ${chartRange === r ? "bg-white/10 text-white" : "text-gray-600 hover:text-gray-400"}`}
                        >
                          {r}
                        </button>
                      ))}
                    </div>
                  </div>
                </div>
                <div className="p-5 h-[360px] flex flex-col">
                  <ComparisonChart stocks={fetchedData} range={chartRange} normalize={normalizeChart} />
                </div>
              </div>
            )}

            {/* AI SUMMARY */}
            <AISummary stocks={fetchedData} />

            {/* METRICS TABLE */}
            {fetchedData.length > 0 && (
              <MetricsTable
                stocks={fetchedData}
                queries={stockQueries.map((q) => ({ isLoading: q.isLoading, data: q.data }))}
              />
            )}

            {/* NEWS */}
            {activeSymbols.length > 0 && (
              <NewsPanel newsQueries={newsQueries} symbols={activeSymbols} />
            )}
          </div>
        )}
      </div>

      {/* ── MODALS ── */}
      <AnimatePresence>
        {alertModal && (
          <AlertModal
            symbol={alertModal}
            onClose={() => setAlertModal(null)}
            alerts={alerts}
            onSave={saveAlert}
          />
        )}
        {showShareModal && (
          <ShareModal symbols={activeSymbols} onClose={() => setShowShareModal(false)} />
        )}
      </AnimatePresence>
    </main>
  );
}