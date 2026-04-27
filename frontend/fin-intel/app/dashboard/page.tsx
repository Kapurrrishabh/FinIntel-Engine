"use client";

// ─────────────────────────────────────────────────────────────────────────────
//  FinIntel · Complete Dashboard
// ─────────────────────────────────────────────────────────────────────────────
import { Suspense } from "react";

import { useRouter, useSearchParams } from "next/navigation";
import { useState, useEffect, useCallback, useRef, useMemo } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import toast from "react-hot-toast"; // Added react-hot-toast
import {
  Search, X, Star, Bell, BarChart2, TrendingUp, TrendingDown,
  RefreshCw, ExternalLink, Command, Plus, Trash2, AlertCircle,
  ChevronDown, ChevronUp, ArrowLeft, ArrowRight, ArrowUp, ArrowDown,
  Globe, Clock, Shield, Zap, Activity, Filter, GitCompare, Briefcase,
  BellOff, CheckCircle2, TrendingUp as TUp, Info, Eye, Minus,
  ShieldCheck, ShieldAlert, Building2, Newspaper, DollarSign,
} from "lucide-react";
import {
  Chart as ChartJS, LineElement, PointElement, LinearScale,
  CategoryScale, Tooltip, Filler,
} from "chart.js";
import { Line } from "react-chartjs-2";

ChartJS.register(LineElement, PointElement, LinearScale, CategoryScale, Tooltip, Filler);

// ─────────────────────────────────────────────────────────────────────────────
// CONSTANTS & CONFIG
// ─────────────────────────────────────────────────────────────────────────────

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

type View = "dashboard" | "portfolio" | "alerts" | "compare";
type Period = "1W" | "1M" | "3M" | "6M" | "1Y" | "ALL";

const PERIOD_DAYS: Record<Period, number> = {
  "1W": 7, "1M": 30, "3M": 90, "6M": 180, "1Y": 365, "ALL": Infinity,
};
const PERIODS: Period[] = ["1W", "1M", "3M", "6M", "1Y", "ALL"];
const COMPARE_COLORS = ["#3b82f6", "#34d399", "#f59e0b", "#a78bfa", "#fb7185"];
const POPULAR = ["AAPL", "NVDA", "MSFT", "TSLA", "RELIANCE.NS", "BTC-USD", "TCS.NS", "GOOGL"];

// ─────────────────────────────────────────────────────────────────────────────
// TYPES
// ─────────────────────────────────────────────────────────────────────────────

interface SearchResult   { symbol: string; name?: string; exchange?: string; type?: string; }
interface NewsItem       { id?: string | number; title: string; text?: string; url?: string; link?: string; source?: string; time?: string; sentiment?: string; thumbnail?: string; }
interface PricePoint     { date: string; close: number; }
interface Holding        { symbol: string; quantity: number; price: number; current_price: number; invested: number; current_value: number; pnl: number; pnl_percent: number; is_profit: boolean; }
interface PortfolioData  { holdings: Holding[]; summary: { total_invested: number; total_current_value: number; total_pnl: number; total_pnl_percent: number; count: number; }; }
interface AlertItem      { id: number; symbol: string; target_price: number; condition: "above" | "below"; triggered: boolean; triggered_at?: string; triggered_price?: number; created_at: string; }
interface CompareStock   { symbol: string; company_name?: string; sector?: string; prices: PricePoint[]; market_cap?: number; pe_ratio?: number; eps?: number; beta?: number; dividend_yield?: number; change_1m?: number; current_price?: number; currency?: string; error?: string; }
interface StockData      {
  symbol: string; company_name?: string; sector?: string; industry?: string;
  description?: string; website?: string; currency?: string;
  prices: PricePoint[];
  final_score: number; sentiment_score: number; risk_score: number;
  technical_score: number; fundamental_score: number;
  fundamentals: {
    roe?: number | null; debt_equity?: number | null; revenue_growth?: number | null;
    profit_margin?: number | null; market_cap?: number | null; pe_ratio?: number | null;
    eps?: number | null; "52w_high"?: number | null; "52w_low"?: number | null;
    avg_volume?: number | null; dividend_yield?: number | null; beta?: number | null;
  };
  explanation: string; decision?: string; cached?: boolean; analyzed_at?: string;
}

// ─────────────────────────────────────────────────────────────────────────────
// UTILITY FUNCTIONS
// ─────────────────────────────────────────────────────────────────────────────

// Intelligently determine currency based on symbol or API response
function getCurrencySymbol(symbol: string, currencyCode?: string): string {
  if (currencyCode === "INR" || symbol.toUpperCase().endsWith(".NS") || symbol.toUpperCase().endsWith(".BO")) {
    return "₹";
  }
  if (currencyCode === "EUR") return "€";
  if (currencyCode === "GBP") return "£";
  return "$";
}

// Strict formatter to catch NaNs and empty values
function fmtNum(n: number | null | undefined, type = "number", prefix = "", decimals = 2): string {
  if (n == null || isNaN(n) || !isFinite(n)) return "N/A";
  if (type === "large") {
    if (Math.abs(n) >= 1e12) return `${prefix}${(n / 1e12).toFixed(2)}T`;
    if (Math.abs(n) >= 1e9)  return `${prefix}${(n / 1e9).toFixed(2)}B`;
    if (Math.abs(n) >= 1e6)  return `${prefix}${(n / 1e6).toFixed(2)}M`;
    return `${prefix}${n.toLocaleString()}`;
  }
  if (type === "pct") return `${n > 0 ? "+" : ""}${n.toFixed(decimals)}%`;
  return `${prefix}${n.toFixed(decimals)}`;
}

function useDebounce<T>(val: T, ms: number): T {
  const [dv, setDv] = useState(val);
  useEffect(() => { const t = setTimeout(() => setDv(val), ms); return () => clearTimeout(t); }, [val, ms]);
  return dv;
}

function useLocalStorage<T>(key: string, init: T): [T, (v: T) => void] {
  const [val, setVal] = useState<T>(init);
  useEffect(() => { try { const s = localStorage.getItem(key); if (s) setVal(JSON.parse(s)); } catch {} }, [key]);
  const set = useCallback((v: T) => { setVal(v); try { localStorage.setItem(key, JSON.stringify(v)); } catch {} }, [key]);
  return [val, set];
}

// ─────────────────────────────────────────────────────────────────────────────
// SHARED MICRO COMPONENTS
// ─────────────────────────────────────────────────────────────────────────────

const Skeleton = ({ className }: { className?: string }) => (
  <div className={`relative overflow-hidden bg-white/[0.04] rounded-xl ${className}`}>
    <div className="absolute inset-0 -translate-x-full bg-gradient-to-r from-transparent via-white/[0.05] to-transparent animate-[shimmer_1.6s_infinite]" />
  </div>
);

const Tag = ({ children, color = "blue" }: { children: React.ReactNode; color?: string }) => {
  const cls: Record<string, string> = {
    blue:   "bg-blue-500/10 text-blue-400 border-blue-500/20",
    green:  "bg-emerald-500/10 text-emerald-400 border-emerald-500/20",
    red:    "bg-rose-500/10 text-rose-400 border-rose-500/20",
    amber:  "bg-amber-500/10 text-amber-400 border-amber-500/20",
    purple: "bg-purple-500/10 text-purple-400 border-purple-500/20",
    gray:   "bg-white/[0.04] text-gray-500 border-white/[0.08]",
  };
  return <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full border uppercase tracking-wider ${cls[color] ?? cls.gray}`}>{children}</span>;
};

const SentimentTag = ({ s }: { s?: string }) => {
  if (!s) return null;
  const l = s.toLowerCase();
  const color = l.includes("bull") || l.includes("pos") ? "green" : l.includes("bear") || l.includes("neg") ? "red" : "amber";
  const label = color === "green" ? "Bullish" : color === "red" ? "Bearish" : "Neutral";
  return <Tag color={color}>{label}</Tag>;
};

const CompanyLogo = ({ symbol, size = 9 }: { symbol: string; size?: number }) => {
  const [err, setErr] = useState(false);
  const clean = symbol.split(".")[0].split("-")[0].toLowerCase();
  const dim = `w-${size} h-${size}`;
  if (err) return (
    <div className={`${dim} rounded-full bg-gradient-to-br from-blue-500/20 to-purple-500/20 border border-white/10 flex items-center justify-center text-[10px] font-black text-blue-400 shrink-0`}>
      {symbol.slice(0, 2)}
    </div>
  );
  return <img src={`https://logo.clearbit.com/${clean}.com`} alt={symbol} onError={() => setErr(true)} className={`${dim} rounded-full object-cover shrink-0 border border-white/[0.06] bg-white/5`} />;
};

const ProgressBar = ({ value, color = "bg-blue-500", height = "h-1.5" }: { value: number; color?: string; height?: string }) => (
  <div className={`w-full bg-black/40 ${height} rounded-full overflow-hidden`}>
    <motion.div initial={{ width: 0 }} animate={{ width: `${Math.max(0, Math.min(100, (value || 0) * 100))}%` }}
      transition={{ duration: 0.9, ease: "easeOut" }} className={`${height} rounded-full ${color}`} />
  </div>
);

// ─────────────────────────────────────────────────────────────────────────────
// SCORE CARD  (Recommendation / Sentiment / Risk)
// ─────────────────────────────────────────────────────────────────────────────

function ScoreCard({ label, value, score, icon, barColor, bg, color, left, mid, right, sub }:
  { label: string; value: string; score: number; icon: React.ReactNode; barColor: string; bg: string; color: string; left: string; mid: string; right: string; sub?: string; }) {
  return (
    <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}
      className={`p-5 rounded-2xl border border-white/[0.07] ${bg} relative overflow-hidden`}>
      <div className={`absolute -top-8 -right-8 w-24 h-24 rounded-full blur-2xl opacity-20 ${barColor}`} />
      <p className="text-[10px] text-gray-500 uppercase font-bold tracking-widest mb-3">{label}</p>
      <div className="flex items-center justify-between mb-4">
        <div className={`flex items-center gap-2 ${color}`}>{icon}<span className="text-xl font-black">{value}</span></div>
        <div className="text-right">
          <p className="text-[10px] text-gray-600">Score</p>
          <p className="text-lg font-black text-white">{isNaN(score) ? "0" : (score * 100).toFixed(0)}%</p>
        </div>
      </div>
      {sub && <p className="text-[10px] text-gray-600 mb-3">{sub}</p>}
      <ProgressBar value={score || 0} color={barColor} height="h-2" />
      <div className="flex justify-between mt-2 text-[9px] text-gray-700 font-semibold uppercase tracking-widest">
        <span>{left}</span><span>{mid}</span><span>{right}</span>
      </div>
    </motion.div>
  );
}

function RecommendationCard({ score }: { score: number }) {
  const s = Math.max(0, Math.min(1, score || 0));
  if (s > 0.68) return <ScoreCard label="AI Recommendation" value="Buy" score={s} icon={<TrendingUp size={20}/>} barColor="bg-emerald-500" bg="bg-emerald-500/[0.07]" color="text-emerald-400" left="Avoid" mid="Hold" right="Buy" sub={`${((s-0.68)/0.32*100).toFixed(0)}% above buy threshold`} />;
  if (s < 0.38) return <ScoreCard label="AI Recommendation" value="Avoid" score={s} icon={<TrendingDown size={20}/>} barColor="bg-rose-500" bg="bg-rose-500/[0.07]" color="text-rose-400" left="Avoid" mid="Hold" right="Buy" sub="Below buy threshold" />;
  return <ScoreCard label="AI Recommendation" value="Hold" score={s} icon={<Minus size={20}/>} barColor="bg-amber-500" bg="bg-amber-500/[0.07]" color="text-amber-400" left="Avoid" mid="Hold" right="Buy" sub="In neutral zone" />;
}

function SentimentCard({ score }: { score: number }) {
  const s = Math.max(0, Math.min(1, score || 0));
  if (s > 0.6) return <ScoreCard label="Market Sentiment" value="Positive" score={s} icon={<TrendingUp size={20}/>} barColor="bg-emerald-500" bg="bg-emerald-500/[0.07]" color="text-emerald-400" left="Negative" mid="Neutral" right="Positive" />;
  if (s < 0.4) return <ScoreCard label="Market Sentiment" value="Negative" score={s} icon={<TrendingDown size={20}/>} barColor="bg-rose-500" bg="bg-rose-500/[0.07]" color="text-rose-400" left="Negative" mid="Neutral" right="Positive" />;
  return <ScoreCard label="Market Sentiment" value="Neutral" score={s} icon={<Minus size={20}/>} barColor="bg-amber-500" bg="bg-amber-500/[0.07]" color="text-amber-400" left="Negative" mid="Neutral" right="Positive" />;
}

function RiskCard({ risk }: { risk: number }) {
  const s = Math.max(0, Math.min(1, risk || 0));
  if (s < 0.35) return <ScoreCard label="Risk Assessment" value="Low Risk" score={s} icon={<ShieldCheck size={20}/>} barColor="bg-emerald-500" bg="bg-emerald-500/[0.07]" color="text-emerald-400" left="Low" mid="Moderate" right="High" />;
  if (s > 0.65) return <ScoreCard label="Risk Assessment" value="High Risk" score={s} icon={<ShieldAlert size={20}/>} barColor="bg-rose-500" bg="bg-rose-500/[0.07]" color="text-rose-400" left="Low" mid="Moderate" right="High" />;
  return <ScoreCard label="Risk Assessment" value="Moderate" score={s} icon={<AlertCircle size={20}/>} barColor="bg-amber-500" bg="bg-amber-500/[0.07]" color="text-amber-400" left="Low" mid="Moderate" right="High" />;
}

// ─────────────────────────────────────────────────────────────────────────────
// PRICE CHART
// ─────────────────────────────────────────────────────────────────────────────

function PriceChart({ data, symbol, cs = "$" }: { data: PricePoint[]; symbol: string; cs?: string }) {
  const [period, setPeriod] = useState<Period>("3M");

  const filtered = useMemo(() => {
    if (!data?.length) return [];
    if (period === "ALL") return data;
    const cutoff = new Date();
    cutoff.setDate(cutoff.getDate() - PERIOD_DAYS[period]);
    return data.filter(d => new Date(d.date) >= cutoff);
  }, [data, period]);

  const { isUp, changePct } = useMemo(() => {
    if (filtered.length < 2) return { isUp: true, changePct: 0 };
    const first = filtered[0].close, last = filtered[filtered.length - 1].close;
    if (!first) return { isUp: true, changePct: 0 };
    return { isUp: last >= first, changePct: ((last - first) / first) * 100 };
  }, [filtered]);

  if (!data?.length) return <div className="bg-[#0d0d0d] border border-white/[0.06] rounded-2xl h-64 flex items-center justify-center text-gray-600 text-sm">No price data</div>;

  const lc = isUp ? "#34d399" : "#fb7185";
  const chartData = {
    labels: filtered.map(d => new Date(d.date).toLocaleDateString("en-US", { month: "short", day: "numeric" })),
    datasets: [{
      data: filtered.map(d => d.close),
      borderColor: lc,
      backgroundColor: (ctx: any) => {
        const g = ctx.chart.canvas.getContext("2d").createLinearGradient(0, 0, 0, ctx.chart.canvas.height);
        g.addColorStop(0, isUp ? "rgba(52,211,153,0.18)" : "rgba(251,113,133,0.18)");
        g.addColorStop(1, "transparent");
        return g;
      },
      fill: true, borderWidth: 2, tension: 0.35, pointRadius: 0,
      pointHoverRadius: 5, pointHoverBackgroundColor: lc, pointHoverBorderColor: "#fff", pointHoverBorderWidth: 2,
    }],
  };

  return (
    <div className="bg-[#0d0d0d] border border-white/[0.06] rounded-2xl p-5">
      <div className="flex items-center justify-between mb-4 flex-wrap gap-3">
        <div className="flex items-center gap-3">
          <div>
            <p className="text-[10px] text-gray-600 uppercase font-semibold tracking-wider">{symbol} Price</p>
            {filtered.length > 0 && (
              <p className="text-lg font-black">{cs}{filtered.at(-1)!.close.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</p>
            )}
          </div>
          <span className={`flex items-center gap-1 text-sm font-bold ${isUp ? "text-emerald-400" : "text-rose-400"}`}>
            {isUp ? <TrendingUp size={14}/> : <TrendingDown size={14}/>}
            {changePct >= 0 ? "+" : ""}{isNaN(changePct) ? "0.00" : changePct.toFixed(2)}%
            <span className="text-gray-600 text-xs font-normal">({period})</span>
          </span>
        </div>
        <div className="flex gap-1 bg-white/[0.03] border border-white/[0.05] rounded-xl p-1">
          {PERIODS.map(p => (
            <button key={p} onClick={() => setPeriod(p)}
              className={`px-2.5 py-1.5 rounded-lg text-[11px] font-bold transition-all ${period === p ? "bg-blue-500/20 text-blue-400" : "text-gray-600 hover:text-gray-300"}`}>
              {p}
            </button>
          ))}
        </div>
      </div>
      <div style={{ height: 240 }}>
        <Line data={chartData} options={{
          responsive: true, maintainAspectRatio: false,
          animation: { duration: 400 },
          plugins: {
            legend: { display: false },
            tooltip: {
              backgroundColor: "#111", borderColor: "rgba(255,255,255,0.08)", borderWidth: 1,
              padding: 10, titleColor: "#9ca3af", bodyColor: "#fff", displayColors: false,
              callbacks: {
                title: (i: any[]) => i[0]?.label,
                label: (i: any) => `${cs}${i.parsed.y >= 1000 ? i.parsed.y.toLocaleString("en-US", { minimumFractionDigits: 2 }) : i.parsed.y.toFixed(2)}`,
              },
            },
          },
          scales: {
            x: { grid: { display: false }, border: { display: false }, ticks: { color: "#4b5563", maxTicksLimit: 6, font: { size: 10 }, maxRotation: 0 } },
            y: { grid: { color: "rgba(255,255,255,0.03)" }, border: { display: false }, position: "right",
              ticks: { color: "#4b5563", font: { size: 10 }, callback: (v: any) => v >= 1000 ? `${cs}${(v/1000).toFixed(1)}k` : `${cs}${v}` } },
          },
          interaction: { mode: "index", intersect: false },
        }} />
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// 52-WEEK RANGE BAR
// ─────────────────────────────────────────────────────────────────────────────

function RangeBar({ low, high, current, cs = "$" }: { low?: number | null; high?: number | null; current?: number; cs?: string }) {
  if (low == null || high == null || current == null || isNaN(low) || isNaN(high)) return null;
  const pct = Math.max(0, Math.min(100, ((current - low) / (high - low)) * 100));
  return (
    <div className="bg-[#0d0d0d] border border-white/[0.06] rounded-2xl p-4">
      <p className="text-[10px] text-gray-500 uppercase font-semibold tracking-wider mb-3">52-Week Range</p>
      <div className="relative h-2 bg-white/[0.06] rounded-full mb-3">
        <div className="absolute inset-y-0 left-0 bg-gradient-to-r from-rose-500 via-amber-500 to-emerald-500 rounded-full" style={{ width: "100%" }} />
        <div className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-4 h-4 bg-white rounded-full border-2 border-blue-500 shadow-lg shadow-blue-500/30 z-10" style={{ left: `${isNaN(pct) ? 50 : pct}%` }} />
      </div>
      <div className="flex justify-between text-[11px] font-semibold">
        <span className="text-rose-400">{cs}{low.toFixed(2)} <span className="text-gray-600 font-normal text-[10px]">Low</span></span>
        <span className="text-white font-black">{cs}{current.toFixed(2)}</span>
        <span className="text-emerald-400"><span className="text-gray-600 font-normal text-[10px]">High</span> {cs}{high.toFixed(2)}</span>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// FUNDAMENTALS GRID
// ─────────────────────────────────────────────────────────────────────────────

function FundamentalsSection({ data, cs = "$" }: { data: StockData["fundamentals"]; cs?: string }) {
  const getC = (v: number, type: string) => {
    if (isNaN(v)) return "text-gray-300";
    if (type === "debt") return v < 1 ? "text-emerald-400" : v < 2 ? "text-amber-400" : "text-rose-400";
    return v > 15 ? "text-emerald-400" : v > 5 ? "text-amber-400" : v > 0 ? "text-gray-300" : "text-rose-400";
  };

  const metrics = [
    { label: "ROE",            val: data.roe,            suf: "%",  type: "roe",    tip: "Return on Equity — profitability per shareholder dollar." },
    { label: "Debt/Equity",    val: data.debt_equity,    suf: "x",  type: "debt",   tip: "Leverage ratio. Under 1 is generally healthy." },
    { label: "Rev. Growth",    val: data.revenue_growth, suf: "%",  type: "growth", tip: "Year-over-year revenue growth rate." },
    { label: "Profit Margin",  val: data.profit_margin,  suf: "%",  type: "margin", tip: "Net income as % of revenue." },
    { label: "P/E Ratio",      val: data.pe_ratio,       suf: "x",  type: "pe",     tip: "Price-to-Earnings. Lower may signal undervaluation." },
    { label: "EPS",            val: data.eps,            pre: cs,   type: "eps",    tip: "Earnings Per Share (trailing 12 months)." },
    { label: "Beta",           val: data.beta,           suf: "",   type: "beta",   tip: ">1 = more volatile than market." },
    { label: "Div. Yield",     val: data.dividend_yield != null ? (data.dividend_yield * 100) : null, suf: "%", type: "div", tip: "Annual dividend as % of current price." },
  ];

  const hasAny = metrics.some(m => m.val != null);
  if (!hasAny) return null;

  return (
    <div className="bg-[#0d0d0d] border border-white/[0.06] rounded-2xl p-5">
      <p className="text-[10px] text-gray-500 uppercase font-bold tracking-widest mb-4">Fundamentals</p>
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        {metrics.map((m, i) => (
          <motion.div key={m.label} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.04 }}
            title={m.tip}
            className="group p-3.5 rounded-xl bg-white/[0.02] border border-white/[0.05] hover:bg-white/[0.05] hover:border-white/[0.09] transition-all cursor-help">
            <p className="text-[10px] text-gray-600 font-semibold uppercase tracking-wider mb-1.5 truncate">{m.label}</p>
            <p className={`text-base font-black ${m.val != null && !isNaN(m.val) ? getC(m.val, m.type) : "text-gray-700"}`}>
              {m.val != null && !isNaN(m.val) ? `${m.pre ?? ""}${m.val.toFixed(2)}${m.suf ?? ""}` : "N/A"}
            </p>
            <p className="text-[9px] text-gray-700 mt-1 hidden group-hover:block leading-snug">{m.tip}</p>
          </motion.div>
        ))}
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// NEWS PANEL
// ─────────────────────────────────────────────────────────────────────────────

function NewsPanel({ news, sentimentScore }: { news: NewsItem[]; sentimentScore?: number }) {
  const [expanded, setExpanded] = useState(false);
  const shown = expanded ? news : news.slice(0, 4);

  return (
    <div className="bg-[#0d0d0d] border border-white/[0.06] rounded-2xl p-5">
      <div className="flex items-center justify-between mb-4">
        <p className="text-[10px] text-gray-500 uppercase font-bold tracking-widest">Related News</p>
        {sentimentScore != null && !isNaN(sentimentScore) && (
          <div className="flex items-center gap-2">
            <span className={`text-xs font-bold ${sentimentScore > 0.6 ? "text-emerald-400" : sentimentScore < 0.4 ? "text-rose-400" : "text-amber-400"}`}>
              {(sentimentScore * 100).toFixed(0)}% {sentimentScore > 0.6 ? "Positive" : sentimentScore < 0.4 ? "Negative" : "Neutral"}
            </span>
            <div className="w-16 h-1.5 bg-white/[0.06] rounded-full overflow-hidden">
              <div className={`h-full rounded-full ${sentimentScore > 0.6 ? "bg-emerald-500" : sentimentScore < 0.4 ? "bg-rose-500" : "bg-amber-500"}`}
                style={{ width: `${sentimentScore * 100}%` }} />
            </div>
          </div>
        )}
      </div>

      {!news.length ? (
        <p className="text-gray-700 text-xs text-center py-6">No recent headlines found.</p>
      ) : (
        <div className="space-y-2.5">
          {shown.map((n, i) => {
            const href = n.url?.trim() || n.link?.trim() || `https://www.google.com/search?tbm=nws&q=${encodeURIComponent(n.title)}`;
            return (
              <a key={n.id ?? i} href={href} target="_blank" rel="noopener noreferrer"
                className="group flex gap-3 p-3 rounded-xl bg-white/[0.02] border border-white/[0.04] hover:bg-white/[0.05] hover:border-white/[0.08] transition-all">
                {n.thumbnail && (
                  <img src={n.thumbnail} alt="" onError={e => (e.currentTarget.style.display = "none")}
                    className="w-12 h-12 rounded-lg object-cover shrink-0 border border-white/[0.06]" />
                )}
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-1.5 mb-1 flex-wrap">
                    {n.source && <span className="text-[10px] font-bold text-blue-400/80">{n.source}</span>}
                    <SentimentTag s={n.sentiment} />
                    {n.time && <span className="text-[10px] text-gray-700 ml-auto flex items-center gap-0.5"><Clock size={9}/>{n.time}</span>}
                  </div>
                  {/* Prioritize n.title, but if it is generic, fallback or append the description (text) so it's readable */}
                  <p className="text-xs font-semibold text-gray-300 group-hover:text-white transition-colors line-clamp-2 leading-snug">
                    {n.title || n.text || "Market Update"}
                  </p>
                  {n.text && n.text !== n.title && <p className="text-[10px] text-gray-600 line-clamp-1 mt-0.5">{n.text}</p>}
                </div>
                <ExternalLink size={11} className="text-gray-700 group-hover:text-blue-400 transition-colors mt-1 shrink-0" />
              </a>
            );
          })}
        </div>
      )}

      {news.length > 4 && (
        <button onClick={() => setExpanded(p => !p)}
          className="w-full mt-3 py-2.5 rounded-xl bg-white/[0.03] border border-white/[0.05] text-[11px] font-bold text-gray-500 hover:text-white hover:bg-white/[0.06] transition-all flex items-center justify-center gap-1.5">
          <ChevronDown size={13} className={`transition-transform ${expanded ? "rotate-180" : ""}`} />
          {expanded ? "Show less" : `${news.length - 4} more articles`}
        </button>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// STOCK ANALYSIS SECTION
// ─────────────────────────────────────────────────────────────────────────────

function StockAnalysis({ symbol }: { symbol: string }) {
  const [descExpanded, setDescExpanded] = useState(false);
  const router = useRouter(); // For routing to portfolio page

  const { data: stock, isLoading, isError, refetch, isFetching } = useQuery<StockData>({
    queryKey: ["stock", symbol],
    queryFn: async () => {
      const r = await fetch(`${API}/stock/${symbol}`);
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      return r.json();
    },
    staleTime: 5 * 60 * 1000, retry: 2,
  });

  const { data: newsRaw } = useQuery({
    queryKey: ["news", symbol],
    queryFn: async () => {
      const r = await fetch(`${API}/news/${symbol}`);
      const j = await r.json();
      return j.news || [];
    },
    staleTime: 10 * 60 * 1000,
  });

  const qc = useQueryClient();
  const addToPortfolio = async () => {
    const price = stock?.prices?.at(-1)?.close;
    if (!price || isNaN(price)) { toast.error("No price data available"); return; }
    
    // Create loading toast
    const loadingToast = toast.loading("Adding to portfolio...");
    try {
      const r = await fetch(`${API}/portfolio/add`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ symbol, quantity: 1, price }),
      });
      const d = await r.json();
      if (!r.ok) throw new Error(d.detail || d.message || "Failed to add position");
      
      qc.invalidateQueries({ queryKey: ["portfolio"] });
      toast.success(`${symbol} added to portfolio!`, { id: loadingToast });
      
      // Redirect to portfolio route
      router.push("/portfolio");
      
    } catch (e: any) { 
      toast.error(e.message || "Failed to add to portfolio", { id: loadingToast }); 
    }
  };

  if (isLoading) return (
    <div className="space-y-4">
      <Skeleton className="h-28" />
      <div className="grid grid-cols-3 gap-4">{[0,1,2].map(i => <Skeleton key={i} className="h-36" />)}</div>
      <Skeleton className="h-64" />
      <p className="text-center text-xs text-gray-600 animate-pulse">Analyzing {symbol} — running AI models…</p>
    </div>
  );

  if (isError || !stock) return (
    <div className="p-5 bg-rose-500/[0.07] border border-rose-500/20 rounded-2xl flex items-start gap-3">
      <AlertCircle size={17} className="text-rose-400 shrink-0 mt-0.5" />
      <div>
        <p className="font-bold text-rose-400 text-sm">Analysis failed for {symbol}</p>
        <p className="text-xs text-gray-500 mt-1">Symbol may be invalid or data provider is unavailable.</p>
        <button onClick={() => refetch()} className="mt-2 text-xs text-blue-400 flex items-center gap-1 hover:text-blue-300">
          <RefreshCw size={11}/> Try again
        </button>
      </div>
    </div>
  );

  const currentPrice = stock.prices?.at(-1)?.close;
  const prevPrice    = stock.prices?.at(-2)?.close;
  const dayChange    = currentPrice && prevPrice ? ((currentPrice - prevPrice) / prevPrice) * 100 : null;
  const cs           = getCurrencySymbol(symbol, stock.currency);
  const news: NewsItem[] = newsRaw || [];

  const scoreItems = [
    { label: "Technical",  value: stock.technical_score || 0,  color: "bg-blue-500" },
    { label: "Fundamental",  value: stock.fundamental_score || 0, color: "bg-purple-500" },
    { label: "Sentiment",  value: stock.sentiment_score || 0,  color: "bg-emerald-500" },
    { label: "Overall",      value: stock.final_score || 0,      color: "bg-amber-500" },
  ];

  return (
    <div className="space-y-4">
      {/* Header card */}
      <div className="bg-[#0d0d0d] border border-white/[0.06] rounded-2xl p-5">
        <div className="flex items-start justify-between gap-4 flex-wrap">
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2.5 mb-1 flex-wrap">
              <CompanyLogo symbol={symbol} size={10} />
              <h1 className="text-2xl font-black tracking-tight">{symbol}</h1>
              {stock.cached && <Tag color="blue">⚡ Cached</Tag>}
              {isFetching && <RefreshCw size={12} className="text-blue-400 animate-spin" />}
            </div>
            {stock.company_name && <p className="text-gray-400 text-sm font-medium mb-1.5">{stock.company_name}</p>}
            <div className="flex items-center gap-1.5 flex-wrap">
              {stock.sector   && <Tag color="gray">{stock.sector}</Tag>}
              {stock.industry && <Tag color="gray">{stock.industry}</Tag>}
              {stock.website  && <a href={stock.website} target="_blank" rel="noopener noreferrer" className="text-[10px] text-blue-400/70 hover:text-blue-400 flex items-center gap-1 transition-colors"><Globe size={10}/> Website</a>}
            </div>
          </div>
          {currentPrice && !isNaN(currentPrice) && (
            <div className="text-right shrink-0">
              <p className="text-3xl font-black tracking-tight">{cs}{currentPrice.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</p>
              {dayChange != null && !isNaN(dayChange) && (
                <p className={`text-sm font-bold flex items-center justify-end gap-1 mt-0.5 ${dayChange >= 0 ? "text-emerald-400" : "text-rose-400"}`}>
                  {dayChange >= 0 ? <TrendingUp size={13}/> : <TrendingDown size={13}/>}
                  {dayChange >= 0 ? "+" : ""}{dayChange.toFixed(2)}% today
                </p>
              )}
            </div>
          )}
        </div>

        {stock.description && (
          <div className="mt-4 pt-4 border-t border-white/[0.05]">
            <p className={`text-xs text-gray-500 leading-relaxed ${descExpanded ? "" : "line-clamp-2"}`}>{stock.description}</p>
            <button onClick={() => setDescExpanded(p => !p)} className="mt-1.5 text-[10px] text-blue-400/60 hover:text-blue-400 flex items-center gap-1 transition-colors">
              {descExpanded ? <><ChevronUp size={10}/> Less</> : <><ChevronDown size={10}/> More</>}
            </button>
          </div>
        )}
      </div>

      {/* Quick stats strip */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5">
        {[
          { label: "Market Cap",  val: stock.fundamentals.market_cap  ? fmtNum(stock.fundamentals.market_cap, "large", cs)  : "N/A" },
          { label: "P/E Ratio",   val: stock.fundamentals.pe_ratio    ? `${stock.fundamentals.pe_ratio.toFixed(1)}x`          : "N/A" },
          { label: "EPS",         val: stock.fundamentals.eps         ? `${cs}${stock.fundamentals.eps.toFixed(2)}`            : "N/A" },
          { label: "Beta",        val: stock.fundamentals.beta        ? stock.fundamentals.beta.toFixed(2)                    : "N/A",
            cls: stock.fundamentals.beta ? (stock.fundamentals.beta > 1.5 ? "text-rose-400" : stock.fundamentals.beta < 0.8 ? "text-emerald-400" : "text-amber-400") : "" },
          { label: "Div. Yield",  val: stock.fundamentals.dividend_yield ? fmtNum(stock.fundamentals.dividend_yield * 100, "pct") : "N/A" },
          { label: "Avg Volume",  val: stock.fundamentals.avg_volume  ? fmtNum(stock.fundamentals.avg_volume, "large")       : "N/A" },
          { label: "Profit Margin", val: stock.fundamentals.profit_margin != null ? `${stock.fundamentals.profit_margin.toFixed(1)}%` : "N/A",
            cls: stock.fundamentals.profit_margin != null ? (stock.fundamentals.profit_margin > 15 ? "text-emerald-400" : stock.fundamentals.profit_margin > 0 ? "text-amber-400" : "text-rose-400") : "" },
          { label: "ROE",         val: stock.fundamentals.roe != null ? `${stock.fundamentals.roe.toFixed(1)}%` : "N/A",
            cls: stock.fundamentals.roe != null ? (stock.fundamentals.roe > 15 ? "text-emerald-400" : "text-amber-400") : "" },
        ].map(s => (
          <div key={s.label} className="bg-[#0d0d0d] border border-white/[0.05] rounded-xl p-3">
            <p className="text-[10px] text-gray-600 uppercase font-semibold tracking-wider mb-1">{s.label}</p>
            <p className={`text-sm font-black ${(s as any).cls || "text-white"}`}>{s.val}</p>
          </div>
        ))}
      </div>

      {/* Score cards */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <RecommendationCard score={stock.final_score} />
        <SentimentCard score={stock.sentiment_score} />
        <RiskCard risk={stock.risk_score} />
      </div>

      {/* 52W range */}
      <RangeBar low={stock.fundamentals["52w_low"]} high={stock.fundamentals["52w_high"]} current={currentPrice} cs={cs} />

      {/* Main 2-col layout */}
      <div className="grid lg:grid-cols-3 gap-4">
        <div className="lg:col-span-2 space-y-4">
          <PriceChart data={stock.prices} symbol={symbol} cs={cs} />
          <NewsPanel news={news} sentimentScore={stock.sentiment_score} />
        </div>

        <div className="space-y-4">
          <button onClick={addToPortfolio}
            className="w-full py-3.5 rounded-xl bg-gradient-to-r from-blue-600 to-blue-500 hover:opacity-90 active:scale-[0.98] font-black text-sm transition-all shadow-lg shadow-blue-500/20">
            + Add to Portfolio
          </button>

          {/* Score breakdown */}
          <div className="bg-[#0d0d0d] border border-white/[0.06] rounded-2xl p-5">
            <p className="text-[10px] text-gray-500 uppercase font-bold tracking-widest mb-4">Score Breakdown</p>
            <div className="space-y-4">
              {scoreItems.map(item => (
                <div key={item.label}>
                  <div className="flex justify-between text-xs mb-1.5">
                    <span className="text-gray-400 font-semibold">{item.label}</span>
                    <span className="text-white font-black">{isNaN(item.value) ? "0" : (item.value * 100).toFixed(0)}%</span>
                  </div>
                  <ProgressBar value={item.value || 0} color={item.color} height="h-1.5" />
                </div>
              ))}
            </div>
          </div>

          {/* AI Explanation */}
          <div className="bg-[#0d0d0d] border border-white/[0.06] rounded-2xl p-5">
            <p className="text-[10px] text-gray-500 uppercase font-bold tracking-widest mb-3">AI Analysis</p>
            <div className="space-y-2 text-xs text-gray-400 leading-relaxed">
              {stock.explanation
                ? stock.explanation.split("\n").filter(Boolean).map((l, i) => <p key={i}>{l}</p>)
                : <p className="text-gray-600 italic">No explanation available.</p>}
            </div>
            {stock.analyzed_at && (
              <p className="text-[10px] text-gray-700 mt-3">Analyzed {new Date(stock.analyzed_at).toLocaleString()}</p>
            )}
          </div>
        </div>
      </div>

      {/* Full fundamentals */}
      <FundamentalsSection data={stock.fundamentals} cs={cs} />
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// PORTFOLIO VIEW
// ─────────────────────────────────────────────────────────────────────────────

function PortfolioView({ onSelectSymbol }: { onSelectSymbol: (s: string) => void }) {
  const qc = useQueryClient();
  const [showAdd, setShowAdd] = useState(false);
  const [newSym, setNewSym]   = useState("");
  const [newQty, setNewQty]   = useState("");
  const [newPrc, setNewPrc]   = useState("");
  const [adding, setAdding]   = useState(false);

  const { data, isLoading, refetch } = useQuery<PortfolioData>({
    queryKey: ["portfolio"],
    queryFn: async () => { const r = await fetch(`${API}/portfolio`); if (!r.ok) throw new Error("Failed"); return r.json(); },
    refetchInterval: 60000,
  });

  const removeMut = useMutation({
    mutationFn: async (sym: string) => { const r = await fetch(`${API}/portfolio/remove/${sym}`, { method: "DELETE" }); if (!r.ok) throw new Error("Failed"); },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["portfolio"] });
      toast.success("Position removed");
    },
    onError: () => toast.error("Failed to remove position")
  });

  const handleAdd = async () => {
    if (!newSym.trim() || !newQty || !newPrc) { toast.error("All fields required"); return; }
    if (isNaN(+newQty) || isNaN(+newPrc)) { toast.error("Quantity and price must be valid numbers"); return; }
    
    setAdding(true);
    const loadingToast = toast.loading("Adding position...");
    try {
      const r = await fetch(`${API}/portfolio/add`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ symbol: newSym.trim().toUpperCase(), quantity: +newQty, price: +newPrc }),
      });
      if (!r.ok) { const d = await r.json(); throw new Error(d.detail || "Failed"); }
      
      qc.invalidateQueries({ queryKey: ["portfolio"] });
      setNewSym(""); setNewQty(""); setNewPrc(""); setShowAdd(false);
      toast.success("Position added!", { id: loadingToast });
    } catch (e: any) { 
      toast.error(e.message, { id: loadingToast }); 
    }
    setAdding(false);
  };

  const summary  = data?.summary;
  const holdings = data?.holdings || [];
  const isProfit = (summary?.total_pnl ?? 0) >= 0;

  return (
    <div className="space-y-5">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Briefcase size={17} className="text-blue-400" />
          <h2 className="font-black text-lg">Portfolio</h2>
          {summary && <span className="text-xs bg-white/[0.06] text-gray-500 px-2 py-0.5 rounded-full">{summary.count} holdings</span>}
        </div>
        <div className="flex items-center gap-2">
          <button onClick={() => refetch()} className="w-8 h-8 rounded-xl bg-white/[0.04] border border-white/[0.06] flex items-center justify-center text-gray-500 hover:text-white transition-all">
            <RefreshCw size={13} className={isLoading ? "animate-spin" : ""} />
          </button>
          <button onClick={() => setShowAdd(p => !p)}
            className="flex items-center gap-1.5 px-3.5 py-2 rounded-xl bg-blue-600 hover:bg-blue-500 font-bold text-xs transition-all">
            <Plus size={14}/> Add Position
          </button>
        </div>
      </div>

      {/* Add form inline */}
      <AnimatePresence>
        {showAdd && (
          <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: "auto", opacity: 1 }} exit={{ height: 0, opacity: 0 }}
            className="overflow-hidden">
            <div className="bg-[#0d0d0d] border border-blue-500/20 rounded-2xl p-5 space-y-4">
              <p className="text-xs font-bold text-gray-400 uppercase tracking-wider">New Position</p>
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                {[
                  { label: "Symbol", val: newSym, set: (v: string) => setNewSym(v.toUpperCase()), ph: "AAPL", mono: true },
                  { label: "Quantity", val: newQty, set: setNewQty, ph: "10", type: "number" },
                  { label: "Buy Price", val: newPrc, set: setNewPrc, ph: "150.00", type: "number" },
                ].map(f => (
                  <div key={f.label}>
                    <label className="text-[10px] text-gray-500 font-semibold uppercase tracking-wider block mb-1.5">{f.label}</label>
                    <input value={f.val} onChange={e => f.set(e.target.value)} type={f.type || "text"} placeholder={f.ph}
                      className={`w-full bg-white/[0.04] border border-white/[0.07] rounded-xl px-3.5 py-2.5 text-white text-sm outline-none focus:border-blue-500/50 placeholder-gray-700 transition-colors ${f.mono ? "font-mono" : ""}`} />
                  </div>
                ))}
              </div>
              <div className="flex gap-2">
                <button onClick={handleAdd} disabled={adding}
                  className="px-5 py-2.5 rounded-xl bg-blue-600 hover:bg-blue-500 disabled:opacity-50 font-bold text-sm transition-all">
                  {adding ? "Adding…" : "Add"}
                </button>
                <button onClick={() => { setShowAdd(false); }}
                  className="px-5 py-2.5 rounded-xl bg-white/[0.04] border border-white/[0.07] font-bold text-sm text-gray-400 hover:text-white transition-all">
                  Cancel
                </button>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {isLoading ? (
        <div className="flex justify-center py-16"><RefreshCw className="animate-spin text-blue-400" size={22}/></div>
      ) : holdings.length === 0 ? (
        <div className="py-20 text-center space-y-3">
          <Briefcase size={36} className="text-gray-700 mx-auto" />
          <p className="text-gray-500 text-sm">No positions yet. Add your first holding.</p>
          <button onClick={() => setShowAdd(true)} className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-blue-600 hover:bg-blue-500 font-bold text-sm transition-all">
            <Plus size={14}/> Add Position
          </button>
        </div>
      ) : (
        <>
          {/* Summary cards */}
          {summary && (
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              {[
                { label: "Total Invested",  val: `$${summary.total_invested.toLocaleString("en-US", { minimumFractionDigits: 2 })}`,      cls: "text-white" },
                { label: "Current Value",   val: `$${summary.total_current_value.toLocaleString("en-US", { minimumFractionDigits: 2 })}`, cls: "text-white" },
                { label: "Total P&L",       val: `${isProfit ? "+" : ""}$${Math.abs(summary.total_pnl).toLocaleString("en-US", { minimumFractionDigits: 2 })}`, cls: isProfit ? "text-emerald-400" : "text-rose-400" },
                { label: "Return",          val: `${summary.total_pnl_percent > 0 ? "+" : ""}${summary.total_pnl_percent.toFixed(2)}%`,  cls: isProfit ? "text-emerald-400" : "text-rose-400" },
              ].map((c, i) => (
                <motion.div key={c.label} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.04 }}
                  className="bg-[#0d0d0d] border border-white/[0.06] rounded-2xl p-4">
                  <p className="text-[10px] text-gray-600 uppercase font-semibold tracking-wider mb-1">{c.label}</p>
                  <p className={`text-xl font-black ${c.cls}`}>{c.val}</p>
                </motion.div>
              ))}
            </div>
          )}

          {/* Holdings table */}
          <div className="bg-[#0d0d0d] border border-white/[0.06] rounded-2xl overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-white/[0.05]">
                    {["Symbol", "Qty", "Avg Cost", "Current", "Invested", "Value", "P&L", "Return", ""].map(h => (
                      <th key={h} className="px-4 py-3 text-left text-[10px] text-gray-600 uppercase tracking-wider font-semibold whitespace-nowrap">{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {holdings.map((h, i) => {
                    const cS = getCurrencySymbol(h.symbol);
                    return (
                    <motion.tr key={h.symbol} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: i * 0.03 }}
                      className="border-b border-white/[0.03] hover:bg-white/[0.02] transition-colors group">
                      <td className="px-4 py-3.5">
                        <button onClick={() => onSelectSymbol(h.symbol)} className="font-black text-blue-400 hover:text-blue-300 transition-colors">{h.symbol}</button>
                      </td>
                      <td className="px-4 py-3.5 text-gray-400 text-xs">{h.quantity}</td>
                      <td className="px-4 py-3.5 text-gray-400 text-xs">{cS}{h.price.toFixed(2)}</td>
                      <td className="px-4 py-3.5 font-bold text-xs">{cS}{h.current_price.toFixed(2)}</td>
                      <td className="px-4 py-3.5 text-gray-500 text-xs">{cS}{h.invested.toFixed(2)}</td>
                      <td className="px-4 py-3.5 font-bold text-xs">{cS}{h.current_value.toFixed(2)}</td>
                      <td className={`px-4 py-3.5 font-black text-xs ${h.is_profit ? "text-emerald-400" : "text-rose-400"}`}>
                        {h.is_profit ? "+" : "-"}{cS}{Math.abs(h.pnl).toFixed(2)}
                      </td>
                      <td className={`px-4 py-3.5 text-xs font-black ${h.is_profit ? "text-emerald-400" : "text-rose-400"}`}>
                        <span className="flex items-center gap-1">
                          {h.is_profit ? <TrendingUp size={12}/> : <TrendingDown size={12}/>}
                          {h.pnl_percent > 0 ? "+" : ""}{h.pnl_percent.toFixed(2)}%
                        </span>
                      </td>
                      <td className="px-4 py-3.5">
                        <button onClick={() => removeMut.mutate(h.symbol)}
                          className="opacity-0 group-hover:opacity-100 p-1.5 rounded-lg text-gray-700 hover:text-rose-400 hover:bg-rose-500/10 transition-all">
                          <Trash2 size={12}/>
                        </button>
                      </td>
                    </motion.tr>
                  )})}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// ALERTS VIEW
// ─────────────────────────────────────────────────────────────────────────────

function AlertsView({ onSelectSymbol }: { onSelectSymbol: (s: string) => void }) {
  const qc = useQueryClient();
  const [showAdd, setShowAdd]   = useState(false);
  const [sym, setSym]           = useState("");
  const [price, setPrice]       = useState("");
  const [cond, setCond]         = useState<"above" | "below">("above");
  const [adding, setAdding]     = useState(false);
  const [filter, setFilter]     = useState<"all" | "active" | "triggered">("all");

  const { data, isLoading, refetch } = useQuery<{ alerts: AlertItem[]; count: number }>({
    queryKey: ["alerts"],
    queryFn: async () => { const r = await fetch(`${API}/alerts`); if (!r.ok) throw new Error("Failed"); return r.json(); },
    refetchInterval: 30000,
  });

  const delMut = useMutation({
    mutationFn: async (id: number) => { const r = await fetch(`${API}/alerts/${id}`, { method: "DELETE" }); if (!r.ok) throw new Error("Failed"); },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["alerts"] });
      toast.success("Alert deleted");
    },
    onError: () => toast.error("Failed to delete alert")
  });

  const checkMut = useMutation({
    mutationFn: async () => fetch(`${API}/alerts/check`, { method: "POST" }),
    onSuccess: () => {
      toast.success("Checked alert triggers");
      setTimeout(() => qc.invalidateQueries({ queryKey: ["alerts"] }), 2000);
    },
  });

  const handleAdd = async () => {
    if (!sym.trim() || !price) { toast.error("All fields required"); return; }
    if (isNaN(+price) || +price <= 0) { toast.error("Enter a valid price"); return; }
    
    setAdding(true);
    const alertToast = toast.loading("Creating alert...");
    try {
      const r = await fetch(`${API}/alerts/add`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ symbol: sym.trim().toUpperCase(), target_price: +price, condition: cond }),
      });
      if (!r.ok) { const d = await r.json(); throw new Error(d.detail || "Failed"); }
      
      qc.invalidateQueries({ queryKey: ["alerts"] });
      toast.success("Alert created!", { id: alertToast });
      setShowAdd(false); setSym(""); setPrice("");
    } catch (e: any) { 
      toast.error(e.message, { id: alertToast }); 
    }
    setAdding(false);
  };

  const all    = data?.alerts || [];
  const shown  = filter === "active" ? all.filter(a => !a.triggered) : filter === "triggered" ? all.filter(a => a.triggered) : all;

  return (
    <div className="space-y-5">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Bell size={17} className="text-amber-400" />
          <h2 className="font-black text-lg">Price Alerts</h2>
          {data && <span className="text-xs bg-white/[0.06] text-gray-500 px-2 py-0.5 rounded-full">{data.count}</span>}
        </div>
        <div className="flex items-center gap-2">
          <button onClick={() => checkMut.mutate()} title="Check now"
            className="w-8 h-8 rounded-xl bg-white/[0.04] border border-white/[0.06] flex items-center justify-center text-gray-500 hover:text-white transition-all">
            <RefreshCw size={13} className={checkMut.isPending ? "animate-spin" : ""} />
          </button>
          <button onClick={() => setShowAdd(p => !p)}
            className="flex items-center gap-1.5 px-3.5 py-2 rounded-xl bg-amber-500/80 hover:bg-amber-500 font-bold text-xs text-black transition-all">
            <Plus size={14}/> New Alert
          </button>
        </div>
      </div>

      {/* Add form inline */}
      <AnimatePresence>
        {showAdd && (
          <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: "auto", opacity: 1 }} exit={{ height: 0, opacity: 0 }} className="overflow-hidden">
            <div className="bg-[#0d0d0d] border border-amber-500/20 rounded-2xl p-5 space-y-4">
              <p className="text-xs font-bold text-gray-400 uppercase tracking-wider">New Alert</p>
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                <div>
                  <label className="text-[10px] text-gray-500 font-semibold uppercase tracking-wider block mb-1.5">Symbol</label>
                  <input value={sym} onChange={e => setSym(e.target.value.toUpperCase())} placeholder="AAPL"
                    className="w-full bg-white/[0.04] border border-white/[0.07] rounded-xl px-3.5 py-2.5 text-white text-sm outline-none focus:border-amber-500/40 placeholder-gray-700 font-mono transition-colors" />
                </div>
                <div>
                  <label className="text-[10px] text-gray-500 font-semibold uppercase tracking-wider block mb-1.5">Condition</label>
                  <div className="grid grid-cols-2 gap-1.5">
                    {(["above", "below"] as const).map(c => (
                      <button key={c} onClick={() => setCond(c)}
                        className={`py-2 rounded-xl text-xs font-bold transition-all ${cond === c ? (c === "above" ? "bg-emerald-500/15 text-emerald-400 border border-emerald-500/25" : "bg-rose-500/15 text-rose-400 border border-rose-500/25") : "bg-white/[0.03] text-gray-600 border border-white/[0.06]"}`}>
                        {c === "above" ? "↑ Above" : "↓ Below"}
                      </button>
                    ))}
                  </div>
                </div>
                <div>
                  <label className="text-[10px] text-gray-500 font-semibold uppercase tracking-wider block mb-1.5">Target Price</label>
                  <input type="number" value={price} onChange={e => setPrice(e.target.value)} placeholder="200.00"
                    className="w-full bg-white/[0.04] border border-white/[0.07] rounded-xl px-3.5 py-2.5 text-white text-sm outline-none focus:border-amber-500/40 placeholder-gray-700 transition-colors" />
                </div>
              </div>
              <div className="flex gap-2">
                <button onClick={handleAdd} disabled={adding}
                  className="px-5 py-2.5 rounded-xl bg-amber-500/80 hover:bg-amber-500 text-black disabled:opacity-50 font-bold text-sm transition-all">
                  {adding ? "Creating…" : "Create Alert"}
                </button>
                <button onClick={() => { setShowAdd(false); }} className="px-5 py-2.5 rounded-xl bg-white/[0.04] border border-white/[0.07] font-bold text-sm text-gray-400 hover:text-white transition-all">Cancel</button>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Filter */}
      <div className="flex gap-2">
        {(["all", "active", "triggered"] as const).map(f => (
          <button key={f} onClick={() => setFilter(f)}
            className={`px-3.5 py-2 rounded-xl text-[11px] font-bold uppercase tracking-wider transition-all ${filter === f ? "bg-amber-500/15 text-amber-400 border border-amber-500/20" : "text-gray-600 border border-transparent hover:border-white/10"}`}>
            {f}
          </button>
        ))}
      </div>

      {isLoading ? (
        <div className="flex justify-center py-16"><RefreshCw className="animate-spin text-amber-400" size={22}/></div>
      ) : shown.length === 0 ? (
        <div className="py-20 text-center space-y-3">
          <BellOff size={36} className="text-gray-700 mx-auto"/>
          <p className="text-gray-500 text-sm">{filter === "all" ? "No alerts yet." : `No ${filter} alerts.`}</p>
          {filter === "all" && (
            <button onClick={() => setShowAdd(true)} className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-amber-500/80 hover:bg-amber-500 text-black font-bold text-sm transition-all">
              <Plus size={14}/> Create Alert
            </button>
          )}
        </div>
      ) : (
        <div className="space-y-2.5">
          <AnimatePresence>
            {shown.map((alert, i) => {
              const cS = getCurrencySymbol(alert.symbol);
              return (
              <motion.div key={alert.id} initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, x: -20 }} transition={{ delay: i * 0.03 }}
                className={`group flex items-center gap-4 p-4 rounded-2xl border transition-all ${alert.triggered ? "bg-emerald-500/[0.05] border-emerald-500/15" : "bg-[#0d0d0d] border-white/[0.06] hover:border-white/[0.10]"}`}>
                <div className={`w-10 h-10 rounded-xl flex items-center justify-center shrink-0 ${alert.triggered ? "bg-emerald-500/15" : alert.condition === "above" ? "bg-emerald-500/10" : "bg-rose-500/10"}`}>
                  {alert.triggered ? <CheckCircle2 size={18} className="text-emerald-400"/> : alert.condition === "above" ? <TrendingUp size={18} className="text-emerald-400"/> : <TrendingDown size={18} className="text-rose-400"/>}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 flex-wrap">
                    <button onClick={() => onSelectSymbol(alert.symbol)} className="font-black text-white hover:text-blue-400 transition-colors text-sm">{alert.symbol}</button>
                    <span className={`text-xs font-bold ${alert.condition === "above" ? "text-emerald-400" : "text-rose-400"}`}>
                      {alert.condition} {cS}{alert.target_price.toFixed(2)}
                    </span>
                    {alert.triggered && <Tag color="green">Triggered</Tag>}
                  </div>
                  <p className="text-[10px] text-gray-600 mt-0.5">
                    {alert.triggered && alert.triggered_at
                      ? `At ${cS}${alert.triggered_price?.toFixed(2)} · ${new Date(alert.triggered_at).toLocaleString()}`
                      : `Created ${new Date(alert.created_at).toLocaleDateString()}`}
                  </p>
                </div>
                <button onClick={() => delMut.mutate(alert.id)}
                  className="opacity-0 group-hover:opacity-100 p-2 rounded-xl text-gray-700 hover:text-rose-400 hover:bg-rose-500/10 transition-all">
                  <Trash2 size={13}/>
                </button>
              </motion.div>
            )})}
          </AnimatePresence>
        </div>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// COMPARE VIEW
// ─────────────────────────────────────────────────────────────────────────────

function CompareView({ initialSymbols = [] }: { initialSymbols?: string[] }) {
  const [symbols, setSymbols]           = useState<string[]>(initialSymbols);
  const [searchInput, setSearchInput]   = useState("");
  const [searchRes, setSearchRes]       = useState<SearchResult[]>([]);
  const [searching, setSearching]       = useState(false);

  useEffect(() => {
    if (searchInput.length < 1) { setSearchRes([]); return; }
    const t = setTimeout(async () => {
      setSearching(true);
      try { const r = await fetch(`${API}/search/${encodeURIComponent(searchInput)}`); setSearchRes((await r.json()).slice(0, 5)); }
      catch { setSearchRes([]); }
      setSearching(false);
    }, 300);
    return () => clearTimeout(t);
  }, [searchInput]);

  const addSym = (s: string) => {
    const u = s.trim().toUpperCase();
    if (!u || symbols.includes(u) || symbols.length >= 4) return;
    setSymbols(prev => [...prev, u]);
    setSearchInput(""); setSearchRes([]);
  };
  const removeSym = (s: string) => setSymbols(prev => prev.filter(x => x !== s));

  const { data: cmpData, isLoading } = useQuery<{ data: CompareStock[] }>({
    queryKey: ["compare", symbols.join(",")],
    queryFn: async () => {
      const r = await fetch(`${API}/compare?symbols=${symbols.join(",")}`);
      if (!r.ok) throw new Error("Failed");
      return r.json();
    },
    enabled: symbols.length >= 2,
    staleTime: 5 * 60 * 1000,
  });

  const stocks = cmpData?.data?.filter(s => !s.error) || [];

  // Normalized chart
  const allDates = Array.from(new Set(stocks.flatMap(s => s.prices?.map(p => p.date) || []))).sort().slice(-90);
  const chartData = {
    labels: allDates.map(d => new Date(d).toLocaleDateString("en-US", { month: "short", day: "numeric" })),
    datasets: stocks.map((s, i) => {
      const base = s.prices?.[0]?.close || 1;
      const map  = Object.fromEntries((s.prices || []).map(p => [p.date, ((p.close - base) / base) * 100]));
      return {
        label: s.symbol,
        data: allDates.map(d => map[d] ?? null),
        borderColor: COMPARE_COLORS[i % COMPARE_COLORS.length],
        backgroundColor: "transparent",
        borderWidth: 2, tension: 0.3, pointRadius: 0, pointHoverRadius: 4, spanGaps: true,
      };
    }),
  };

  const METRICS = [
    { key: "current_price",  label: "Price",     fmt: (v: number, s: CompareStock) => `${getCurrencySymbol(s.symbol, s.currency)}${v?.toLocaleString("en-US", { minimumFractionDigits: 2 }) ?? "—"}`, hi: (_: number) => "text-white" },
    { key: "change_1m",      label: "1M Return", fmt: (v: number) => fmtNum(v, "pct"),                                                                                                                     hi: (v: number) => v >= 0 ? "text-emerald-400" : "text-rose-400" },
    { key: "market_cap",     label: "Mkt Cap",   fmt: (v: number, s: CompareStock) => fmtNum(v, "large", getCurrencySymbol(s.symbol, s.currency)),                                                         hi: () => "text-white" },
    { key: "pe_ratio",       label: "P/E",       fmt: (v: number) => v ? `${v.toFixed(1)}x` : "—",                                                                                                         hi: (v: number) => v < 20 ? "text-emerald-400" : v < 35 ? "text-amber-400" : "text-rose-400" },
    { key: "eps",            label: "EPS",       fmt: (v: number, s: CompareStock) => fmtNum(v, "number", getCurrencySymbol(s.symbol, s.currency)),                                                        hi: () => "text-white" },
    { key: "beta",           label: "Beta",      fmt: (v: number) => v?.toFixed(2) ?? "—",                                                                                                                 hi: (v: number) => v < 0.8 ? "text-emerald-400" : v < 1.5 ? "text-amber-400" : "text-rose-400" },
    { key: "dividend_yield", label: "Div Yield", fmt: (v: number) => v ? `${(v * 100).toFixed(2)}%` : "—",                                                                                                 hi: () => "text-white" },
  ];

  return (
    <div className="space-y-5">
      <div className="flex items-center gap-2">
        <GitCompare size={17} className="text-purple-400"/>
        <h2 className="font-black text-lg">Compare Assets</h2>
      </div>

      {/* Symbol chips + search */}
      <div className="flex items-center gap-2 flex-wrap bg-[#0d0d0d] border border-white/[0.06] rounded-2xl p-3">
        {symbols.map((s, i) => (
          <div key={s} className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-black border"
            style={{ borderColor: COMPARE_COLORS[i % COMPARE_COLORS.length] + "50", backgroundColor: COMPARE_COLORS[i % COMPARE_COLORS.length] + "15", color: COMPARE_COLORS[i % COMPARE_COLORS.length] }}>
            {s}
            <button onClick={() => removeSym(s)} className="opacity-70 hover:opacity-100"><X size={11}/></button>
          </div>
        ))}
        {symbols.length < 4 && (
          <div className="relative">
            <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-white/[0.04] border border-white/[0.07]">
              <Search size={12} className="text-gray-600"/>
              <input value={searchInput} onChange={e => setSearchInput(e.target.value)}
                onKeyDown={e => { if (e.key === "Enter" && searchInput.trim()) addSym(searchInput); }}
                placeholder="Add symbol…" className="bg-transparent text-xs outline-none text-white placeholder-gray-600 w-20" />
              {searching && <RefreshCw size={10} className="animate-spin text-blue-400"/>}
            </div>
            {searchRes.length > 0 && (
              <div className="absolute top-full mt-1 left-0 w-52 bg-[#111] border border-white/[0.08] rounded-xl overflow-hidden shadow-2xl z-50">
                {searchRes.map(s => (
                  <button key={s.symbol} onClick={() => addSym(s.symbol)}
                    className="w-full text-left px-3 py-2.5 hover:bg-white/[0.05] text-xs flex items-center gap-2 transition-colors">
                    <Plus size={10} className="text-blue-400 shrink-0"/>
                    <span className="font-black text-white">{s.symbol}</span>
                    <span className="text-gray-600 truncate">{s.name}</span>
                  </button>
                ))}
              </div>
            )}
          </div>
        )}
        <span className="text-[10px] text-gray-700 ml-auto">Up to 4 assets</span>
      </div>

      {symbols.length < 2 ? (
        <div className="py-16 text-center space-y-2">
          <p className="text-gray-500 text-sm">Add at least 2 symbols to compare.</p>
        </div>
      ) : isLoading ? (
        <div className="flex justify-center py-16"><RefreshCw className="animate-spin text-purple-400" size={22}/></div>
      ) : (
        <>
          {/* Perf chart */}
          {stocks.length >= 2 && (
            <div className="bg-[#0d0d0d] border border-white/[0.06] rounded-2xl p-5">
              <p className="text-[10px] text-gray-500 uppercase font-bold tracking-widest mb-1">Relative Performance (90 days)</p>
              <p className="text-[10px] text-gray-700 mb-4">Normalized to % return from period start</p>
              <div style={{ height: 260 }}>
                <Line data={chartData} options={{
                  responsive: true, maintainAspectRatio: false,
                  plugins: {
                    legend: { display: true, labels: { color: "#6b7280", font: { size: 11 }, boxWidth: 12, padding: 16 } },
                    tooltip: { backgroundColor: "#111", borderColor: "rgba(255,255,255,0.08)", borderWidth: 1, padding: 10, titleColor: "#9ca3af", bodyColor: "#fff", callbacks: { label: (ctx: any) => ` ${ctx.dataset.label}: ${ctx.parsed.y?.toFixed(2) ?? "—"}%` } },
                  },
                  scales: {
                    x: { grid: { display: false }, ticks: { color: "#4b5563", maxTicksLimit: 6, font: { size: 10 } } },
                    y: { grid: { color: "rgba(255,255,255,0.03)" }, ticks: { color: "#4b5563", font: { size: 10 }, callback: (v: any) => `${Number(v).toFixed(0)}%` } },
                  },
                  interaction: { mode: "index", intersect: false },
                }} />
              </div>
            </div>
          )}

          {/* Metrics table */}
          <div className="bg-[#0d0d0d] border border-white/[0.06] rounded-2xl overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-white/[0.05]">
                    <th className="text-left px-5 py-4 text-[10px] text-gray-600 uppercase tracking-wider font-semibold w-32">Metric</th>
                    {stocks.map((s, i) => (
                      <th key={s.symbol} className="px-5 py-4 text-center">
                        <div className="flex flex-col items-center gap-1">
                          <span className="text-sm font-black" style={{ color: COMPARE_COLORS[i % COMPARE_COLORS.length] }}>{s.symbol}</span>
                          {s.company_name && <span className="text-[10px] text-gray-600 font-normal max-w-[100px] truncate">{s.company_name}</span>}
                          {s.sector && <span className="text-[9px] bg-white/[0.04] text-gray-600 border border-white/[0.05] px-1.5 py-0.5 rounded-full">{s.sector}</span>}
                        </div>
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {METRICS.map(m => (
                    <tr key={m.key} className="border-b border-white/[0.03] hover:bg-white/[0.02] transition-colors">
                      <td className="px-5 py-3.5 text-[11px] text-gray-500 font-semibold">{m.label}</td>
                      {stocks.map(s => {
                        const raw = s[m.key as keyof CompareStock] as number | undefined;
                        return (
                          <td key={s.symbol} className={`px-5 py-3.5 text-center text-sm font-black ${raw != null ? m.hi(raw) : "text-gray-700"}`}>
                            {raw != null ? m.fmt(raw, s) : "N/A"}
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// SEARCH BAR
// ─────────────────────────────────────────────────────────────────────────────

function SearchBar({ onSelect, activeSymbol }: { onSelect: (s: string) => void; activeSymbol: string | null }) {
  const [input, setInput]               = useState("");
  const [results, setResults]           = useState<SearchResult[]>([]);
  const [loading, setLoading]           = useState(false);
  const [open, setOpen]                 = useState(false);
  const [idx, setIdx]                   = useState(-1);
  const [recent, setRecent]             = useLocalStorage<string[]>("fin_recent_v2", []);
  const debouncedInput                  = useDebounce(input, 320);
  const inputRef                        = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (debouncedInput.length < 1) { setResults([]); return; }
    setLoading(true);
    fetch(`${API}/search/${encodeURIComponent(debouncedInput)}`)
      .then(r => r.json()).then(d => { setResults(Array.isArray(d) ? d : []); setOpen(true); })
      .catch(() => setResults([]))
      .finally(() => setLoading(false));
  }, [debouncedInput]);

  // Ctrl+K
  useEffect(() => {
    const h = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "k") { e.preventDefault(); inputRef.current?.focus(); }
      if (e.key === "Escape") { setOpen(false); setInput(""); }
    };
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  }, []);

  const handleSelect = (sym: string) => {
    const s = sym.trim().toUpperCase();
    if (!s) return;
    setRecent([s, ...recent.filter(x => x !== s)].slice(0, 6));
    setInput(""); setOpen(false); setIdx(-1);
    onSelect(s);
  };

  const handleKey = (e: React.KeyboardEvent) => {
    const list = results.length ? results : [];
    if (e.key === "Enter") { if (list[idx]) handleSelect(list[idx].symbol); else if (input.trim()) handleSelect(input.trim()); }
    else if (e.key === "ArrowDown") { e.preventDefault(); setIdx(p => (p + 1) % Math.max(list.length, 1)); }
    else if (e.key === "ArrowUp")   { e.preventDefault(); setIdx(p => p <= 0 ? list.length - 1 : p - 1); }
  };

  const showDrop = open && (input.length > 0 || recent.length > 0);
  const typeMap: Record<string, string> = { EQUITY: "Stock", CRYPTOCURRENCY: "Crypto", ETF: "ETF", INDEX: "Index" };

  return (
    <>
      {showDrop && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-40" onClick={() => { setOpen(false); setInput(""); }} />
      )}
      <div className="relative z-50">
        <div className={`flex items-center gap-2 p-1.5 rounded-2xl bg-[#0d0d0d] border transition-all duration-300 ${open && input ? "border-blue-500/40 shadow-[0_0_30px_rgba(59,130,246,0.12)]" : "border-white/[0.07]"}`}>
          <div className="ml-3">
            {loading ? <RefreshCw size={16} className="text-blue-400 animate-spin"/> : <Search size={16} className={open ? "text-blue-400" : "text-gray-600"}/>}
          </div>
          <input ref={inputRef} value={input} onChange={e => { setInput(e.target.value); setIdx(-1); if (e.target.value) setOpen(true); }}
            onKeyDown={handleKey} onFocus={() => setOpen(true)}
            placeholder="Search symbol… (Ctrl+K)"
            className="flex-1 bg-transparent px-2 py-3 outline-none text-white text-sm placeholder-gray-700 font-medium"
            autoComplete="off" spellCheck={false}
          />
          {input ? (
            <button onClick={() => { setInput(""); inputRef.current?.focus(); }} className="mr-2 text-gray-600 hover:text-white transition-colors"><X size={14}/></button>
          ) : (
            <div className="hidden sm:flex items-center gap-1 px-2 py-1 mr-2 rounded-lg bg-white/[0.04] border border-white/[0.06] text-gray-600 text-[10px] font-bold uppercase"><Command size={9}/> K</div>
          )}
          <button onClick={() => input.trim() && handleSelect(input.trim())}
            className="px-5 py-3 rounded-xl bg-gradient-to-r from-blue-600 to-blue-500 font-bold text-sm hover:opacity-90 active:scale-95 transition-all flex items-center gap-1.5 whitespace-nowrap">
            Analyze <ArrowRight size={14}/>
          </button>
        </div>

        <AnimatePresence>
          {showDrop && (
            <motion.div initial={{ opacity: 0, y: -8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -8 }}
              transition={{ type: "spring", stiffness: 500, damping: 40 }}
              className="absolute w-full mt-2 bg-[#0d0d0d] border border-white/[0.07] rounded-2xl overflow-hidden shadow-[0_40px_80px_rgba(0,0,0,0.6)] max-h-96 overflow-y-auto">

              {!input && recent.length > 0 && (
                <div>
                  <div className="px-4 pt-4 pb-2 flex justify-between items-center">
                    <span className="text-[10px] text-gray-600 font-bold uppercase tracking-wider">Recent</span>
                    <button onClick={() => setRecent([])} className="text-[10px] text-gray-700 hover:text-gray-400">Clear</button>
                  </div>
                  {recent.map((s, i) => (
                    <div key={i} onClick={() => handleSelect(s)} className="flex items-center gap-3 px-4 py-3 hover:bg-white/[0.03] cursor-pointer group transition-colors">
                      <Clock size={12} className="text-gray-700 group-hover:text-gray-500"/>
                      <span className="text-sm text-gray-400 group-hover:text-white font-medium">{s}</span>
                      <ArrowRight size={12} className="text-gray-700 ml-auto"/>
                    </div>
                  ))}
                  <div className="border-t border-white/[0.04] my-1"/>
                </div>
              )}

              {input && loading && (
                <div className="p-6 flex flex-col items-center gap-2 text-gray-600">
                  <RefreshCw size={18} className="animate-spin text-blue-500"/>
                  <span className="text-xs">Scanning markets…</span>
                </div>
              )}
              {input && !loading && results.length === 0 && (
                <div className="p-6 text-center text-gray-600 text-sm">
                  No results — press <span className="text-white font-bold">Enter</span> to search &ldquo;{input}&rdquo; directly
                </div>
              )}
              {results.map((item, i) => (
                <div key={`${item.symbol}-${i}`} onClick={() => handleSelect(item.symbol)} onMouseEnter={() => setIdx(i)}
                  className={`px-4 py-3.5 cursor-pointer flex items-center justify-between border-l-2 transition-all ${i === idx ? "bg-white/[0.05] border-blue-500" : "border-transparent hover:bg-white/[0.02]"}`}>
                  <div className="flex items-center gap-3">
                    <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-white/[0.06] to-white/[0.02] border border-white/[0.07] flex items-center justify-center text-xs font-black text-blue-400">
                      {item.symbol.slice(0, 2)}
                    </div>
                    <div>
                      <p className="font-black text-sm text-white">{item.symbol}</p>
                      <p className="text-xs text-gray-600 truncate max-w-[200px]">{item.name}</p>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-[10px] text-gray-700">{item.exchange}</span>
                    <span className="text-[10px] bg-white/[0.04] px-2 py-0.5 rounded text-gray-500 font-mono">{typeMap[item.type || ""] || "EQUITY"}</span>
                  </div>
                </div>
              ))}
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// MAIN PAGE
// ─────────────────────────────────────────────────────────────────────────────

export default function DashboardPage() {
  return (
    <Suspense fallback={<div className="text-center mt-20">Loading Dashboard...</div>}>
      <DashboardContent />
    </Suspense>
  );
}

function DashboardContent() {
  const searchParams                    = useSearchParams();
  const [view, setView]                 = useState<View>("dashboard");
  const [activeSymbol, setActiveSymbol] = useState<string | null>(searchParams.get("symbol") || null);
  const [watchlist, setWatchlist]       = useLocalStorage<string[]>("fin_watchlist", []);
  const [compareSyms, setCompareSyms]   = useState<string[]>([]);
  const [lastUpdated, setLastUpdated]   = useState("");
  const qc                              = useQueryClient();

  // Restore last symbol on mount
  useEffect(() => {
    try {
      const last = localStorage.getItem("lastStock");
      if (!searchParams.get("symbol") && last) setActiveSymbol(last);
    } catch {}
    setLastUpdated(new Date().toLocaleTimeString());
  }, []);

  useEffect(() => {
    if (activeSymbol) { localStorage.setItem("lastStock", activeSymbol); setLastUpdated(new Date().toLocaleTimeString()); }
  }, [activeSymbol]);

  const handleSelect = (sym: string) => {
    setActiveSymbol(sym);
    setView("dashboard");
  };

  const handleRefresh = async () => {
    if (!activeSymbol) return;
    const loadToast = toast.loading("Refreshing market data...");
    await qc.invalidateQueries({ queryKey: ["stock", activeSymbol] });
    setLastUpdated(new Date().toLocaleTimeString());
    toast.success("Data refreshed", { id: loadToast });
  };

  const toggleWatchlist = (sym: string) => {
    const isAdded = watchlist.includes(sym);
    setWatchlist(isAdded ? watchlist.filter(s => s !== sym) : [...watchlist, sym]);
    toast.success(isAdded ? "Removed from watchlist" : "Added to watchlist");
  };

  const openCompare = (sym: string) => {
    setCompareSyms(sym ? [sym] : []);
    setView("compare");
  };

  const navItems: { id: View; icon: React.ReactNode; label: string; color: string }[] = [
    { id: "dashboard", icon: <BarChart2 size={16}/>, label: "Analysis",  color: "text-blue-400" },
    { id: "portfolio", icon: <Briefcase size={16}/>, label: "Portfolio", color: "text-emerald-400" },
    { id: "alerts",    icon: <Bell size={16}/>,      label: "Alerts",    color: "text-amber-400" },
    { id: "compare",   icon: <GitCompare size={16}/>,label: "Compare",   color: "text-purple-400" },
  ];

  return (
    <main className="min-h-screen bg-[#050505] text-white overflow-x-hidden">
      <style>{`
        @keyframes shimmer { 0%{transform:translateX(-100%)} 100%{transform:translateX(200%)} }
        .scrollbar-hide::-webkit-scrollbar{display:none}
        .scrollbar-hide{-ms-overflow-style:none;scrollbar-width:none}
      `}</style>

      {/* Background glows */}
      <div className="fixed inset-0 -z-10 pointer-events-none overflow-hidden">
        <div className="absolute top-0 left-[-10%] w-[500px] h-[500px] rounded-full bg-blue-600/[0.04] blur-[130px]"/>
        <div className="absolute bottom-0 right-0 w-[400px] h-[400px] rounded-full bg-purple-600/[0.03] blur-[120px]"/>
        <div className="absolute top-[50%] left-[50%] w-[300px] h-[300px] rounded-full bg-indigo-600/[0.02] blur-[100px]"/>
      </div>

      {/* ── TOP NAV ── */}
      <div className="sticky top-20 z-50 bg-[#050505]/90 backdrop-blur-xl border-b border-white/[0.04]">
        <div className="max-w-7xl mx-auto px-4 sm:px-6">
          {/* Top row: logo + search + actions */}
          {/* <div className="flex items-center gap-3 py-3.5"> */}
            {/* Logo */}
            {/* <div className="flex items-center gap-2 shrink-0">
              <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-blue-500/30 to-blue-600/20 border border-blue-500/30 flex items-center justify-center">
                <Activity size={15} className="text-blue-400"/>
              </div>
              <span className="font-black text-sm hidden sm:block">FinIntel</span>
            </div> */}

            {/* Search */}
            {/* <div className="flex-1 m  ax-w-2xl">
              <SearchBar onSelect={handleSelect} activeSymbol={activeSymbol} />
            </div> */}

            {/* Right actions */}
            {/* {activeSymbol && view === "dashboard" && (
              <div className="flex items-center gap-1.5 shrink-0">
                <button onClick={handleRefresh}
                  className="w-9 h-9 rounded-xl bg-white/[0.04] border border-white/[0.06] flex items-center justify-center text-gray-500 hover:text-white transition-all" title="Refresh">
                  <RefreshCw size={14}/>
                </button>
                <button onClick={() => toggleWatchlist(activeSymbol)}
                  className="w-9 h-9 rounded-xl bg-white/[0.04] border border-white/[0.06] flex items-center justify-center transition-all" title="Watchlist">
                  <Star size={14} className={watchlist.includes(activeSymbol) ? "text-yellow-400 fill-yellow-400" : "text-gray-500"}/>
                </button>
                <button onClick={() => openCompare(activeSymbol)}
                  className="hidden sm:flex items-center gap-1.5 px-3 py-2 rounded-xl bg-purple-500/10 border border-purple-500/20 text-purple-400 hover:bg-purple-500/20 font-bold text-xs transition-all">
                  <GitCompare size={13}/> Compare
                </button>
              </div>
            )} */}
          {/* </div> */}

          {/* Nav tabs */}
          <div className="flex items-center gap-1 pb-0 -mb-px overflow-x-auto scrollbar-hide">
            {navItems.map(n => (
              <button key={n.id} onClick={() => setView(n.id)}
                className={`flex items-center gap-1.5 px-4 py-2.5 text-xs font-bold whitespace-nowrap transition-all border-b-2 ${view === n.id ? `${n.color} border-current` : "text-gray-600 border-transparent hover:text-gray-400"}`}>
                {n.icon}{n.label}
              </button>
            ))}
            {activeSymbol && view === "dashboard" && (
              <div className="ml-auto pb-2.5 flex items-center gap-1.5 text-[10px] text-gray-700 font-mono">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse"/>
                LIVE · {lastUpdated}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* ── CONTENT ── */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 py-25 grid grid-cols-1 lg:grid-cols-12 gap-6">

        {/* Main content */}
        <div className="lg:col-span-8 min-w-0 ">
          <AnimatePresence mode="wait">
            {view === "dashboard" && (
              <motion.div key="dashboard" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}>
                {!activeSymbol ? (
                  <EmptyState watchlist={watchlist} onSelect={handleSelect} onRemove={toggleWatchlist} />
                ) : (
                  <StockAnalysis symbol={activeSymbol} />
                )}
              </motion.div>
            )}
            {view === "portfolio" && (
              <motion.div key="portfolio" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}>
                <PortfolioView onSelectSymbol={sym => { handleSelect(sym); setView("dashboard"); }} />
              </motion.div>
            )}
            {view === "alerts" && (
              <motion.div key="alerts" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}>
                <AlertsView onSelectSymbol={sym => { handleSelect(sym); setView("dashboard"); }} />
              </motion.div>
            )}
            {view === "compare" && (
              <motion.div key="compare" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}>
                <CompareView initialSymbols={compareSyms} />
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Right sidebar */}
        <aside className="lg:col-span-4 space-y-5">
          <div className="sticky top-[100px] space-y-5">
            {/* Live news for active symbol */}
            <RightNewsSidebar symbol={activeSymbol} />

            {/* Watchlist */}
            {watchlist.length > 0 && (
              <div className="bg-[#0d0d0d] border border-white/[0.06] rounded-2xl p-5">
                <p className="text-[10px] text-gray-500 uppercase font-bold tracking-widest mb-4 flex items-center gap-2">
                  <Star size={11} className="text-yellow-400 fill-yellow-400"/> Watchlist
                </p>
                <div className="space-y-1.5">
                  {watchlist.slice(0, 7).map(sym => (
                    <div key={sym} className="flex items-center justify-between group">
                      <button onClick={() => handleSelect(sym)} className="flex items-center gap-2.5 flex-1 px-2 py-2 rounded-xl hover:bg-white/[0.03] transition-colors text-left">
                        <CompanyLogo symbol={sym} size={7}/>
                        <span className="text-sm font-bold text-gray-300 group-hover:text-white transition-colors">{sym}</span>
                      </button>
                      <button onClick={() => toggleWatchlist(sym)} className="p-1.5 text-gray-700 hover:text-rose-400 transition-colors opacity-0 group-hover:opacity-100">
                        <X size={11}/>
                      </button>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </aside>
      </div>
    </main>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// EMPTY STATE
// ─────────────────────────────────────────────────────────────────────────────

function EmptyState({ watchlist, onSelect, onRemove }: { watchlist: string[]; onSelect: (s: string) => void; onRemove: (s: string) => void }) {
  return (
    <div className="space-y-10 py-8">
      <div className="text-center space-y-3">
        <div className="w-14 h-14 rounded-2xl bg-blue-500/10 border border-blue-500/20 flex items-center justify-center mx-auto">
          <BarChart2 size={24} className="text-blue-400"/>
        </div>
        <h1 className="text-3xl font-black tracking-tight">Market Intelligence</h1>
        <p className="text-gray-600 text-sm max-w-sm mx-auto">Search any stock, crypto, or index above to get instant AI-powered analysis.</p>
      </div>

      {watchlist.length > 0 && (
        <section>
          <p className="text-[10px] text-gray-600 uppercase font-bold tracking-widest mb-4 flex items-center gap-2">
            <Star size={11} className="text-yellow-400 fill-yellow-400"/> Watchlist
          </p>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {watchlist.map(sym => (
              <motion.div key={sym} whileHover={{ y: -2 }} onClick={() => onSelect(sym)}
                className="p-4 rounded-2xl bg-[#0d0d0d] border border-white/[0.05] hover:border-blue-500/25 cursor-pointer flex justify-between items-center group transition-all">
                <div className="flex items-center gap-3">
                  <CompanyLogo symbol={sym}/>
                  <span className="font-black">{sym}</span>
                </div>
                <div className="flex items-center gap-2">
                  <ArrowRight size={13} className="text-gray-700 group-hover:text-blue-400 transition-colors"/>
                  <button onClick={e => { e.stopPropagation(); onRemove(sym); }} className="p-1 text-gray-700 hover:text-rose-400 transition-colors opacity-0 group-hover:opacity-100">
                    <X size={11}/>
                  </button>
                </div>
              </motion.div>
            ))}
          </div>
        </section>
      )}

      <section>
        <p className="text-[10px] text-gray-600 uppercase font-bold tracking-widest mb-4">Popular</p>
        <div className="flex flex-wrap gap-2">
          {POPULAR.map(sym => (
            <button key={sym} onClick={() => onSelect(sym)}
              className="px-4 py-2 rounded-xl bg-white/[0.03] border border-white/[0.06] hover:border-blue-500/30 hover:bg-blue-500/[0.07] text-sm font-bold text-gray-400 hover:text-white transition-all">
              {sym}
            </button>
          ))}
        </div>
      </section>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// RIGHT NEWS SIDEBAR
// ─────────────────────────────────────────────────────────────────────────────

function RightNewsSidebar({ symbol }: { symbol: string | null }) {
  const { data, isLoading } = useQuery<NewsItem[]>({
    queryKey: ["news", symbol || "GLOBAL"],
    queryFn: async () => {
      const r = await fetch(`${API}/news/${symbol || "AAPL"}`);
      const j = await r.json();
      return j.news || [];
    },
    staleTime: 10 * 60 * 1000,
  });

  const news = data || [];

  return (
    <div className="bg-[#0d0d0d] border border-white/[0.06] rounded-2xl p-5">
      <div className="flex items-center justify-between mb-5">
        <p className="text-[10px] text-gray-500 uppercase font-bold tracking-widest flex items-center gap-2">
          <Newspaper size={12} className="text-blue-400"/>
          {symbol ? `${symbol} News` : "Market News"}
        </p>
        <div className="flex items-center gap-1.5 text-[10px] text-emerald-400">
          <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse"/> Live
        </div>
      </div>

      {isLoading ? (
        <div className="space-y-3">{[...Array(4)].map((_, i) => <Skeleton key={i} className="h-16 w-full"/>)}</div>
      ) : news.length > 0 ? (
        <div className="space-y-4">
          {news.slice(0, 7).map((n, i) => {
            const href = n.url?.trim() || n.link?.trim() || `https://www.google.com/search?tbm=nws&q=${encodeURIComponent(n.title)}`;
            return (
              <a key={n.id ?? i} href={href} target="_blank" rel="noopener noreferrer"
                className="block group space-y-1 pl-3 border-l-2 border-transparent hover:border-blue-500 transition-all">
                <div className="flex items-center gap-1.5 text-[10px]">
                  <SentimentTag s={n.sentiment}/>
                  <span className="text-blue-400/70 font-bold">{n.source}</span>
                  <span className="text-gray-700 ml-auto flex items-center gap-0.5"><Clock size={9}/>{n.time}</span>
                </div>
                {/* Fixed the fallback to check n.text if title is generic */}
                <p className="text-xs font-semibold text-gray-400 group-hover:text-white leading-snug transition-colors line-clamp-2 flex gap-1.5">
                  <span className="flex-1">{n.title || n.text || "Market Update"}</span>
                  <ExternalLink size={9} className="opacity-0 group-hover:opacity-60 shrink-0 mt-0.5"/>
                </p>
              </a>
            );
          })}
        </div>
      ) : (
        <p className="text-gray-700 text-xs text-center py-8">No recent headlines.</p>
      )}
    </div>
  );
}