"use client";

import { useEffect, useState, useMemo } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Wallet, TrendingUp, TrendingDown, PieChart, Activity,
  Plus, Search, Building2, ArrowUpRight, ArrowDownRight,
  RefreshCcw, ChevronRight, BarChart2, Layers, Eye, EyeOff,
} from "lucide-react";
import Link from "next/link";

// ─── TYPES ───────────────────────────────────────────────────
interface PortfolioItem {
  symbol: string;
  quantity: number;
  price: number;
  current_price?: number;
  pnl?: number;
  pnl_percent?: number;
  sentiment_score?: number;
}

// ─── CONFIG ──────────────────────────────────────────────────
const API_BASE = (process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000").replace(/\/$/, "");

// ─── FORMATTERS ──────────────────────────────────────────────
const fmtINR = (v: number) =>
  new Intl.NumberFormat("en-IN", { style: "currency", currency: "INR", maximumFractionDigits: 2 }).format(v);

const fmtCompact = (v: number) =>
  new Intl.NumberFormat("en-IN", { style: "currency", currency: "INR", notation: "compact", maximumFractionDigits: 1 }).format(v);

// ─── SENTIMENT ───────────────────────────────────────────────
function getSentiment(v?: number): { label: string; cls: string; dot: string } {
  if (v === undefined) return { label: "Neutral", cls: "text-gray-500 bg-gray-500/10 border-gray-500/20", dot: "bg-gray-500" };
  if (v > 0.6) return { label: "Bullish", cls: "text-emerald-400 bg-emerald-500/10 border-emerald-500/20", dot: "bg-emerald-400" };
  if (v < 0.4) return { label: "Bearish", cls: "text-rose-400 bg-rose-500/10 border-rose-500/20", dot: "bg-rose-400" };
  return { label: "Neutral", cls: "text-amber-400 bg-amber-500/10 border-amber-500/20", dot: "bg-amber-400" };
}

// ─── MINI SPARKLINE ──────────────────────────────────────────
function Sparkline({ positive }: { positive: boolean }) {
  const pts = positive
    ? "0,20 10,18 20,15 30,16 40,12 50,10 60,8 70,9 80,5 90,4 100,2"
    : "0,2 10,4 20,5 30,4 40,8 50,10 60,12 70,11 80,15 90,17 100,20";
  const color = positive ? "#34d399" : "#f87171";
  return (
    <svg viewBox="0 0 100 22" className="w-20 h-6" preserveAspectRatio="none">
      <defs>
        <linearGradient id={`sg-${positive}`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity="0.3" />
          <stop offset="100%" stopColor={color} stopOpacity="0" />
        </linearGradient>
      </defs>
      <polygon points={`0,22 ${pts} 100,22`} fill={`url(#sg-${positive})`} />
      <polyline points={pts} fill="none" stroke={color} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

// ─── ALLOCATION BAR ──────────────────────────────────────────
function AllocationBar({ items }: { items: { symbol: string; value: number; color: string }[] }) {
  const total = items.reduce((s, i) => s + i.value, 0);
  return (
    <div className="flex h-1.5 rounded-full overflow-hidden gap-px">
      {items.map((item) => (
        <motion.div
          key={item.symbol}
          initial={{ flex: 0 }}
          animate={{ flex: item.value / total }}
          transition={{ duration: 0.8, ease: "easeOut" }}
          style={{ backgroundColor: item.color }}
        />
      ))}
    </div>
  );
}

// ─── COMPANY AVATAR ──────────────────────────────────────────
function CompanyAvatar({ symbol, size = "md" }: { symbol: string; size?: "sm" | "md" }) {
  const [err, setErr] = useState(false);
  const clean = symbol.split(".")[0];
  const dim = size === "sm" ? "w-8 h-8 text-xs" : "w-11 h-11 text-sm";
  if (err)
    return (
      <div className={`${dim} rounded-xl bg-gradient-to-br from-blue-500/20 to-cyan-500/20 border border-white/10 flex items-center justify-center font-bold text-white shrink-0`}>
        {clean[0]}
      </div>
    );
  return (
    <img
      src={`https://logo.clearbit.com/${clean.toLowerCase()}.com`}
      alt={symbol}
      onError={() => setErr(true)}
      className={`${dim} rounded-xl object-cover bg-white/5 border border-white/5 shrink-0`}
    />
  );
}

// ─── METRIC CARD ─────────────────────────────────────────────
function MetricCard({
  icon, label, value, sub, accent, delay = 0,
}: {
  icon: React.ReactNode; label: string; value: string; sub?: string; accent?: "green" | "red" | "blue"; delay?: number;
}) {
  const accents = {
    green: { bar: "from-emerald-500 to-emerald-400", glow: "shadow-emerald-500/10", text: "text-emerald-400" },
    red: { bar: "from-rose-500 to-rose-400", glow: "shadow-rose-500/10", text: "text-rose-400" },
    blue: { bar: "from-blue-500 to-cyan-400", glow: "shadow-blue-500/10", text: "text-blue-400" },
  };
  const a = accent ? accents[accent] : null;

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay, duration: 0.5, ease: "easeOut" }}
      className={`relative bg-[#0d0d14] border border-white/[0.06] rounded-2xl p-5 overflow-hidden ${a ? `shadow-lg ${a.glow}` : ""}`}
    >
      {a && (
        <div className={`absolute top-0 left-0 right-0 h-px bg-gradient-to-r ${a.bar} opacity-60`} />
      )}
      <div className="flex items-center gap-2 text-gray-500 text-xs font-medium mb-3 uppercase tracking-widest">
        {icon}
        {label}
      </div>
      <div className={`text-2xl font-bold tracking-tight ${a ? a.text : "text-white"}`}>{value}</div>
      {sub && <div className="text-xs text-gray-600 mt-1 font-mono">{sub}</div>}
    </motion.div>
  );
}

// ─── HOLDING ROW ─────────────────────────────────────────────
const PALETTE = ["#3b82f6", "#06b6d4", "#8b5cf6", "#f59e0b", "#10b981", "#ec4899", "#f97316"];

function HoldingRow({ stock, idx, totalValue, hideValues }: {
  stock: PortfolioItem; idx: number; totalValue: number; hideValues: boolean;
}) {
  const [expanded, setExpanded] = useState(false);
  const price = stock.current_price ?? stock.price;
  const invested = stock.price * stock.quantity;
  const current = price * stock.quantity;
  const pnlPct = stock.pnl_percent ?? 0;
  const pnlAbs = stock.pnl ?? current - invested;
  const isUp = pnlPct >= 0;
  const alloc = totalValue > 0 ? (current / totalValue) * 100 : 0;
  const sentiment = getSentiment(stock.sentiment_score);
  const color = PALETTE[idx % PALETTE.length];

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: idx * 0.06, duration: 0.4, ease: "easeOut" }}
      className="group"
    >
      <div
        className="relative bg-[#0d0d14] border border-white/[0.06] rounded-2xl overflow-hidden cursor-pointer hover:border-white/[0.12] transition-all duration-200"
        onClick={() => setExpanded((p) => !p)}
      >
        {/* Allocation accent bar */}
        <div
          className="absolute left-0 top-0 bottom-0 w-[3px] transition-opacity opacity-40 group-hover:opacity-80"
          style={{ backgroundColor: color }}
        />

        <div className="flex items-center gap-4 px-5 py-4 pl-6">
          {/* Identity */}
          <CompanyAvatar symbol={stock.symbol} />
          <div className="min-w-0 flex-1">
            <div className="flex items-center gap-2 mb-0.5">
              <span className="font-bold text-white text-[15px]">{stock.symbol}</span>
              <span className={`text-[9px] font-bold uppercase tracking-wider px-1.5 py-0.5 rounded border flex items-center gap-1 ${sentiment.cls}`}>
                <span className={`w-1 h-1 rounded-full ${sentiment.dot}`} />
                {sentiment.label}
              </span>
            </div>
            <div className="text-xs text-gray-600 font-mono">
              {stock.quantity} units · avg {hideValues ? "••••" : fmtINR(stock.price)}
            </div>
          </div>

          {/* Sparkline */}
          <div className="hidden sm:block shrink-0">
            <Sparkline positive={isUp} />
          </div>

          {/* Price */}
          <div className="hidden md:block text-right shrink-0 min-w-[90px]">
            <div className="text-sm font-mono text-gray-300">{hideValues ? "••••" : fmtINR(price)}</div>
            <div className="text-[10px] text-gray-600">market price</div>
          </div>

          {/* Value */}
          <div className="hidden lg:block text-right shrink-0 min-w-[100px]">
            <div className="text-sm font-semibold text-gray-200">{hideValues ? "••••" : fmtCompact(current)}</div>
            <div className="text-[10px] text-gray-600">{alloc.toFixed(1)}% of portfolio</div>
          </div>

          {/* PnL */}
          <div className="text-right shrink-0 min-w-[90px]">
            <div className={`text-base font-bold flex items-center justify-end gap-0.5 ${isUp ? "text-emerald-400" : "text-rose-400"}`}>
              {isUp ? <ArrowUpRight size={14} /> : <ArrowDownRight size={14} />}
              {hideValues ? "••" : `${isUp ? "+" : ""}${pnlPct.toFixed(2)}%`}
            </div>
            <div className={`text-[11px] font-mono ${isUp ? "text-emerald-600" : "text-rose-600"}`}>
              {hideValues ? "••••" : `${isUp ? "+" : ""}${fmtINR(pnlAbs)}`}
            </div>
          </div>

          <ChevronRight
            size={14}
            className={`text-gray-700 transition-transform duration-200 shrink-0 ${expanded ? "rotate-90" : ""}`}
          />
        </div>

        {/* Allocation bar */}
        <div className="px-6 pb-3">
          <div className="h-px bg-white/[0.04] rounded-full overflow-hidden">
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: `${alloc}%` }}
              transition={{ duration: 0.9, delay: idx * 0.06, ease: "easeOut" }}
              className="h-full rounded-full"
              style={{ backgroundColor: color }}
            />
          </div>
        </div>
      </div>

      {/* Expanded Detail */}
      <AnimatePresence>
        {expanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 px-1 pt-2 pb-1">
              {[
                { label: "Avg Buy Price", value: fmtINR(stock.price) },
                { label: "Current Price", value: fmtINR(price) },
                { label: "Invested", value: fmtINR(invested) },
                { label: "Current Value", value: fmtINR(current) },
              ].map((d) => (
                <div key={d.label} className="bg-[#0d0d14] border border-white/[0.05] rounded-xl px-4 py-3">
                  <div className="text-[10px] text-gray-600 uppercase tracking-wider mb-1">{d.label}</div>
                  <div className="text-sm font-mono font-semibold text-gray-300">
                    {hideValues ? "••••" : d.value}
                  </div>
                </div>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}

// ─── SKELETON ────────────────────────────────────────────────
function Skeleton() {
  return (
    <div className="space-y-3">
      {[0, 1, 2].map((i) => (
        <div key={i} className="h-[76px] bg-white/[0.03] rounded-2xl animate-pulse border border-white/[0.04]" />
      ))}
    </div>
  );
}

// ─── MAIN PAGE ───────────────────────────────────────────────
export default function Portfolio() {
  const [data, setData] = useState<PortfolioItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [hideValues, setHideValues] = useState(false);
  const [sortBy, setSortBy] = useState<"value" | "pnl" | "alloc">("value");

  const fetchPortfolio = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/portfolio`);
      if (!res.ok) throw new Error(`Server error ${res.status}`);
      const json = await res.json();
      setData(json);
    } catch (e) {
      setError("Unable to load portfolio. Check your connection and try again.");
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { fetchPortfolio(); }, []);

  const metrics = useMemo(() => {
    let invested = 0, current = 0;
    data.forEach((s) => {
      invested += s.price * s.quantity;
      current += (s.current_price ?? s.price) * s.quantity;
    });
    const pnl = current - invested;
    const pnlPct = invested > 0 ? (pnl / invested) * 100 : 0;
    return { invested, current, pnl, pnlPct, up: pnl >= 0 };
  }, [data]);

  const sorted = useMemo(() => {
    return [...data].sort((a, b) => {
      const valA = (a.current_price ?? a.price) * a.quantity;
      const valB = (b.current_price ?? b.price) * b.quantity;
      if (sortBy === "pnl") return (b.pnl_percent ?? 0) - (a.pnl_percent ?? 0);
      if (sortBy === "alloc") return valB - valA;
      return valB - valA;
    });
  }, [data, sortBy]);

  const allocationItems = useMemo(
    () =>
      [...data]
        .sort((a, b) => (b.current_price ?? b.price) * b.quantity - (a.current_price ?? a.price) * a.quantity)
        .map((s, i) => ({
          symbol: s.symbol,
          value: (s.current_price ?? s.price) * s.quantity,
          color: PALETTE[i % PALETTE.length],
        })),
    [data]
  );

  const bestPerformer = useMemo(
    () => data.length > 0 ? [...data].sort((a, b) => (b.pnl_percent ?? 0) - (a.pnl_percent ?? 0))[0] : null,
    [data]
  );

  return (
    <main className="min-h-screen bg-[#080810] text-white font-sans antialiased pb-24 selection:bg-blue-500/20">

      {/* Background */}
      <div className="fixed inset-0 -z-10 pointer-events-none overflow-hidden">
        <div className="absolute top-0 left-1/4 w-[600px] h-[600px] bg-blue-600/5 rounded-full blur-[120px]" />
        <div className="absolute bottom-0 right-1/4 w-[500px] h-[500px] bg-cyan-600/5 rounded-full blur-[100px]" />
        {/* Subtle grid */}
        <div
          className="absolute inset-0 opacity-[0.015]"
          style={{
            backgroundImage: "linear-gradient(rgba(255,255,255,0.5) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.5) 1px, transparent 1px)",
            backgroundSize: "40px 40px",
          }}
        />
      </div>

      <div className="max-w-5xl mx-auto px-4 md:px-6 pt-24">

        {/* ── HEADER ── */}
        <div className="flex items-start justify-between mb-10 gap-4">
          <div>
            <div className="flex items-center gap-2.5 mb-1">
              <div className="w-8 h-8 rounded-lg bg-blue-500/10 border border-blue-500/20 flex items-center justify-center">
                <PieChart size={16} className="text-blue-400" />
              </div>
              <h1 className="text-2xl font-bold tracking-tight text-white">Portfolio</h1>
            </div>
            <p className="text-sm text-gray-600 ml-[42px]">Real-time asset performance and exposure</p>
          </div>

          <div className="flex items-center gap-2 shrink-0">
            <button
              onClick={() => setHideValues((p) => !p)}
              className="w-9 h-9 rounded-xl bg-white/[0.04] hover:bg-white/[0.08] border border-white/[0.06] flex items-center justify-center text-gray-500 hover:text-gray-300 transition-all"
              title={hideValues ? "Show values" : "Hide values"}
            >
              {hideValues ? <EyeOff size={15} /> : <Eye size={15} />}
            </button>
            <button
              onClick={fetchPortfolio}
              className="w-9 h-9 rounded-xl bg-white/[0.04] hover:bg-white/[0.08] border border-white/[0.06] flex items-center justify-center text-gray-500 hover:text-gray-300 transition-all"
              title="Refresh"
            >
              <RefreshCcw size={15} className={loading ? "animate-spin" : ""} />
            </button>
            <Link
              href="/dashboard"
              className="flex items-center gap-1.5 px-4 py-2 rounded-xl bg-blue-600/90 hover:bg-blue-500 text-white text-sm font-semibold transition-all shadow-lg shadow-blue-500/20"
            >
              <Search size={14} /> Research
            </Link>
          </div>
        </div>

        {/* ── METRICS ── */}
        {!loading && !error && data.length > 0 && (
          <>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-4">
              <MetricCard
                icon={<Wallet size={13} />}
                label="Portfolio Value"
                value={hideValues ? "••••" : fmtCompact(metrics.current)}
                sub={hideValues ? undefined : `Invested ${fmtCompact(metrics.invested)}`}
                accent="blue"
                delay={0}
              />
              <MetricCard
                icon={<Activity size={13} />}
                label="Total Return"
                value={hideValues ? "••" : `${metrics.up ? "+" : ""}${metrics.pnlPct.toFixed(2)}%`}
                sub={hideValues ? undefined : `${metrics.up ? "+" : ""}${fmtINR(metrics.pnl)}`}
                accent={metrics.up ? "green" : "red"}
                delay={0.05}
              />
              <MetricCard
                icon={<Layers size={13} />}
                label="Holdings"
                value={`${data.length}`}
                sub="assets tracked"
                delay={0.1}
              />
              <MetricCard
                icon={<BarChart2 size={13} />}
                label="Best Performer"
                value={bestPerformer?.symbol ?? "—"}
                sub={
                  bestPerformer
                    ? `+${(bestPerformer.pnl_percent ?? 0).toFixed(2)}%`
                    : undefined
                }
                accent="green"
                delay={0.15}
              />
            </div>

            {/* Allocation bar */}
            <div className="bg-[#0d0d14] border border-white/[0.06] rounded-2xl px-5 py-4 mb-6">
              <div className="flex items-center justify-between mb-3">
                <span className="text-xs text-gray-600 uppercase tracking-widest font-medium">Allocation</span>
                <div className="flex items-center gap-3 flex-wrap justify-end">
                  {allocationItems.slice(0, 5).map((item) => (
                    <div key={item.symbol} className="flex items-center gap-1.5">
                      <div className="w-2 h-2 rounded-full" style={{ backgroundColor: item.color }} />
                      <span className="text-[10px] text-gray-500 font-medium">{item.symbol}</span>
                    </div>
                  ))}
                </div>
              </div>
              <AllocationBar items={allocationItems} />
            </div>
          </>
        )}

        {/* ── LOADING ── */}
        {loading && (
          <div className="space-y-3 mt-2">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-4">
              {[0, 1, 2, 3].map((i) => (
                <div key={i} className="h-24 bg-white/[0.03] rounded-2xl animate-pulse border border-white/[0.04]" />
              ))}
            </div>
            <Skeleton />
          </div>
        )}

        {/* ── ERROR ── */}
        {error && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}
            className="p-6 bg-rose-500/5 border border-rose-500/15 rounded-2xl text-center mt-4">
            <p className="text-rose-400 font-medium mb-4">{error}</p>
            <button
              onClick={fetchPortfolio}
              className="px-5 py-2 bg-rose-500/10 hover:bg-rose-500/20 text-rose-300 border border-rose-500/20 rounded-xl text-sm font-medium transition-all inline-flex items-center gap-2"
            >
              <RefreshCcw size={14} /> Try Again
            </button>
          </motion.div>
        )}

        {/* ── HOLDINGS ── */}
        {!loading && !error && data.length > 0 && (
          <div>
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-widest">Holdings</h2>
              <div className="flex items-center gap-1 bg-white/[0.03] border border-white/[0.06] rounded-xl p-1">
                {(["value", "pnl", "alloc"] as const).map((s) => (
                  <button
                    key={s}
                    onClick={() => setSortBy(s)}
                    className={`px-3 py-1 rounded-lg text-[11px] font-semibold transition-all ${
                      sortBy === s ? "bg-white/10 text-white" : "text-gray-600 hover:text-gray-400"
                    }`}
                  >
                    {s === "value" ? "Value" : s === "pnl" ? "Return" : "Weight"}
                  </button>
                ))}
              </div>
            </div>

            <div className="space-y-2.5">
              {sorted.map((stock, i) => (
                <HoldingRow
                  key={`${stock.symbol}-${i}`}
                  stock={stock}
                  idx={data.indexOf(stock)}
                  totalValue={metrics.current}
                  hideValues={hideValues}
                />
              ))}
            </div>

            {/* Footer note */}
            <div className="mt-6 flex items-center justify-between">
              <p className="text-[11px] text-gray-700">Click any holding to expand details</p>
              <Link href="/dashboard" className="text-[11px] text-blue-500 hover:text-blue-400 flex items-center gap-1 transition-colors">
                <Plus size={11} /> Add assets
              </Link>
            </div>
          </div>
        )}

        {/* ── EMPTY STATE ── */}
        {!loading && !error && data.length === 0 && (
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-center py-24 border border-white/[0.05] border-dashed rounded-2xl bg-white/[0.01] mt-4"
          >
            <div className="w-14 h-14 rounded-2xl bg-blue-500/10 border border-blue-500/15 flex items-center justify-center mx-auto mb-5">
              <Activity className="text-blue-400" size={26} />
            </div>
            <h3 className="text-lg font-bold mb-2">No holdings yet</h3>
            <p className="text-sm text-gray-500 max-w-sm mx-auto mb-7">
              Start tracking your investments. Add your first asset from the research dashboard.
            </p>
            <Link
              href="/dashboard"
              className="inline-flex items-center gap-2 bg-blue-600 hover:bg-blue-500 text-white px-6 py-2.5 rounded-xl text-sm font-semibold transition-all shadow-lg shadow-blue-500/20"
            >
              <Plus size={16} /> Add First Asset
            </Link>
          </motion.div>
        )}
      </div>
    </main>
  );
}