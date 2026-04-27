// app/page.tsx

"use client";

import { useRouter } from "next/navigation";
import { useState, useEffect, useRef, useCallback } from "react";
import useSWR from "swr";
import { motion, AnimatePresence, useReducedMotion } from "framer-motion";
import {
  Search, TrendingUp, TrendingDown, ArrowRight, Activity, Globe, Clock,
  Star, X, ChevronRight, AlertTriangle, RefreshCw, BarChart2, Zap,
  Shield, ArrowUp, ArrowDown, Eye, ExternalLink, Filter, Loader2,
  ChevronDown, Info, Award, LayoutGrid,
} from "lucide-react";

// ─────────────────────────────────────────────
// TYPES
// ─────────────────────────────────────────────
interface SearchResult { symbol: string; name?: string; exchange?: string; }
interface NewsItem { id?: string; title: string; text: string; link?: string; url?: string; source?: string; time?: string; sentiment?: "bullish" | "bearish" | "neutral"; thumbnail?: string; }
interface Ticker { symbol: string; price: string; change: number; }
interface TrendingStock { symbol: string; name: string; price: string; change: string; isUp: boolean; history: number[]; volume?: string; marketCap?: string; category: "stocks" | "crypto" | "indian" | "us"; }
interface Insight { label: string; value: string; color: string; bg: string; conf?: number; icon?: string; signal?: "buy" | "sell" | "hold"; risk?: "low" | "medium" | "high"; trend?: "up" | "down" | "flat"; tooltip?: string; }

// ─────────────────────────────────────────────
// MOCK DATA
// ─────────────────────────────────────────────
const LIVE_TICKER: Ticker[] = [
  { symbol: "SPY", price: "512.85", change: 1.2 },
  { symbol: "QQQ", price: "438.21", change: 1.8 },
  { symbol: "DIA", price: "390.11", change: -0.4 },
  { symbol: "BTC", price: "64,210", change: 5.4 },
  { symbol: "ETH", price: "3,450", change: 3.2 },
  { symbol: "AAPL", price: "173.50", change: 0.8 },
  { symbol: "NVDA", price: "880.00", change: 4.2 },
  { symbol: "MSFT", price: "415.30", change: 1.1 },
  { symbol: "TSLA", price: "175.22", change: -2.1 },
  { symbol: "NIFTY", price: "22,450", change: 0.6 },
];

const TRENDING_STOCKS: TrendingStock[] = [
  { symbol: "NVDA", name: "NVIDIA Corp", price: "$880.00", change: "+4.2%", isUp: true, history: [40,45,42,60,55,70,85,90], volume: "48.2M", marketCap: "$2.16T", category: "us" },
  { symbol: "TSLA", name: "Tesla Inc", price: "$175.22", change: "-2.1%", isUp: false, history: [80,75,78,65,60,55,60,50], volume: "92.1M", marketCap: "$557B", category: "us" },
  { symbol: "AAPL", name: "Apple Inc", price: "$173.50", change: "+0.8%", isUp: true, history: [50,52,51,55,54,58,60,62], volume: "55.3M", marketCap: "$2.71T", category: "us" },
  { symbol: "RELIANCE.NS", name: "Reliance Ind", price: "₹2,950", change: "+1.5%", isUp: true, history: [20,22,25,24,28,30,35,36], volume: "12.5M", marketCap: "₹19.9T", category: "indian" },
  { symbol: "BTC", name: "Bitcoin", price: "$64,210", change: "+5.4%", isUp: true, history: [30,35,32,45,50,60,72,80], volume: "$38B", marketCap: "$1.26T", category: "crypto" },
  { symbol: "ETH", name: "Ethereum", price: "$3,450", change: "+3.2%", isUp: true, history: [25,28,30,35,38,45,50,55], volume: "$18B", marketCap: "$414B", category: "crypto" },
  { symbol: "INFY.NS", name: "Infosys Ltd", price: "₹1,540", change: "+0.9%", isUp: true, history: [40,42,41,44,43,46,48,50], volume: "8.2M", marketCap: "₹6.4T", category: "indian" },
  { symbol: "MSFT", name: "Microsoft", price: "$415.30", change: "+1.1%", isUp: true, history: [55,57,56,60,62,65,68,70], volume: "22.4M", marketCap: "$3.08T", category: "stocks" },
];

const MOCK_NEWS: NewsItem[] = [
  { id: "1", title: "Global Markets Rally on Stellar Tech Earnings Beat", text: "Tech sector pushes indices to record highs as major players report Q3 earnings 15% above consensus estimates, driving significant institutional inflows.", link: "#", source: "Bloomberg", time: "2h ago", sentiment: "bullish" },
  { id: "2", title: "Federal Reserve Signals Potential Rate Cuts in H2 2024", text: "In a pivotal policy shift, the Fed indicated that cooling inflation data may pave the way for rate reductions later this year, sparking a bond market rally.", link: "#", source: "Reuters", time: "4h ago", sentiment: "bullish" },
  { id: "3", title: "AI Semiconductor Demand Hits All-Time High", text: "Enterprise AI adoption is accelerating globally, with chipmakers reporting record backlogs that could sustain elevated margins well into 2025.", link: "#", source: "WSJ", time: "5h ago", sentiment: "bullish" },
  { id: "4", title: "Oil Prices Dip on China Demand Uncertainty", text: "Brent crude slipped below $80 per barrel as weaker-than-expected Chinese PMI data raised concerns about future energy demand from the world's largest importer.", link: "#", source: "FT", time: "6h ago", sentiment: "bearish" },
  { id: "5", title: "Small-Cap Rotation Gains Steam Ahead of FOMC", text: "Institutional money is quietly rotating out of mega-cap tech into Russell 2000 constituents, a pattern historically observed before dovish Fed pivots.", link: "#", source: "MarketWatch", time: "8h ago", sentiment: "neutral" },
];

const MOCK_INSIGHTS: Insight[] = [
  { label: "Market Sentiment", value: "Bullish", color: "text-emerald-400", bg: "bg-emerald-400/10", conf: 85, icon: "📈", signal: "buy", risk: "medium", trend: "up", tooltip: "Based on options flow, put/call ratio, and institutional positioning." },
  { label: "Volatility Index", value: "Elevated", color: "text-amber-400", bg: "bg-amber-400/10", conf: 62, icon: "⚡", signal: "hold", risk: "high", trend: "up", tooltip: "VIX above 20 suggests elevated uncertainty. Expect wider price swings." },
  { label: "Tech Sector", value: "Overbought", color: "text-rose-400", bg: "bg-rose-400/10", conf: 75, icon: "🖥️", signal: "sell", risk: "high", trend: "flat", tooltip: "RSI above 70 on XLK ETF. Mean reversion likely in short term." },
  { label: "Small Cap Momentum", value: "Building", color: "text-blue-400", bg: "bg-blue-400/10", conf: 68, icon: "🚀", signal: "buy", risk: "medium", trend: "up", tooltip: "IWM showing accumulation patterns with above-average volume." },
];

const CATEGORY_FILTERS = ["All", "Stocks", "Crypto", "Indian", "US"] as const;
type CategoryFilter = typeof CATEGORY_FILTERS[number];

// ─────────────────────────────────────────────
// UTILS & HOOKS
// ─────────────────────────────────────────────
const fetcher = (url: string) => fetch(url).then(r => { if (!r.ok) throw new Error("Fetch failed"); return r.json(); });

function useDebounce<T>(value: T, delay: number): T {
  const [dv, setDv] = useState<T>(value);
  useEffect(() => { const t = setTimeout(() => setDv(value), delay); return () => clearTimeout(t); }, [value, delay]);
  return dv;
}

function checkIsIndianMarketOpen(): boolean {
  const now = new Date();
  const istTime = new Date(now.getTime() + now.getTimezoneOffset() * 60000 + 5.5 * 3600000);
  const day = istTime.getDay();
  if (day === 0 || day === 6) return false;
  const t = istTime.getHours() * 60 + istTime.getMinutes();
  return t >= 555 && t <= 930;
}

function useLocalStorage<T>(key: string, initial: T): [T, (v: T) => void] {
  const [val, setVal] = useState<T>(initial);
  useEffect(() => {
    try { const s = localStorage.getItem(key); if (s) setVal(JSON.parse(s)); } catch {}
  }, [key]);
  const set = useCallback((v: T) => { setVal(v); try { localStorage.setItem(key, JSON.stringify(v)); } catch {} }, [key]);
  return [val, set];
}

function useMarketCountdown(isOpen: boolean) {
  const [display, setDisplay] = useState("");
  useEffect(() => {
    const tick = () => {
      const now = new Date();
      const ist = new Date(now.getTime() + now.getTimezoneOffset() * 60000 + 5.5 * 3600000);
      const h = ist.getHours(), m = ist.getMinutes();
      if (isOpen) {
        const closeMin = 15 * 60 + 30 - (h * 60 + m);
        setDisplay(`Closes in ${Math.floor(closeMin / 60)}h ${closeMin % 60}m`);
      } else {
        const openMin = (9 * 60 + 15) - (h * 60 + m);
        const adj = openMin < 0 ? openMin + 24 * 60 : openMin;
        setDisplay(`Opens in ${Math.floor(adj / 60)}h ${adj % 60}m`);
      }
    };
    tick();
    const id = setInterval(tick, 60000);
    return () => clearInterval(id);
  }, [isOpen]);
  return display;
}

// ─────────────────────────────────────────────
// MICRO COMPONENTS
// ─────────────────────────────────────────────
const Skeleton = ({ className }: { className: string }) => (
  <div className={`relative overflow-hidden bg-white/[0.04] rounded-xl ${className}`}>
    <div className="absolute inset-0 -translate-x-full animate-[shimmer_1.5s_infinite] bg-gradient-to-r from-transparent via-white/[0.06] to-transparent" />
  </div>
);

const HighlightText = ({ text, highlight }: { text: string; highlight: string }) => {
  if (!highlight.trim() || !text) return <>{text}</>;
  const parts = text.split(new RegExp(`(${highlight.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")})`, "gi"));
  return <span>{parts.map((p, i) => p.toLowerCase() === highlight.toLowerCase() ? <span key={i} className="text-blue-400 font-bold">{p}</span> : <span key={i}>{p}</span>)}</span>;
};

const Sparkline = ({ data, color }: { data: number[]; color: string }) => {
  if (!data?.length) return null;
  const max = Math.max(...data), min = Math.min(...data), range = max - min || 1;
  const pts = data.map((d, i) => `${(i / (data.length - 1)) * 100},${100 - ((d - min) / range) * 100}`).join(" ");
  const areaPath = `M ${pts.split(" ").join(" L ")} L 100,100 L 0,100 Z`;
  return (
    <svg viewBox="0 0 100 100" className="w-20 h-10 overflow-visible" preserveAspectRatio="none">
      <defs>
        <linearGradient id={`g-${color.replace("#","")}`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity="0.3" />
          <stop offset="100%" stopColor={color} stopOpacity="0" />
        </linearGradient>
      </defs>
      <path d={areaPath} fill={`url(#g-${color.replace("#","")})`} />
      <polyline points={pts} fill="none" stroke={color} strokeWidth="4" vectorEffect="non-scaling-stroke" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
};

const SignalBadge = ({ signal }: { signal?: "buy" | "sell" | "hold" }) => {
  if (!signal) return null;
  const map = { buy: "bg-emerald-500/20 text-emerald-400 border-emerald-500/30", sell: "bg-rose-500/20 text-rose-400 border-rose-500/30", hold: "bg-amber-500/20 text-amber-400 border-amber-500/30" };
  return <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full border uppercase tracking-widest ${map[signal]}`}>{signal}</span>;
};

const RiskDots = ({ risk }: { risk?: "low" | "medium" | "high" }) => {
  if (!risk) return null;
  const levels = { low: 1, medium: 2, high: 3 };
  const colors = { low: "bg-emerald-400", medium: "bg-amber-400", high: "bg-rose-400" };
  const n = levels[risk];
  return (
    <div className="flex gap-0.5 items-center">
      {[1,2,3].map(i => <div key={i} className={`w-1.5 h-1.5 rounded-full ${i <= n ? colors[risk] : "bg-white/10"}`} />)}
    </div>
  );
};

const SentimentPill = ({ sentiment }: { sentiment?: string }) => {
  if (!sentiment) return null;
  const map: Record<string, string> = { bullish: "bg-emerald-500/15 text-emerald-400 border-emerald-500/25", bearish: "bg-rose-500/15 text-rose-400 border-rose-500/25", neutral: "bg-gray-500/15 text-gray-400 border-gray-500/25" };
  return <span className={`text-[10px] font-semibold px-2 py-0.5 rounded-full border uppercase tracking-wider ${map[sentiment] || map.neutral}`}>{sentiment}</span>;
};

const ErrorState = ({ onRetry }: { onRetry: () => void }) => (
  <div className="flex flex-col items-center justify-center py-10 gap-3 text-gray-500">
    <AlertTriangle size={24} className="text-amber-400" />
    <p className="text-sm">Failed to load data</p>
    <button onClick={onRetry} className="flex items-center gap-1.5 text-xs text-blue-400 hover:text-blue-300 transition-colors">
      <RefreshCw size={12} /> Retry
    </button>
  </div>
);

// ─────────────────────────────────────────────
// TICKER BAR
// ─────────────────────────────────────────────
const TickerBar = ({ data, isLoading, onSymbolClick }: { data: Ticker[]; isLoading: boolean; onSymbolClick: (s: string) => void }) => {
  const [paused, setPaused] = useState(false);
  const [hoveredTick, setHoveredTick] = useState<string | null>(null);

  return (
    <div
      className="w-full bg-[#080808] border-b border-white/[0.06] overflow-hidden flex whitespace-nowrap py-2.5 relative z-10"
      onMouseEnter={() => setPaused(true)}
      onMouseLeave={() => { setPaused(false); setHoveredTick(null); }}
    >
      <style>{`
        @keyframes ticker { 0% { transform: translateX(0); } 100% { transform: translateX(-50%); } }
        @keyframes shimmer { 100% { transform: translateX(100%); } }
        .animate-ticker { animation: ticker 40s linear infinite; }
        .ticker-paused { animation-play-state: paused !important; }
        .scrollbar-hide::-webkit-scrollbar { display: none; }
        .scrollbar-hide { -ms-overflow-style: none; scrollbar-width: none; }
      `}</style>
      {isLoading && !data.length ? (
        <div className="flex px-6 gap-6">{[...Array(7)].map((_, i) => <Skeleton key={i} className="w-28 h-4" />)}</div>
      ) : (
        <div className={`flex w-max animate-ticker ${paused ? "ticker-paused" : ""}`}>
          {[...data, ...data].map((tick, i) => (
            <button
              key={i}
              onClick={() => onSymbolClick(tick.symbol)}
              onMouseEnter={() => setHoveredTick(`${tick.symbol}-${i}`)}
              onMouseLeave={() => setHoveredTick(null)}
              className="group flex items-center gap-2 px-5 border-r border-white/[0.06] text-xs font-medium cursor-pointer hover:bg-white/[0.04] transition-colors py-1 rounded-sm"
            >
              <span className="text-gray-400 group-hover:text-white transition-colors">{tick.symbol}</span>
              <span className="text-white font-semibold">{tick.price}</span>
              <span className={`flex items-center gap-0.5 ${tick.change >= 0 ? "text-emerald-400" : "text-rose-400"}`}>
                {tick.change >= 0 ? <ArrowUp size={10} /> : <ArrowDown size={10} />}
                {Math.abs(tick.change)}%
              </span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
};

// ─────────────────────────────────────────────
// SEARCH BAR
// ─────────────────────────────────────────────
const SearchBar = ({ onSearch }: { onSearch: (sym: string) => void }) => {
  const [symbol, setSymbol] = useState("");
  const [activeIndex, setActiveIndex] = useState(-1);
  const [isFocused, setIsFocused] = useState(false);
  const [recentSearches, setRecentSearches] = useLocalStorage<string[]>("fin_recent_searches", []);
  const debouncedSymbol = useDebounce(symbol, 400);
  const inputRef = useRef<HTMLInputElement>(null);
  const dropdownRef = useRef<HTMLDivElement>(null);

  const API_BASE = process.env.NEXT_PUBLIC_API_URL!;
  const { data: searchResults, isValidating: isSearching } = useSWR<SearchResult[]>(
    debouncedSymbol ? `${API_BASE}/search/${encodeURIComponent(debouncedSymbol)}` : null,
    fetcher
  );

  const addRecentSearch = useCallback((sym: string) => {
    setRecentSearches([sym, ...recentSearches.filter(s => s !== sym)].slice(0, 6));
  }, [recentSearches, setRecentSearches]);

  const handleSearch = useCallback((sym?: string) => {
    const final = (sym || symbol).trim().toUpperCase();
    if (!final) return;
    addRecentSearch(final);
    setSymbol("");
    setIsFocused(false);
    onSearch(final);
  }, [symbol, onSearch, addRecentSearch]);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    const results = searchResults || [];
    if (e.key === "Escape") { setIsFocused(false); setSymbol(""); return; }
    if (!results.length) { if (e.key === "Enter") handleSearch(); return; }
    if (e.key === "ArrowDown") { e.preventDefault(); setActiveIndex(p => (p + 1) % results.length); }
    else if (e.key === "ArrowUp") { e.preventDefault(); setActiveIndex(p => p <= 0 ? results.length - 1 : p - 1); }
    else if (e.key === "Enter") { e.preventDefault(); handleSearch(activeIndex >= 0 ? results[activeIndex].symbol : undefined); }
  };

  const showDropdown = isFocused && (symbol.length > 0 || recentSearches.length > 0);

  return (
    <>
      {/* Overlay */}
      <AnimatePresence>
        {showDropdown && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/70 backdrop-blur-sm z-40" onClick={() => { setIsFocused(false); setSymbol(""); }} />
        )}
      </AnimatePresence>

      <div className="max-w-3xl mx-auto relative w-full z-50">
        <div className={`flex items-center gap-2 p-2 rounded-2xl bg-[#0e0e0e] border transition-all duration-300 shadow-2xl ${isFocused ? "border-blue-500/50 shadow-[0_0_50px_rgba(59,130,246,0.18)]" : "border-white/[0.08]"}`}>
          <div className="relative ml-3">
            {isSearching
              ? <Loader2 size={20} className="text-blue-400 animate-spin" />
              : <Search size={20} className={`transition-colors ${isFocused ? "text-blue-400" : "text-gray-500"}`} />}
          </div>
          <input
            ref={inputRef}
            value={symbol}
            onChange={e => { setSymbol(e.target.value); setActiveIndex(-1); }}
            onKeyDown={handleKeyDown}
            onFocus={() => setIsFocused(true)}
            placeholder="Search markets, stocks, or indices..."
            className="flex-1 bg-transparent px-3 py-3.5 outline-none text-white text-base placeholder-gray-600 font-medium"
            autoComplete="off"
          />
          {symbol && (
            <button onClick={() => { setSymbol(""); inputRef.current?.focus(); }} className="text-gray-500 hover:text-white transition-colors p-1">
              <X size={16} />
            </button>
          )}
          <button
            onClick={() => handleSearch()}
            className="px-6 py-3.5 rounded-xl bg-gradient-to-r from-blue-600 to-blue-500 font-semibold text-sm hover:opacity-90 active:scale-95 transition-all flex items-center gap-2 whitespace-nowrap"
          >
            <span>Analyze</span>
            <ArrowRight size={16} />
          </button>
        </div>

        {/* Dropdown */}
        <AnimatePresence>
          {showDropdown && (
            <motion.div
              ref={dropdownRef}
              initial={{ opacity: 0, y: -8, scale: 0.98 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: -8, scale: 0.98 }}
              transition={{ type: "spring", stiffness: 500, damping: 40 }}
              className="absolute w-full mt-3 bg-[#0e0e0e] border border-white/[0.08] rounded-2xl overflow-hidden shadow-[0_40px_80px_rgba(0,0,0,0.6)] max-h-[420px] overflow-y-auto scrollbar-hide z-50"
            >
              {/* Recent Searches */}
              {!symbol && recentSearches.length > 0 && (
                <div>
                  <div className="px-4 pt-4 pb-2 flex items-center justify-between">
                    <span className="text-xs text-gray-500 font-semibold uppercase tracking-wider">Recent</span>
                    <button onClick={() => setRecentSearches([])} className="text-xs text-gray-600 hover:text-gray-400 transition-colors">Clear</button>
                  </div>
                  {recentSearches.map((s, i) => (
                    <div key={i} onClick={() => handleSearch(s)}
                      className="flex items-center gap-3 px-4 py-3 hover:bg-white/[0.03] cursor-pointer group transition-colors">
                      <Clock size={14} className="text-gray-600 group-hover:text-gray-400 transition-colors" />
                      <span className="text-sm text-gray-300 group-hover:text-white transition-colors font-medium">{s}</span>
                      <ChevronRight size={14} className="text-gray-700 ml-auto" />
                    </div>
                  ))}
                  <div className="border-t border-white/[0.05] my-1" />
                </div>
              )}

              {/* Search Results */}
              {symbol && isSearching && (
                <div className="p-5 flex flex-col gap-3">
                  <Skeleton className="w-full h-12" />
                  <Skeleton className="w-3/4 h-12" />
                  <Skeleton className="w-1/2 h-12" />
                </div>
              )}
              {symbol && !isSearching && (!searchResults || searchResults.length === 0) && (
                <div className="p-8 text-gray-500 text-center text-sm">No assets found for &ldquo;{symbol}&rdquo;</div>
              )}
              {searchResults?.map((item, i) => (
                <div key={`${item.symbol}-${i}`} onClick={() => handleSearch(item.symbol)} onMouseEnter={() => setActiveIndex(i)}
                  className={`px-4 py-3.5 cursor-pointer flex items-center justify-between border-l-2 transition-all ${i === activeIndex ? "bg-white/[0.05] border-blue-500" : "border-transparent hover:bg-white/[0.02]"}`}>
                  <div className="flex items-center gap-3">
                    <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-white/[0.06] to-white/[0.02] border border-white/[0.08] flex items-center justify-center text-xs font-bold text-blue-400">
                      {item.symbol.slice(0, 2)}
                    </div>
                    <div className="flex flex-col">
                      <span className="font-bold text-sm"><HighlightText text={item.symbol} highlight={symbol} /></span>
                      <span className="text-xs text-gray-500"><HighlightText text={item.name || ""} highlight={symbol} /></span>
                    </div>
                  </div>
                  <span className="text-[10px] bg-white/[0.06] px-2 py-1 rounded-lg text-gray-400 font-medium">{item.exchange || "EQUITY"}</span>
                </div>
              ))}
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </>
  );
};

// ─────────────────────────────────────────────
// TRENDING ASSETS
// ─────────────────────────────────────────────
const TrendingAssets = ({ data, isLoading, onSymbolClick, watchlist, onToggleWatchlist }: {
  data: TrendingStock[]; isLoading: boolean; onSymbolClick: (s: string) => void;
  watchlist: string[]; onToggleWatchlist: (s: string, e: React.MouseEvent) => void;
}) => {
  const [filter, setFilter] = useState<CategoryFilter>("All");
  const [expanded, setExpanded] = useState<string | null>(null);
  const shouldReduce = useReducedMotion();

  const filtered = filter === "All" ? data : data.filter(s => s.category === filter.toLowerCase());

  return (
    <section className="mb-16">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between mb-6 gap-4">
        <div className="flex items-center gap-2">
          <Activity className="text-blue-400" size={20} />
          <h2 className="text-lg font-bold tracking-tight">Trending Assets</h2>
        </div>
        <div className="flex items-center gap-2 flex-wrap">
          <Filter size={14} className="text-gray-500" />
          {CATEGORY_FILTERS.map(cat => (
            <button key={cat} onClick={() => setFilter(cat)}
              className={`px-3 py-1.5 rounded-xl text-xs font-semibold transition-all ${filter === cat ? "bg-blue-500/20 text-blue-400 border border-blue-500/30" : "text-gray-500 hover:text-gray-300 border border-transparent hover:border-white/10"}`}>
              {cat}
            </button>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
        {isLoading && !data.length
          ? [...Array(4)].map((_, i) => <Skeleton key={i} className="h-40 w-full rounded-2xl" />)
          : filtered.map((stock, idx) => (
            <motion.div
              key={stock.symbol}
              layout
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: shouldReduce ? 0 : idx * 0.05 }}
              onClick={() => setExpanded(expanded === stock.symbol ? null : stock.symbol)}
              className="group relative bg-[#0e0e0e] hover:bg-[#141414] border border-white/[0.06] hover:border-white/[0.12] p-5 rounded-2xl cursor-pointer transition-all duration-300 hover:shadow-[0_8px_40px_rgba(0,0,0,0.5)]"
            >
              {/* Top gainer badge */}
              {idx === 0 && filter === "All" && (
                <div className="absolute -top-2.5 left-4 flex items-center gap-1 bg-gradient-to-r from-amber-500 to-yellow-400 text-black text-[10px] font-bold px-2.5 py-1 rounded-full">
                  <Award size={10} /> TOP GAINER
                </div>
              )}

              <button onClick={(e) => onToggleWatchlist(stock.symbol, e)}
                className="absolute top-4 right-4 z-10 transition-all active:scale-90">
                <Star size={15} className={watchlist.includes(stock.symbol) ? "fill-yellow-400 text-yellow-400" : "text-gray-600 hover:text-white"} />
              </button>

              <div className="flex justify-between items-start mb-4">
                <div>
                  <div className="flex items-center gap-2 mb-0.5">
                    <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-white/[0.08] to-white/[0.02] border border-white/[0.08] flex items-center justify-center text-[10px] font-bold text-blue-400">
                      {stock.symbol.slice(0, 2)}
                    </div>
                    <h3 className="font-bold text-base">{stock.symbol}</h3>
                  </div>
                  <p className="text-xs text-gray-500 truncate max-w-[110px] mt-1">{stock.name}</p>
                </div>
                <Sparkline data={stock.history} color={stock.isUp ? "#34d399" : "#fb7185"} />
              </div>

              <div className="flex justify-between items-end">
                <span className="text-xl font-bold tracking-tight">{stock.price}</span>
                <span className={`text-sm font-semibold flex items-center gap-1 ${stock.isUp ? "text-emerald-400" : "text-rose-400"}`}>
                  {stock.isUp ? <TrendingUp size={13} /> : <TrendingDown size={13} />}
                  {stock.change}
                </span>
              </div>

              {/* Expandable details */}
              <AnimatePresence>
                {expanded === stock.symbol && (
                  <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: "auto", opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    transition={{ duration: 0.25, ease: "easeInOut" }}
                    className="overflow-hidden"
                    onClick={e => e.stopPropagation()}
                  >
                    <div className="mt-4 pt-4 border-t border-white/[0.06] grid grid-cols-2 gap-2 text-xs">
                      <div className="flex flex-col gap-0.5">
                        <span className="text-gray-500">Volume</span>
                        <span className="text-white font-semibold">{stock.volume || "—"}</span>
                      </div>
                      <div className="flex flex-col gap-0.5">
                        <span className="text-gray-500">Mkt Cap</span>
                        <span className="text-white font-semibold">{stock.marketCap || "—"}</span>
                      </div>
                      <button
                        onClick={(e) => { e.stopPropagation(); onSymbolClick(stock.symbol); }}
                        className="col-span-2 mt-2 flex items-center justify-center gap-1.5 py-2.5 rounded-xl bg-blue-500/15 text-blue-400 hover:bg-blue-500/25 font-semibold text-xs transition-colors border border-blue-500/20"
                      >
                        <BarChart2 size={13} /> Full Analysis
                      </button>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>

              <div className="absolute bottom-4 right-4">
                <ChevronDown size={14} className={`text-gray-600 transition-transform duration-200 ${expanded === stock.symbol ? "rotate-180" : ""}`} />
              </div>
            </motion.div>
          ))}
      </div>
    </section>
  );
};

// ─────────────────────────────────────────────
// NEWS FEED
// ─────────────────────────────────────────────
const NewsFeed = ({ data, isLoading, error, onRetry }: { data: NewsItem[]; isLoading: boolean; error?: boolean; onRetry: () => void }) => {
  const [selectedNews, setSelectedNews] = useState<NewsItem | null>(null);
  const [featured, ...rest] = data;

  return (
    <div className="md:col-span-2 bg-[#0a0a0a] border border-white/[0.06] p-6 rounded-3xl flex flex-col">
      <div className="flex items-center justify-between mb-5">
        <h2 className="text-lg font-bold flex items-center gap-2"><Globe className="text-blue-400" size={18} /> Live Market Feed</h2>
        <span className="text-[10px] font-bold bg-red-500/20 text-red-400 border border-red-500/30 px-3 py-1 rounded-full flex items-center gap-1.5 uppercase tracking-wider">
          <span className="w-1.5 h-1.5 rounded-full bg-red-400 animate-pulse" /> Live
        </span>
      </div>

      {error ? <ErrorState onRetry={onRetry} /> : isLoading && !data.length ? (
        <div className="flex flex-col gap-3">{[...Array(3)].map((_, i) => <Skeleton key={i} className="h-24 w-full rounded-xl" />)}</div>
      ) : (
        <div className="flex flex-col gap-3 overflow-y-auto scrollbar-hide max-h-[420px] pr-1">
          {/* Featured top story */}
          {featured && (
            <button onClick={() => setSelectedNews(featured)}
              className="group text-left bg-gradient-to-br from-blue-900/20 to-blue-800/10 border border-blue-500/20 p-5 rounded-2xl hover:border-blue-500/40 transition-all">
              <div className="flex items-center gap-2 mb-3">
                <span className="text-[10px] font-bold bg-blue-500/20 text-blue-400 px-2 py-0.5 rounded-full uppercase tracking-wider border border-blue-500/20">Featured</span>
                <SentimentPill sentiment={featured.sentiment} />
                <span className="text-[10px] text-gray-500 ml-auto flex items-center gap-1"><Clock size={10} />{featured.time}</span>
              </div>
              <h3 className="font-bold text-base leading-snug group-hover:text-blue-300 transition-colors mb-2 line-clamp-2">{featured.title}</h3>
              <p className="text-xs text-gray-400 line-clamp-2 leading-relaxed">{featured.text}</p>
              <div className="flex items-center gap-1.5 mt-3 text-xs text-blue-400 font-medium">
                <Eye size={12} /> Read more
              </div>
            </button>
          )}

          {rest.map((n, i) => (
            <button key={i} onClick={() => setSelectedNews(n)}
              className="group text-left bg-white/[0.02] border border-white/[0.05] p-4 rounded-2xl hover:bg-white/[0.04] hover:border-white/10 transition-all">
              <div className="flex items-center gap-2 mb-2">
                <span className="text-[10px] font-semibold text-blue-400">{n.source}</span>
                <span className="text-gray-600">·</span>
                <SentimentPill sentiment={n.sentiment} />
                <span className="text-[10px] text-gray-600 ml-auto flex items-center gap-1"><Clock size={10} />{n.time}</span>
              </div>
              <h3 className="font-semibold text-sm leading-snug group-hover:text-blue-300 transition-colors line-clamp-1 mb-1">{n.title}</h3>
              <p className="text-xs text-gray-500 line-clamp-2 leading-relaxed">{n.text}</p>
            </button>
          ))}
        </div>
      )}

      {/* News Modal */}
      <AnimatePresence>
        {selectedNews && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/80 backdrop-blur-md z-50 flex items-center justify-center p-4"
            onClick={() => setSelectedNews(null)}>
            <motion.div initial={{ scale: 0.95, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} exit={{ scale: 0.95, opacity: 0 }}
              transition={{ type: "spring", stiffness: 400, damping: 35 }}
              className="bg-[#0e0e0e] border border-white/10 rounded-3xl p-8 max-w-2xl w-full shadow-[0_40px_100px_rgba(0,0,0,0.8)]"
              onClick={e => e.stopPropagation()}>
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2">
                  <span className="text-xs font-semibold text-blue-400">{selectedNews.source}</span>
                  <SentimentPill sentiment={selectedNews.sentiment} />
                </div>
                <button onClick={() => setSelectedNews(null)} className="text-gray-500 hover:text-white transition-colors"><X size={20} /></button>
              </div>
              <h2 className="text-xl font-bold leading-tight mb-4">{selectedNews.title}</h2>
              <p className="text-gray-300 leading-relaxed text-sm mb-6">{selectedNews.text}</p>
              <a href={selectedNews.link || selectedNews.url} target="_blank" rel="noopener noreferrer"
                className="inline-flex items-center gap-2 px-5 py-3 rounded-xl bg-blue-600 hover:bg-blue-500 text-white font-semibold text-sm transition-colors">
                <ExternalLink size={14} /> Read Full Article
              </a>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

// ─────────────────────────────────────────────
// INSIGHTS PANEL
// ─────────────────────────────────────────────
const InsightsPanel = ({ data, isLoading }: { data: Insight[]; isLoading: boolean }) => {
  const [hoveredTooltip, setHoveredTooltip] = useState<number | null>(null);

  const trendIcon = (t?: string) => t === "up" ? <ArrowUp size={12} className="text-emerald-400" /> : t === "down" ? <ArrowDown size={12} className="text-rose-400" /> : <span className="text-xs text-gray-500">—</span>;

  return (
    <div className="bg-gradient-to-b from-[#0e0e0e] to-[#090909] border border-white/[0.06] p-6 rounded-3xl relative overflow-hidden flex flex-col">
      <div className="absolute top-0 right-0 w-40 h-40 bg-blue-600/[0.07] blur-[60px] rounded-full pointer-events-none" />
      <div className="absolute bottom-0 left-0 w-32 h-32 bg-purple-600/[0.05] blur-[50px] rounded-full pointer-events-none" />

      <div className="flex items-center gap-2 mb-6">
        <Zap size={18} className="text-amber-400" />
        <h2 className="text-lg font-bold">AI Macro Insights</h2>
      </div>

      {isLoading ? <Skeleton className="h-56 w-full rounded-xl" /> : (
        <div className="flex flex-col gap-3 flex-1">
          {data.map((insight, i) => (
            <div key={i} className="relative p-4 rounded-2xl bg-white/[0.02] border border-white/[0.05] hover:border-white/10 transition-all group">
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  <span className="text-base">{insight.icon}</span>
                  <span className="text-xs text-gray-400 font-medium">{insight.label}</span>
                  <div className="relative">
                    <button
                      onMouseEnter={() => setHoveredTooltip(i)}
                      onMouseLeave={() => setHoveredTooltip(null)}
                      className="text-gray-600 hover:text-gray-400 transition-colors"
                    >
                      <Info size={11} />
                    </button>
                    <AnimatePresence>
                      {hoveredTooltip === i && (
                        <motion.div initial={{ opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: 4 }}
                          className="absolute left-0 bottom-6 z-50 w-52 bg-[#1a1a1a] border border-white/10 rounded-xl p-3 text-xs text-gray-300 shadow-2xl leading-relaxed pointer-events-none">
                          {insight.tooltip}
                        </motion.div>
                      )}
                    </AnimatePresence>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  {trendIcon(insight.trend)}
                  <RiskDots risk={insight.risk} />
                  <SignalBadge signal={insight.signal} />
                </div>
              </div>

              <div className="flex items-center justify-between mb-2">
                <span className={`text-sm font-bold ${insight.color}`}>{insight.value}</span>
                {insight.conf && <span className="text-[10px] text-gray-600 font-medium">{insight.conf}% conf.</span>}
              </div>

              {insight.conf && (
                <div className="w-full h-1 bg-black/60 rounded-full overflow-hidden">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${insight.conf}%` }}
                    transition={{ duration: 1, ease: "easeOut", delay: i * 0.15 }}
                    className={`h-full rounded-full ${insight.color.replace("text-", "bg-").replace("400", "500")}`}
                  />
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      <div className="mt-5 pt-5 border-t border-white/[0.05]">
        <div className="flex items-start gap-2">
          <Shield size={14} className="text-blue-400 mt-0.5 flex-shrink-0" />
          <p className="text-xs text-gray-500 leading-relaxed">
            <span className="text-blue-400 font-semibold">Auto-Summary:</span>{" "}
            Rotation from tech into small-caps ahead of FOMC. Monitor IWM volume spikes and watch VIX levels for volatility signals.
          </p>
        </div>
      </div>
    </div>
  );
};

// ─────────────────────────────────────────────
// WATCHLIST PANEL
// ─────────────────────────────────────────────
const WatchlistPanel = ({ watchlist, onSymbolClick, onRemove }: { watchlist: string[]; onSymbolClick: (s: string) => void; onRemove: (s: string) => void }) => {
  if (!watchlist.length) return null;
  return (
    <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: "auto" }} className="mb-16">
      <div className="flex items-center gap-2 mb-4">
        <Star size={16} className="text-yellow-400 fill-yellow-400" />
        <h2 className="text-base font-bold">Your Watchlist</h2>
        <span className="text-xs bg-white/[0.06] px-2 py-0.5 rounded-full text-gray-400">{watchlist.length}</span>
      </div>
      <div className="flex gap-2 flex-wrap">
        {watchlist.map(sym => (
          <div key={sym} className="flex items-center gap-2 bg-white/[0.04] border border-white/[0.08] hover:border-white/[0.15] rounded-xl px-3 py-2 transition-all group">
            <button onClick={() => onSymbolClick(sym)} className="text-sm font-bold text-white hover:text-blue-400 transition-colors">{sym}</button>
            <button onClick={() => onRemove(sym)} className="text-gray-600 hover:text-white transition-colors"><X size={12} /></button>
          </div>
        ))}
      </div>
    </motion.div>
  );
};

// ─────────────────────────────────────────────
// MARKET SENTIMENT BAR
// ─────────────────────────────────────────────
const MarketSentimentBar = ({ isOpen, countdown }: { isOpen: boolean; countdown: string }) => (
  <div className="flex flex-col sm:flex-row items-center justify-between gap-3 mb-10 bg-white/[0.02] border border-white/[0.05] rounded-2xl px-5 py-3">
    <div className="flex items-center gap-3 flex-wrap justify-center sm:justify-start">
      <div className="flex items-center gap-2 text-sm">
        <span className={`w-2 h-2 rounded-full ${isOpen ? "bg-emerald-400 animate-pulse" : "bg-gray-600"}`} />
        <span className={`font-semibold ${isOpen ? "text-emerald-400" : "text-gray-400"}`}>NSE/BSE {isOpen ? "Open" : "Closed"}</span>
      </div>
      <span className="text-gray-700 hidden sm:block">·</span>
      <div className="flex items-center gap-1.5 text-xs text-gray-500">
        <Clock size={12} />
        <span>{countdown}</span>
      </div>
    </div>
    <div className="flex items-center gap-3 text-xs text-gray-500">
      <span className="flex items-center gap-1"><LayoutGrid size={12} /> Overall Sentiment:</span>
      <div className="flex items-center gap-0.5 h-2 w-32 rounded-full overflow-hidden bg-white/[0.05]">
        <div className="h-full bg-emerald-500 rounded-l-full" style={{ width: "62%" }} />
        <div className="h-full bg-amber-500" style={{ width: "20%" }} />
        <div className="h-full bg-rose-500 rounded-r-full" style={{ width: "18%" }} />
      </div>
      <span className="text-emerald-400 font-semibold">62% Bullish</span>
    </div>
  </div>
);

// ─────────────────────────────────────────────
// MAIN PAGE
// ─────────────────────────────────────────────
export default function Home() {
  const router = useRouter();
  const [marketOpen, setMarketOpen] = useState(false);
  const [watchlist, setWatchlist] = useLocalStorage<string[]>("fin_watchlist", []);
  const countdown = useMarketCountdown(marketOpen);

  const API_BASE = process.env.NEXT_PUBLIC_API_URL!;

  useEffect(() => {
    setMarketOpen(checkIsIndianMarketOpen());
    const id = setInterval(() => setMarketOpen(checkIsIndianMarketOpen()), 60000);
    return () => clearInterval(id);
  }, []);

  const { data: marketData, isLoading: isMarketLoading, error: marketError, mutate: retryMarket } = useSWR(
    `${API_BASE}/market`, fetcher, { refreshInterval: marketOpen ? 10000 : 0, fallbackData: null }
  );
  const { data: newsData, isLoading: isNewsLoading, error: newsError, mutate: retryNews } = useSWR(
    `${API_BASE}/news/AAPL`, fetcher, { refreshInterval: 300000 }
  );

  const activeTickerData: Ticker[] = marketData?.ticker?.length ? marketData.ticker : LIVE_TICKER;
  const activeTrendingData: TrendingStock[] = marketData?.trending?.length ? marketData.trending : TRENDING_STOCKS;
  const activeNewsData: NewsItem[] = newsData?.news?.length ? newsData.news : MOCK_NEWS;
  const activeInsightsData: Insight[] = marketData?.insights?.length ? marketData.insights : MOCK_INSIGHTS;

  const handleSearch = useCallback((sym: string) => {
    if (!sym.trim()) return;
    router.push(`/dashboard?symbol=${sym.trim().toUpperCase()}`);
  }, [router]);

  const toggleWatchlist = useCallback((sym: string, e: React.MouseEvent) => {
    e.stopPropagation();
    setWatchlist(watchlist.includes(sym) ? watchlist.filter(s => s !== sym) : [...watchlist, sym]);
  }, [watchlist, setWatchlist]);

  const removeFromWatchlist = useCallback((sym: string) => {
    setWatchlist(watchlist.filter(s => s !== sym));
  }, [watchlist, setWatchlist]);

  return (
    <main className="relative min-h-screen bg-[#060606] text-white overflow-hidden font-sans">
      <style>{`
        @keyframes ticker { 0% { transform: translateX(0); } 100% { transform: translateX(-50%); } }
        @keyframes shimmer { 0% { transform: translateX(-100%); } 100% { transform: translateX(200%); } }
        .scrollbar-hide::-webkit-scrollbar { display: none; }
        .scrollbar-hide { -ms-overflow-style: none; scrollbar-width: none; }
      `}</style>

      {/* Background */}
      <div className="absolute inset-0 -z-10 pointer-events-none overflow-hidden">
        <div className="absolute top-[-15%] left-[-10%] w-[55vw] h-[55vw] rounded-full bg-blue-950/30 blur-[140px]" />
        <div className="absolute bottom-[-10%] right-[-10%] w-[40vw] h-[40vw] rounded-full bg-indigo-950/20 blur-[120px]" />
        <div className="absolute top-[40%] right-[20%] w-[20vw] h-[20vw] rounded-full bg-blue-900/10 blur-[80px]" />
      </div>

      {/* Ticker */}
      <TickerBar data={activeTickerData} isLoading={isMarketLoading} onSymbolClick={handleSearch} />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 py-12 lg:py-20">

        {/* Hero */}
        <div className="text-center mb-16 relative z-50">
          <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}
            className="inline-flex items-center gap-2 bg-blue-500/10 text-blue-400 px-4 py-2 rounded-full text-xs font-semibold border border-blue-500/20 mb-6 tracking-wide">
            <Activity size={14} />
            Market Intelligence System · Online
          </motion.div>

          <motion.h1 initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.08 }}
            className="text-5xl md:text-7xl font-black tracking-tight mb-4 leading-[0.95]">
            AI Financial{" "}
            <span className="bg-gradient-to-r from-blue-400 via-cyan-300 to-blue-500 bg-clip-text text-transparent">
              Intelligence
            </span>
          </motion.h1>

          <motion.p initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.15 }}
            className="text-gray-500 text-base mb-10 max-w-xl mx-auto">
            Real-time market data, AI-driven insights, and actionable signals — all in one place.
          </motion.p>

          <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }}>
            <SearchBar onSearch={handleSearch} />
          </motion.div>
        </div>

        {/* Market sentiment + countdown bar */}
        <MarketSentimentBar isOpen={marketOpen} countdown={countdown} />

        {/* Watchlist */}
        <WatchlistPanel watchlist={watchlist} onSymbolClick={handleSearch} onRemove={removeFromWatchlist} />

        {/* Trending */}
        <TrendingAssets
          data={activeTrendingData}
          isLoading={isMarketLoading}
          onSymbolClick={handleSearch}
          watchlist={watchlist}
          onToggleWatchlist={toggleWatchlist}
        />

        {/* News + Insights */}
        <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.3 }}
          className="grid md:grid-cols-3 gap-5">
          <NewsFeed data={activeNewsData} isLoading={isNewsLoading} error={!!newsError} onRetry={() => retryNews()} />
          <InsightsPanel data={activeInsightsData} isLoading={isMarketLoading} />
        </motion.div>

      </div>
    </main>
  );
}