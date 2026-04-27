"use client";

import { motion } from "framer-motion";

type FundamentalsData = {
  roe?: number | null;
  debt_equity?: number | null;
  revenue_growth?: number | null;
  profit_margin?: number | null;
  market_cap?: number | null;
  pe_ratio?: number | null;
  eps?: number | null;
  "52w_high"?: number | null;
  "52w_low"?: number | null;
  avg_volume?: number | null;
  dividend_yield?: number | null;
  beta?: number | null;
};

type MetricItem = {
  label: string;
  value: string;
  colorClass: string;
  description: string;
};

function getColorClass(value: number, type: "roe" | "debt" | "growth" | "margin" | "neutral"): string {
  if (type === "debt") return value < 1 ? "text-emerald-400" : value < 2 ? "text-amber-400" : "text-rose-400";
  if (type === "neutral") return "text-white";
  return value > 15 ? "text-emerald-400" : value > 5 ? "text-amber-400" : value > 0 ? "text-gray-300" : "text-rose-400";
}

function fmtVal(n: number | null | undefined, suffix = "", prefix = "", decimals = 2): string {
  if (n == null || isNaN(n)) return "N/A";
  return `${prefix}${n.toFixed(decimals)}${suffix}`;
}

function fmtLarge(n: number | null | undefined, prefix = ""): string {
  if (n == null || isNaN(n)) return "N/A";
  if (Math.abs(n) >= 1e12) return `${prefix}${(n / 1e12).toFixed(2)}T`;
  if (Math.abs(n) >= 1e9) return `${prefix}${(n / 1e9).toFixed(2)}B`;
  if (Math.abs(n) >= 1e6) return `${prefix}${(n / 1e6).toFixed(2)}M`;
  return `${prefix}${n.toLocaleString()}`;
}

export default function Fundamentals({ data }: { data: FundamentalsData }) {
  const hasAny = data && Object.values(data).some(v => v != null);
  if (!hasAny) {
    return (
      <div className="bg-white/[0.03] border border-white/[0.06] p-6 rounded-2xl text-center text-gray-500 text-sm">
        No fundamental data available
      </div>
    );
  }

  const metrics: MetricItem[] = [
    {
      label: "Return on Equity",
      value: fmtVal(data.roe, "%"),
      colorClass: data.roe != null ? getColorClass(data.roe, "roe") : "text-gray-500",
      description: "Profit generated per dollar of equity. Higher is better.",
    },
    {
      label: "Debt / Equity",
      value: fmtVal(data.debt_equity, "x"),
      colorClass: data.debt_equity != null ? getColorClass(data.debt_equity, "debt") : "text-gray-500",
      description: "How much debt is used to finance assets. Lower is safer.",
    },
    {
      label: "Revenue Growth",
      value: fmtVal(data.revenue_growth, "%"),
      colorClass: data.revenue_growth != null ? getColorClass(data.revenue_growth, "growth") : "text-gray-500",
      description: "Year-over-year revenue growth rate.",
    },
    {
      label: "Profit Margin",
      value: fmtVal(data.profit_margin, "%"),
      colorClass: data.profit_margin != null ? getColorClass(data.profit_margin, "margin") : "text-gray-500",
      description: "Net income as a percentage of revenue.",
    },
    {
      label: "P/E Ratio",
      value: fmtVal(data.pe_ratio, "x"),
      colorClass: data.pe_ratio != null ? (data.pe_ratio < 20 ? "text-emerald-400" : data.pe_ratio < 35 ? "text-amber-400" : "text-rose-400") : "text-gray-500",
      description: "Price relative to earnings. Lower may indicate undervaluation.",
    },
    {
      label: "EPS",
      value: fmtVal(data.eps, "", "$"),
      colorClass: data.eps != null ? (data.eps > 0 ? "text-emerald-400" : "text-rose-400") : "text-gray-500",
      description: "Earnings per share (trailing 12 months).",
    },
    {
      label: "Beta",
      value: fmtVal(data.beta),
      colorClass: data.beta != null ? (data.beta < 0.8 ? "text-emerald-400" : data.beta < 1.5 ? "text-amber-400" : "text-rose-400") : "text-gray-500",
      description: "Volatility vs. the market. >1 means more volatile.",
    },
    {
      label: "Dividend Yield",
      value: data.dividend_yield != null ? fmtVal(data.dividend_yield * 100, "%") : "N/A",
      colorClass: data.dividend_yield != null && data.dividend_yield > 0 ? "text-emerald-400" : "text-gray-500",
      description: "Annual dividends paid as a percentage of current price.",
    },
  ];

  const tableMetrics: MetricItem[] = [
    { label: "Market Cap",   value: fmtLarge(data.market_cap, "$"),   colorClass: "text-white", description: "Total market value of all outstanding shares." },
    { label: "Avg Volume",   value: fmtLarge(data.avg_volume),         colorClass: "text-white", description: "Average daily trading volume over 3 months." },
    { label: "52W High",     value: fmtVal(data["52w_high"], "", "$"),  colorClass: "text-emerald-400", description: "Highest price in the last 52 weeks." },
    { label: "52W Low",      value: fmtVal(data["52w_low"], "", "$"),   colorClass: "text-rose-400",    description: "Lowest price in the last 52 weeks." },
  ];

  return (
    <div className="bg-[#0d0d0d] border border-white/[0.06] p-6 rounded-2xl">
      <h2 className="text-xs font-bold text-gray-500 uppercase tracking-wider mb-5">Fundamentals</h2>

      {/* Core metrics grid */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-5">
        {metrics.map((item, i) => (
          <motion.div key={item.label}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.04 }}
            title={item.description}
            className="group p-4 rounded-xl bg-white/[0.02] border border-white/[0.05] hover:bg-white/[0.05] hover:border-white/[0.09] transition-all cursor-help"
          >
            <p className="text-[10px] text-gray-500 font-semibold uppercase tracking-wider mb-1.5 truncate">{item.label}</p>
            <p className={`text-lg font-bold ${item.colorClass}`}>{item.value}</p>
            <p className="text-[10px] text-gray-600 mt-1 leading-snug hidden group-hover:block">{item.description}</p>
          </motion.div>
        ))}
      </div>

      {/* Secondary stats row */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-4 border-t border-white/[0.04]">
        {tableMetrics.map((item, i) => (
          <div key={item.label} title={item.description}
            className="flex justify-between items-center sm:flex-col sm:items-start gap-1 p-3 rounded-xl bg-white/[0.02] border border-white/[0.04] cursor-help">
            <span className="text-[10px] text-gray-600 font-semibold uppercase tracking-wider">{item.label}</span>
            <span className={`text-sm font-bold ${item.colorClass}`}>{item.value}</span>
          </div>
        ))}
      </div>
    </div>
  );
}