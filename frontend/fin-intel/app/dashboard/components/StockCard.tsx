"use client";

import { useQuery } from "@tanstack/react-query";
import { motion } from "framer-motion";
import Chart from "./Chart";
import Recommendation from "./Recommendation";
import Sentiment from "./Sentiment";
import RiskMeter from "./RiskMeter";
import Fundamentals from "./Fundamentals";
import News from "./News";

type StockData = {
  symbol: string;
  prices: { date: string; close: number }[];
  final_score: number;
  sentiment_score: number;
  risk_score: number;
  technical_score: number;
  fundamental_score: number;
  fundamentals: {
    roe: number;
    debt_equity: number;
    revenue_growth: number;
    profit_margin?: number;
  };
  explanation: string;
  cached?: boolean; // Added from your backend update
};

export default function StockCard({ symbol }: { symbol: string }) {
  
  // 🔥 Fetch stock data using React Query
  const { 
    data: stockData, 
    isLoading: isStockLoading, 
    isError: isStockError 
  } = useQuery<StockData>({
    queryKey: ["stock", symbol],
    queryFn: async () => {
      const baseUrl = process.env.NEXT_PUBLIC_API_URL!;
      const res = await fetch(`${baseUrl}/stock/${symbol}`);
      if (!res.ok) throw new Error("Failed to fetch stock");
      return res.json();
    },
    // staleTime is handled globally in providers.tsx, but can be overridden here if needed
  });

  // 📰 Fetch news using React Query
  const { data: newsData = [] } = useQuery({
    queryKey: ["news", symbol],
    queryFn: async () => {
      const baseUrl = process.env.NEXT_PUBLIC_API_URL!;
      const res = await fetch(`${baseUrl}/news/${symbol}`);
      if (!res.ok) throw new Error("Failed to fetch news");
      const json = await res.json();
      return json.news || [];
    },
  });

  // ➕ Add to portfolio
  const addToPortfolio = async () => {
    try {
      const baseUrl =process.env.NEXT_PUBLIC_API_URL!;
      await fetch(`${baseUrl}/portfolio/add`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          symbol,
          quantity: 1,
          price: stockData?.prices?.at(-1)?.close || 0,
        }),
      });

      alert("Added to portfolio ✅");
    } catch (err) {
      console.error("Portfolio error:", err);
    }
  };

  // 🔄 LOADING STATE
  if (isStockLoading) {
    return (
      <div className="p-8 rounded-2xl bg-white/5 border border-white/10 backdrop-blur-xl animate-pulse">
        <p className="text-lg font-semibold text-white">
          Analyzing {symbol}...
        </p>
        <p className="text-sm text-gray-400 mt-2">
          Running AI models...
        </p>
      </div>
    );
  }

  // ❌ ERROR STATE
  if (isStockError || !stockData) {
    return (
      <div className="p-6 bg-red-500/10 border border-red-500/30 rounded-xl text-red-400">
        Failed to load analysis for {symbol}.
      </div>
    );
  }

  return (
    <div className="space-y-6">

      {/* 🧠 TOP */}
      <div className="grid md:grid-cols-3 gap-6 relative">
        {/* Render a subtle badge if the data was served from the backend cache */}
        {stockData.cached && (
           <span className="absolute -top-3 -right-2 text-[10px] bg-blue-500/20 text-blue-400 px-2 py-1 rounded-full border border-blue-500/30">
             ⚡ Cached
           </span>
        )}
        <Recommendation score={stockData.final_score} />
        <Sentiment score={stockData.sentiment_score} />
        <RiskMeter risk={stockData.risk_score} />
      </div>

      {/* 📊 MAIN */}
      <div className="grid lg:grid-cols-3 gap-6">

        {/* LEFT */}
        <div className="lg:col-span-2 space-y-6">
          {/* Chart */}
          <Chart data={stockData.prices} />

          {/* 📰 News */}
          <News news={newsData} sentiment={stockData} />
        </div>

        {/* RIGHT */}
        <div className="space-y-6">

          {/* ➕ Portfolio Button */}
          <button
            onClick={addToPortfolio}
            className="w-full bg-blue-600 hover:bg-blue-700 py-3 rounded-xl font-semibold transition-colors shadow-lg shadow-blue-500/20"
          >
            ➕ Add to Portfolio
          </button>

          {/* Scores */}
          <div className="bg-white/5 border border-white/10 backdrop-blur-xl p-5 rounded-2xl shadow-lg">
            <h3 className="text-sm text-gray-400 mb-4">
              Score Breakdown
            </h3>

            <div className="space-y-3">
              {[
                { label: "Technical", value: stockData.technical_score },
                { label: "Fundamental", value: stockData.fundamental_score },
                { label: "Final", value: stockData.final_score },
              ].map((item, i) => (
                <div key={i}>
                  <div className="flex justify-between text-sm mb-1">
                    <span className="text-gray-400">{item.label}</span>
                    <span className="text-white font-medium">
                      {(item.value * 100).toFixed(0)}%
                    </span>
                  </div>

                  <div className="w-full bg-white/10 h-2 rounded-full overflow-hidden">
                    <motion.div
                      initial={{ width: 0 }}
                      animate={{ width: `${item.value * 100}%` }}
                      transition={{ duration: 1, ease: "easeOut" }}
                      className="bg-blue-500 h-2 rounded-full"
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* 🧠 Explanation */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="bg-white/5 border border-white/10 backdrop-blur-xl p-5 rounded-2xl shadow-lg"
          >
            <h3 className="text-sm text-gray-400 mb-3">
              AI Explanation
            </h3>

            <div className="space-y-2 text-sm text-gray-300 leading-relaxed">
              {stockData.explanation
                ? stockData.explanation.split("\n").map((line, i) => (
                    <p key={i}>{line}</p>
                  ))
                : "No explanation available."}
            </div>
          </motion.div>

        </div>
      </div>

      {/* 📉 Fundamentals */}
      <Fundamentals data={stockData.fundamentals} />

    </div>
  );
}