"use client";

import { useState } from "react";
import { ExternalLink, Clock, ChevronDown } from "lucide-react";

interface NewsItem {
  id?: string | number;
  title: string;
  text?: string;
  url?: string;
  link?: string;
  source?: string;
  time?: string;
  sentiment?: string;
  thumbnail?: string;
}

const SentimentPill = ({ sentiment }: { sentiment?: string }) => {
  if (!sentiment) return null;
  const s = sentiment.toLowerCase();
  const map: Record<string, string> = {
    bullish: "bg-emerald-500/10 text-emerald-400 border-emerald-500/20",
    positive: "bg-emerald-500/10 text-emerald-400 border-emerald-500/20",
    bearish: "bg-rose-500/10 text-rose-400 border-rose-500/20",
    negative: "bg-rose-500/10 text-rose-400 border-rose-500/20",
    neutral: "bg-gray-500/10 text-gray-400 border-gray-500/20",
  };
  const cls = Object.entries(map).find(([k]) => s.includes(k))?.[1] || map.neutral;
  const label = s.includes("bull") || s.includes("pos") ? "Bullish" : s.includes("bear") || s.includes("neg") ? "Bearish" : "Neutral";
  return <span className={`text-[9px] font-bold px-1.5 py-0.5 rounded-full border uppercase tracking-wider ${cls}`}>{label}</span>;
};

const getSentimentScore = (score: number) => {
  if (score > 0.6) return { label: "Positive", color: "text-emerald-400", bar: "bg-emerald-500" };
  if (score < 0.4) return { label: "Negative", color: "text-rose-400", bar: "bg-rose-500" };
  return { label: "Neutral", color: "text-amber-400", bar: "bg-amber-500" };
};

export default function News({ news, sentiment }: { news: NewsItem[]; sentiment?: { sentiment_score?: number } }) {
  const [expanded, setExpanded] = useState(false);
  const displayNews = expanded ? news : news.slice(0, 4);

  const sentInfo = sentiment?.sentiment_score != null ? getSentimentScore(sentiment.sentiment_score) : null;

  return (
    <div className="bg-[#0d0d0d] border border-white/[0.06] p-5 rounded-2xl">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xs font-bold text-gray-500 uppercase tracking-wider">Related News</h2>
        {sentInfo && (
          <div className="flex items-center gap-2">
            <span className={`text-xs font-semibold ${sentInfo.color}`}>
              {sentInfo.label} {((sentiment!.sentiment_score!) * 100).toFixed(0)}%
            </span>
            <div className="w-16 h-1.5 bg-white/[0.06] rounded-full overflow-hidden">
              <div className={`h-full rounded-full ${sentInfo.bar}`}
                style={{ width: `${(sentiment!.sentiment_score!) * 100}%` }} />
            </div>
          </div>
        )}
      </div>

      {/* News list */}
      {news.length === 0 ? (
        <p className="text-gray-600 text-xs text-center py-6">No news found for this asset.</p>
      ) : (
        <div className="space-y-3">
          {displayNews.map((n, i) => {
            const url = n.url?.trim() || n.link?.trim() ||
              `https://www.google.com/search?tbm=nws&q=${encodeURIComponent(n.title + " stock")}`;
            return (
              <a key={n.id ?? i} href={url} target="_blank" rel="noopener noreferrer"
                className="group flex gap-3 p-3 rounded-xl bg-white/[0.02] border border-white/[0.04] hover:bg-white/[0.05] hover:border-white/[0.08] transition-all">
                {n.thumbnail && (
                  <img src={n.thumbnail} alt="" onError={e => (e.currentTarget.style.display = "none")}
                    className="w-14 h-14 rounded-lg object-cover shrink-0 border border-white/[0.06]" />
                )}
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-1.5 mb-1 flex-wrap">
                    {n.source && <span className="text-[10px] font-bold text-blue-400/80">{n.source}</span>}
                    <SentimentPill sentiment={n.sentiment} />
                    {n.time && (
                      <span className="text-[10px] text-gray-600 flex items-center gap-0.5 ml-auto">
                        <Clock size={9} />{n.time}
                      </span>
                    )}
                  </div>
                  <h3 className="text-xs font-semibold text-gray-300 group-hover:text-white transition-colors leading-snug line-clamp-2 mb-1">
                    {n.title}
                  </h3>
                  {n.text && <p className="text-[10px] text-gray-600 line-clamp-2 leading-snug">{n.text}</p>}
                </div>
                <ExternalLink size={12} className="text-gray-700 group-hover:text-blue-400 transition-colors mt-1 shrink-0" />
              </a>
            );
          })}
        </div>
      )}

      {news.length > 4 && (
        <button onClick={() => setExpanded(p => !p)}
          className="w-full mt-4 py-2.5 rounded-xl bg-white/[0.03] border border-white/[0.05] text-xs font-semibold text-gray-500 hover:text-white hover:bg-white/[0.06] transition-all flex items-center justify-center gap-1.5">
          <ChevronDown size={13} className={`transition-transform ${expanded ? "rotate-180" : ""}`} />
          {expanded ? "Show less" : `Show ${news.length - 4} more articles`}
        </button>
      )}
    </div>
  );
}