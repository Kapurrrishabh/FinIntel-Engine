"use client";
import { Suspense } from "react";
import { useState, useEffect } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import {
  User, Bookmark, Clock, TrendingUp, TrendingDown,
  BarChart2, Settings, ChevronRight, Star, X,
  Activity, Award, Target, Zap, LogOut, Edit3,
} from "lucide-react";
import { useAuth } from "@/hooks/useAuth";
import { signOut } from "firebase/auth";
import { auth } from "@/lib/firebase";
import toast from "react-hot-toast";

// ─────────────────────────────────────────────
// TYPES
// ─────────────────────────────────────────────
interface WatchlistItem { symbol: string; name: string; price: string; change: string; isUp: boolean; }
interface RecentSearch { symbol: string; timestamp: string; }
interface Activity { id: string; type: "view" | "alert" | "compare"; symbol: string; desc: string; time: string; }

// ─────────────────────────────────────────────
// MOCK DATA
// ─────────────────────────────────────────────
const MOCK_WATCHLIST: WatchlistItem[] = [
  { symbol: "AAPL", name: "Apple Inc.", price: "$173.50", change: "+0.8%", isUp: true },
  { symbol: "NVDA", name: "NVIDIA Corp", price: "$880.00", change: "+4.2%", isUp: true },
  { symbol: "RELIANCE.NS", name: "Reliance Ind.", price: "₹2,950", change: "+1.5%", isUp: true },
  { symbol: "BTC", name: "Bitcoin", price: "$64,210", change: "+5.4%", isUp: true },
  { symbol: "TSLA", name: "Tesla Inc.", price: "$175.22", change: "-2.1%", isUp: false },
  { symbol: "INFY.NS", name: "Infosys Ltd.", price: "₹1,540", change: "+0.9%", isUp: true },
];

const MOCK_RECENT: RecentSearch[] = [
  { symbol: "NVDA", timestamp: "2h ago" },
  { symbol: "AAPL", timestamp: "4h ago" },
  { symbol: "BTC", timestamp: "Yesterday" },
  { symbol: "RELIANCE.NS", timestamp: "2 days ago" },
  { symbol: "TSLA", timestamp: "3 days ago" },
];

const MOCK_ACTIVITY: Activity[] = [
  { id: "1", type: "view", symbol: "NVDA", desc: "Viewed NVIDIA Corp analysis", time: "2h ago" },
  { id: "2", type: "alert", symbol: "TSLA", desc: "Price alert triggered at $180", time: "4h ago" },
  { id: "3", type: "compare", symbol: "AAPL vs MSFT", desc: "Compared Apple vs Microsoft", time: "Yesterday" },
  { id: "4", type: "view", symbol: "BTC", desc: "Viewed Bitcoin analysis", time: "2 days ago" },
  { id: "5", type: "alert", symbol: "NVDA", desc: "NVIDIA crossed $900 target", time: "3 days ago" },
];

const TABS = ["Overview", "Watchlist", "Recent", "Activity"] as const;
type Tab = typeof TABS[number];

// ─────────────────────────────────────────────
// AVATAR
// ─────────────────────────────────────────────
const Avatar = ({ user, size = 80 }: { user: { displayName?: string | null; email?: string | null; photoURL?: string | null }; size?: number }) => {
  const [err, setErr] = useState(false);
  const initials = (user.displayName || user.email || "U").slice(0, 2).toUpperCase();
  const colors = ["from-blue-600 to-cyan-500", "from-violet-600 to-blue-500", "from-emerald-600 to-teal-500"];
  const ci = initials.charCodeAt(0) % colors.length;
  if (user.photoURL && !err) return <img src={user.photoURL} alt="Avatar" width={size} height={size} className="rounded-2xl object-cover" style={{ width: size, height: size }} onError={() => setErr(true)} />;
  return <div className={`rounded-2xl bg-gradient-to-br ${colors[ci]} flex items-center justify-center font-black text-white shadow-lg`} style={{ width: size, height: size, fontSize: size * 0.32 }}>{initials}</div>;
};

// ─────────────────────────────────────────────
// STAT CARD
// ─────────────────────────────────────────────
const StatCard = ({ icon, label, value, color }: { icon: React.ReactNode; label: string; value: string; color: string }) => (
  <div className="bg-white/[0.03] border border-white/[0.06] rounded-2xl p-4 flex flex-col gap-2">
    <div className={`w-8 h-8 rounded-xl ${color} flex items-center justify-center`}>{icon}</div>
    <div>
      <p className="text-xl font-black text-white">{value}</p>
      <p className="text-xs text-gray-500 font-medium mt-0.5">{label}</p>
    </div>
  </div>
);

// ─────────────────────────────────────────────
// PAGE
// ─────────────────────────────────────────────
export default function ProfilePage() {
  return (
    <Suspense fallback={<div className="text-center mt-20">Loading Profile...</div>}>
      <ProfileContent />
    </Suspense>
  );
}

function ProfileContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const { user, loading } = useAuth();
  const [activeTab, setActiveTab] = useState<Tab>("Overview");
  const [watchlist, setWatchlist] = useState<WatchlistItem[]>(MOCK_WATCHLIST);

  useEffect(() => {
    const tab = searchParams.get("tab");
    if (tab === "watchlist") setActiveTab("Watchlist");
  }, [searchParams]);

  const handleSignOut = async () => {
    await signOut(auth);
    toast.success("Signed out");
    router.push("/login");
  };

  const removeFromWatchlist = (sym: string) => {
    setWatchlist(prev => prev.filter(w => w.symbol !== sym));
    toast.success(`Removed ${sym} from watchlist`);
  };

  if (loading) return (
    <div className="min-h-screen bg-[#060606] flex items-center justify-center">
      <div className="flex flex-col items-center gap-4">
        <div className="w-10 h-10 rounded-full border-2 border-blue-500 border-t-transparent animate-spin" />
        <p className="text-gray-500 text-sm">Loading profile…</p>
      </div>
    </div>
  );

  if (!user) { router.push("/login"); return null; }

  const activityIcon = (type: string) => ({ view: <BarChart2 size={14} />, alert: <Zap size={14} />, compare: <Activity size={14} /> }[type] || <Activity size={14} />);
  const activityColor = (type: string) => ({ view: "bg-blue-500/15 text-blue-400", alert: "bg-amber-500/15 text-amber-400", compare: "bg-violet-500/15 text-violet-400" }[type] || "bg-gray-500/15 text-gray-400");

  return (
    <main className="min-h-screen bg-[#060606] text-white pt-28 pb-16">
      <style>{`.scrollbar-hide::-webkit-scrollbar{display:none}.scrollbar-hide{-ms-overflow-style:none;scrollbar-width:none}`}</style>

      {/* Background */}
      <div className="fixed inset-0 -z-10 pointer-events-none">
        <div className="absolute top-0 left-1/4 w-96 h-96 bg-blue-900/15 blur-[120px] rounded-full" />
        <div className="absolute bottom-0 right-1/4 w-64 h-64 bg-indigo-900/10 blur-[100px] rounded-full" />
      </div>

      <div className="max-w-5xl mx-auto px-4 sm:px-6">

        {/* Profile Header */}
        <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }}
          className="bg-gradient-to-br from-white/[0.04] to-white/[0.01] border border-white/[0.08] rounded-3xl p-6 sm:p-8 mb-6">
          <div className="flex flex-col sm:flex-row sm:items-center gap-6">
            <div className="relative flex-shrink-0">
              <Avatar user={user} size={88} />
              <button className="absolute -bottom-1 -right-1 w-7 h-7 rounded-full bg-blue-600 border-2 border-[#060606] flex items-center justify-center hover:bg-blue-500 transition-colors">
                <Edit3 size={11} className="text-white" />
              </button>
            </div>

            <div className="flex-1 min-w-0">
              <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-4">
                <div>
                  <h1 className="text-2xl font-black tracking-tight">{user.displayName || "Investor"}</h1>
                  <p className="text-gray-400 text-sm mt-0.5">{user.email}</p>
                  <div className="flex items-center gap-2 mt-2">
                    <span className="text-[10px] font-bold bg-blue-500/15 text-blue-400 border border-blue-500/25 px-2.5 py-1 rounded-full uppercase tracking-wider">Pro Member</span>
                    <span className="text-[10px] text-gray-600">Joined March 2024</span>
                  </div>
                </div>
                <div className="flex gap-2 flex-wrap">
                  <button onClick={() => router.push("/settings")}
                    className="flex items-center gap-2 px-4 py-2 rounded-xl bg-white/[0.05] border border-white/[0.08] text-sm font-semibold text-gray-300 hover:bg-white/[0.08] hover:text-white transition-all">
                    <Settings size={15} /> Settings
                  </button>
                  <button onClick={handleSignOut}
                    className="flex items-center gap-2 px-4 py-2 rounded-xl bg-rose-500/10 border border-rose-500/20 text-sm font-semibold text-rose-400 hover:bg-rose-500/15 transition-all">
                    <LogOut size={15} /> Sign Out
                  </button>
                </div>
              </div>
            </div>
          </div>

          {/* Stats */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-6">
            <StatCard icon={<Bookmark size={16} className="text-blue-400" />} label="Watchlist" value={String(watchlist.length)} color="bg-blue-500/15" />
            <StatCard icon={<Target size={16} className="text-amber-400" />} label="Active Alerts" value="3" color="bg-amber-500/15" />
            <StatCard icon={<Award size={16} className="text-emerald-400" />} label="Portfolio Gain" value="+8.4%" color="bg-emerald-500/15" />
            <StatCard icon={<TrendingUp size={16} className="text-violet-400" />} label="Stocks Tracked" value="24" color="bg-violet-500/15" />
          </div>
        </motion.div>

        {/* Tabs */}
        <div className="flex gap-1 p-1 bg-white/[0.03] border border-white/[0.06] rounded-2xl mb-6">
          {TABS.map(tab => (
            <button key={tab} onClick={() => setActiveTab(tab)}
              className={`flex-1 py-2.5 rounded-xl text-sm font-semibold transition-all ${activeTab === tab ? "bg-white/[0.09] text-white shadow-sm" : "text-gray-500 hover:text-gray-300"}`}>
              {tab}
            </button>
          ))}
        </div>

        {/* Tab Content */}
        <AnimatePresence mode="wait">
          <motion.div key={activeTab} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -8 }} transition={{ duration: 0.18 }}>

            {/* OVERVIEW */}
            {activeTab === "Overview" && (
              <div className="space-y-4">
                <div className="bg-white/[0.02] border border-white/[0.06] rounded-2xl p-5">
                  <h2 className="font-bold text-base mb-4 flex items-center gap-2"><Star size={16} className="text-yellow-400 fill-yellow-400" /> Pinned Watchlist</h2>
                  <div className="grid sm:grid-cols-2 gap-2">
                    {watchlist.slice(0, 4).map(item => (
                      <button key={item.symbol} onClick={() => router.push(`/dashboard?symbol=${item.symbol}`)}
                        className="flex items-center justify-between p-3 rounded-xl bg-white/[0.02] border border-white/[0.05] hover:bg-white/[0.05] hover:border-white/10 transition-all group text-left">
                        <div className="flex items-center gap-3">
                          <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-white/[0.08] to-white/[0.02] border border-white/[0.07] flex items-center justify-center text-[10px] font-black text-blue-400">
                            {item.symbol.slice(0, 2)}
                          </div>
                          <div>
                            <p className="text-sm font-bold">{item.symbol}</p>
                            <p className="text-[11px] text-gray-500 truncate max-w-[100px]">{item.name}</p>
                          </div>
                        </div>
                        <div className="text-right">
                          <p className="text-sm font-bold">{item.price}</p>
                          <p className={`text-xs font-semibold flex items-center justify-end gap-0.5 ${item.isUp ? "text-emerald-400" : "text-rose-400"}`}>
                            {item.isUp ? <TrendingUp size={10} /> : <TrendingDown size={10} />} {item.change}
                          </p>
                        </div>
                      </button>
                    ))}
                  </div>
                  <button onClick={() => setActiveTab("Watchlist")} className="mt-3 w-full py-2.5 rounded-xl text-xs font-semibold text-gray-500 hover:text-white hover:bg-white/[0.04] transition-colors flex items-center justify-center gap-1.5">
                    View All {watchlist.length} <ChevronRight size={13} />
                  </button>
                </div>

                <div className="bg-white/[0.02] border border-white/[0.06] rounded-2xl p-5">
                  <h2 className="font-bold text-base mb-4 flex items-center gap-2"><Clock size={16} className="text-gray-400" /> Recent Activity</h2>
                  <div className="space-y-2">
                    {MOCK_ACTIVITY.slice(0, 3).map(act => (
                      <div key={act.id} className="flex items-center gap-3 p-2 rounded-xl hover:bg-white/[0.03] transition-colors">
                        <div className={`w-7 h-7 rounded-xl flex items-center justify-center flex-shrink-0 ${activityColor(act.type)}`}>{activityIcon(act.type)}</div>
                        <div className="flex-1 min-w-0">
                          <p className="text-xs font-semibold text-gray-200 truncate">{act.desc}</p>
                          <p className="text-[10px] text-gray-600">{act.time}</p>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {/* WATCHLIST */}
            {activeTab === "Watchlist" && (
              <div className="bg-white/[0.02] border border-white/[0.06] rounded-2xl overflow-hidden">
                <div className="px-5 py-4 border-b border-white/[0.06] flex items-center justify-between">
                  <h2 className="font-bold flex items-center gap-2"><Bookmark size={15} className="text-blue-400" /> Saved Stocks</h2>
                  <span className="text-xs text-gray-500">{watchlist.length} assets</span>
                </div>
                <div className="divide-y divide-white/[0.04]">
                  {watchlist.map(item => (
                    <div key={item.symbol} className="flex items-center justify-between px-5 py-4 hover:bg-white/[0.03] transition-colors group">
                      <div className="flex items-center gap-4">
                        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-white/[0.08] to-white/[0.02] border border-white/[0.07] flex items-center justify-center text-xs font-black text-blue-400">
                          {item.symbol.slice(0, 2)}
                        </div>
                        <div>
                          <p className="font-bold">{item.symbol}</p>
                          <p className="text-xs text-gray-500">{item.name}</p>
                        </div>
                      </div>
                      <div className="flex items-center gap-4">
                        <div className="text-right hidden sm:block">
                          <p className="font-bold text-sm">{item.price}</p>
                          <p className={`text-xs font-semibold flex items-center justify-end gap-0.5 ${item.isUp ? "text-emerald-400" : "text-rose-400"}`}>
                            {item.isUp ? <TrendingUp size={11} /> : <TrendingDown size={11} />} {item.change}
                          </p>
                        </div>
                        <div className="flex gap-1.5 opacity-0 group-hover:opacity-100 transition-opacity">
                          <button onClick={() => router.push(`/dashboard?symbol=${item.symbol}`)}
                            className="p-2 rounded-lg bg-blue-500/10 text-blue-400 hover:bg-blue-500/20 transition-colors">
                            <BarChart2 size={14} />
                          </button>
                          <button onClick={() => removeFromWatchlist(item.symbol)}
                            className="p-2 rounded-lg bg-rose-500/10 text-rose-400 hover:bg-rose-500/20 transition-colors">
                            <X size={14} />
                          </button>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* RECENT SEARCHES */}
            {activeTab === "Recent" && (
              <div className="bg-white/[0.02] border border-white/[0.06] rounded-2xl overflow-hidden">
                <div className="px-5 py-4 border-b border-white/[0.06] flex items-center justify-between">
                  <h2 className="font-bold flex items-center gap-2"><Clock size={15} className="text-gray-400" /> Recent Searches</h2>
                  <button className="text-xs text-gray-500 hover:text-white transition-colors">Clear all</button>
                </div>
                <div className="divide-y divide-white/[0.04]">
                  {MOCK_RECENT.map((item, i) => (
                    <div key={i} className="flex items-center justify-between px-5 py-4 hover:bg-white/[0.03] transition-colors group cursor-pointer" onClick={() => router.push(`/dashboard?symbol=${item.symbol}`)}>
                      <div className="flex items-center gap-3">
                        <Clock size={14} className="text-gray-600" />
                        <span className="font-bold text-sm">{item.symbol}</span>
                      </div>
                      <div className="flex items-center gap-3">
                        <span className="text-xs text-gray-600">{item.timestamp}</span>
                        <ChevronRight size={14} className="text-gray-700 opacity-0 group-hover:opacity-100 transition-opacity" />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* ACTIVITY */}
            {activeTab === "Activity" && (
              <div className="bg-white/[0.02] border border-white/[0.06] rounded-2xl overflow-hidden">
                <div className="px-5 py-4 border-b border-white/[0.06]">
                  <h2 className="font-bold flex items-center gap-2"><Activity size={15} className="text-blue-400" /> Activity Log</h2>
                </div>
                <div className="divide-y divide-white/[0.04]">
                  {MOCK_ACTIVITY.map(act => (
                    <div key={act.id} className="flex items-start gap-4 px-5 py-4 hover:bg-white/[0.02] transition-colors">
                      <div className={`w-8 h-8 rounded-xl flex items-center justify-center flex-shrink-0 mt-0.5 ${activityColor(act.type)}`}>{activityIcon(act.type)}</div>
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-semibold text-gray-200">{act.desc}</p>
                        <p className="text-xs text-gray-600 mt-0.5">{act.symbol} · {act.time}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

          </motion.div>
        </AnimatePresence>
      </div>
    </main>
  );
}