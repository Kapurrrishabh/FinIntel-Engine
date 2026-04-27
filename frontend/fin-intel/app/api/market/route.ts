// app/api/market/route.ts
import { NextResponse } from "next/server";

// Cache this endpoint for 10 seconds natively in Next.js
export const revalidate = 10; 

export async function GET() {
  try {
    // In production, replace this with your actual data provider (e.g., yahoo-finance2, Alpha Vantage)
    // Example: const niftyData = await yahooFinance.quote('^NSEI');
    
    // MOCKING UPSTREAM FETCH FOR DEMONSTRATION
    const marketData = {
      ticker: [
        { symbol: "NIFTY 50", price: "22,514.65", change: 0.8 },
        { symbol: "SENSEX", price: "74,221.05", change: 0.7 },
        { symbol: "RELIANCE", price: "2,950.20", change: 1.5 },
        { symbol: "TCS", price: "3,890.00", change: -0.4 },
        { symbol: "INFY", price: "1,480.50", change: -1.2 },
        { symbol: "HDFCBANK", price: "1,530.10", change: 2.1 },
      ],
      trending: [
        { symbol: "RELIANCE.NS", name: "Reliance Ind", price: "₹2,950", change: "+1.5%", isUp: true, history: [20, 22, 25, 24, 28, 30, 35, 36] },
        { symbol: "TCS.NS", name: "Tata Consultancy", price: "₹3,890", change: "-0.4%", isUp: false, history: [80, 75, 78, 65, 60, 55, 60, 50] },
      ],
      // Compute insights dynamically based on the fetched data
      insights: [
        { label: "Market Trend", value: "Bullish", color: "text-emerald-400", bg: "bg-emerald-400/10", conf: 85 },
        { label: "India VIX", value: "12.4 (Stable)", color: "text-blue-400", bg: "bg-blue-400/10", conf: 90 },
        { label: "IT Sector", value: "Underperforming", color: "text-rose-400", bg: "bg-rose-400/10", conf: 75 },
      ]
    };

    return NextResponse.json(marketData);
  } catch (error) {
    return NextResponse.json({ error: "Failed to fetch market data" }, { status: 500 });
  }
}