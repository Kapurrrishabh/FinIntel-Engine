"use client";

import {
  Chart as ChartJS,
  LineElement,
  PointElement,
  LinearScale,
  CategoryScale,
  Tooltip,
  Filler,
} from "chart.js";
import { Line } from "react-chartjs-2";

ChartJS.register(
  LineElement,
  PointElement,
  LinearScale,
  CategoryScale,
  Tooltip,
  Filler
);

type PriceData = {
  date: string;
  close: number;
};

export default function Chart({ data = [] }: { data?: PriceData[] }) {

  // 🧠 Handle empty state
  if (!data || data.length === 0) {
    return (
      <div className="bg-white/5 backdrop-blur-xl border border-white/10 p-6 rounded-2xl h-[350px] flex items-center justify-center text-gray-400">
        Loading chart...
      </div>
    );
  }

  const chartData = {
    labels: data.map((d) => d.date),
    datasets: [
      {
        label: "Price",
        data: data.map((d) => d.close),
        borderColor: "#3b82f6",

        // ✅ simple gradient-like fill (stable)
        backgroundColor: "rgba(59, 130, 246, 0.2)",

        fill: true,
        borderWidth: 2,
        tension: 0.4,
        pointRadius: 0,
        pointHoverRadius: 5,
        pointHoverBackgroundColor: "#3b82f6",
      },
    ],
  };

  const options: any = {
    responsive: true,
    maintainAspectRatio: false,

    plugins: {
      legend: { display: false },
      tooltip: {
        backgroundColor: "#111827",
        borderColor: "#374151",
        borderWidth: 1,
        padding: 10,
        titleColor: "#fff",
        bodyColor: "#9ca3af",
        displayColors: false,
        callbacks: {
          label: (context: any) => `$${context.parsed.y}`,
        },
      },
    },

    scales: {
      x: {
        grid: { display: false },
        ticks: { color: "#6b7280", maxTicksLimit: 6 },
      },
      y: {
        grid: { color: "rgba(255,255,255,0.05)" },
        ticks: {
          color: "#6b7280",
          callback: (value: any) => "$" + value,
        },
      },
    },

    interaction: {
      mode: "index",
      intersect: false,
    },
  };

  return (
    <div className="bg-white/5 backdrop-blur-xl border border-white/10 p-6 rounded-2xl h-[350px] shadow-lg">
      <Line data={chartData} options={options} />
    </div>
  );
}