"use client";

import { motion } from "framer-motion";

type Props = {
  risk: number; // 0 → 1
};

export default function RiskMeter({ risk }: Props) {
  const value = Math.min(Math.max(risk, 0), 1);

  let label = "Moderate Risk";
  let color = "text-yellow-400";
  let barColor = "bg-yellow-400";
  let glow = "shadow-yellow-500/20";
  let bg = "bg-yellow-500/10";

  if (value < 0.3) {
    label = "Low Risk";
    color = "text-green-400";
    barColor = "bg-green-400";
    glow = "shadow-green-500/20";
    bg = "bg-green-500/10";
  } else if (value > 0.7) {
    label = "High Risk";
    color = "text-red-400";
    barColor = "bg-red-400";
    glow = "shadow-red-500/20";
    bg = "bg-red-500/10";
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 15 }}
      animate={{ opacity: 1, y: 0 }}
      className={`p-6 rounded-2xl border border-white/10 backdrop-blur-xl ${bg} ${glow} shadow-lg`}
    >

      {/* HEADER */}
      <div className="flex justify-between items-center mb-4">

        <div>
          <p className="text-sm text-gray-400 mb-1">
            Risk Level
          </p>

          <h2 className={`text-2xl font-bold ${color}`}>
            {label}
          </h2>
        </div>

        <div className="text-right">
          <p className="text-sm text-gray-400">Risk Score</p>
          <p className="text-lg font-semibold text-white">
            {(value * 100).toFixed(0)}%
          </p>
        </div>

      </div>

      {/* PROGRESS BAR */}
      <div className="w-full bg-white/10 h-3 rounded-full overflow-hidden">

        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${value * 100}%` }}
          transition={{ duration: 0.6 }}
          className={`h-3 rounded-full ${barColor}`}
        />

      </div>

      {/* SCALE */}
      <div className="flex justify-between text-xs text-gray-500 mt-2">
        <span>Low</span>
        <span>Moderate</span>
        <span>High</span>
      </div>

    </motion.div>
  );
}