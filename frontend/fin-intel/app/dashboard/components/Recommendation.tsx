"use client";

import { motion } from "framer-motion";

type Props = {
  score: number;
};

export default function Recommendation({ score }: Props) {

  let label = "Hold";
  let color = "text-yellow-400";
  let bg = "bg-yellow-500/10";
  let glow = "shadow-yellow-500/20";
  let barColor = "bg-yellow-400";

  if (score > 0.7) {
    label = "Buy";
    color = "text-green-400";
    bg = "bg-green-500/10";
    glow = "shadow-green-500/20";
    barColor = "bg-green-400";
  } else if (score < 0.4) {
    label = "Avoid";
    color = "text-red-400";
    bg = "bg-red-500/10";
    glow = "shadow-red-500/20";
    barColor = "bg-red-400";
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 15 }}
      animate={{ opacity: 1, y: 0 }}
      className={`p-6 rounded-2xl border border-white/10 backdrop-blur-xl ${bg} ${glow} shadow-lg`}
    >

      {/* HEADER */}
      <div className="flex justify-between items-center">

        <div>
          <p className="text-sm text-gray-400 mb-1">
            AI Recommendation
          </p>

          <h2 className={`text-3xl font-bold tracking-wide ${color}`}>
            {label}
          </h2>
        </div>

        <div className="text-right">
          <p className="text-sm text-gray-400">Confidence</p>
          <p className="text-xl font-semibold text-white">
            {(score * 100).toFixed(0)}%
          </p>
        </div>

      </div>

      {/* PROGRESS BAR */}
      <div className="mt-6">
        <div className="w-full bg-white/10 h-3 rounded-full overflow-hidden">

          <motion.div
            initial={{ width: 0 }}
            animate={{ width: `${score * 100}%` }}
            transition={{ duration: 0.6 }}
            className={`h-3 rounded-full ${barColor}`}
          />

        </div>

        {/* SCALE LABELS */}
        <div className="flex justify-between text-xs text-gray-500 mt-2">
          <span>Avoid</span>
          <span>Hold</span>
          <span>Buy</span>
        </div>
      </div>

    </motion.div>
  );
}