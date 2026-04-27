"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { auth } from "@/lib/firebase";

import {
  signInWithEmailAndPassword,
  createUserWithEmailAndPassword,
  GoogleAuthProvider,
  signInWithPopup,
  sendPasswordResetEmail, // 🔥 Added for password reset
} from "firebase/auth";

import { motion } from "framer-motion";
import { Lock, Mail, TrendingUp, Activity, ShieldCheck, Loader2 } from "lucide-react";
import toast from "react-hot-toast"; // 🔥 Imported toast

export default function LoginPage() {
  const router = useRouter();

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [isLogin, setIsLogin] = useState(true);
  const [loading, setLoading] = useState(false);
  const [resetLoading, setResetLoading] = useState(false); // 🔥 State for reset button

  // Decorative state for the financial background effect
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);

  const handleEmailAuth = async () => {
    try {
      setLoading(true);

      if (isLogin) {
        await signInWithEmailAndPassword(auth, email, password);
        toast.success("Successfully accessed terminal.");
      } else {
        await createUserWithEmailAndPassword(auth, email, password);
        toast.success("Account initialized successfully.");
      }

      router.push("/");
    } catch (err: any) {
      toast.error(err.message || "Authentication failed."); // 🔥 Replaced alert
    } finally {
      setLoading(false);
    }
  };

  const handleGoogleLogin = async () => {
    try {
      setLoading(true);
      const provider = new GoogleAuthProvider();
      await signInWithPopup(auth, provider);
      toast.success("Authorized via Google.");
      router.push("/");
    } catch (err: any) {
      toast.error(err.message || "Google authorization failed."); // 🔥 Replaced alert
    } finally {
      setLoading(false);
    }
  };

  // 🔥 New Password Reset Function
  const handlePasswordReset = async () => {
    if (!email) {
      toast.error("Please enter your email address first.");
      return;
    }

    try {
      setResetLoading(true);
      await sendPasswordResetEmail(auth, email);
      toast.success("Recovery instructions sent to your email.");
    } catch (err: any) {
      toast.error(err.message || "Failed to send recovery email.");
    } finally {
      setResetLoading(false);
    }
  };

  return (
    <main className="relative min-h-screen bg-[#050505] text-white flex overflow-hidden font-sans selection:bg-blue-500/30">
      
      {/* 🌌 GLOBAL BACKGROUND NOISE & GRID */}
      <div className="absolute inset-0 z-0 opacity-[0.03] pointer-events-none" style={{ backgroundImage: 'linear-gradient(#ffffff 1px, transparent 1px), linear-gradient(90deg, #ffffff 1px, transparent 1px)', backgroundSize: '40px 40px' }} />
      <div className="absolute inset-0 bg-[url('https://grainy-gradients.vercel.app/noise.svg')] opacity-20 mix-blend-overlay pointer-events-none z-0"></div>

      {/* 📈 LEFT PANEL - FINANCIAL BRANDING (Hidden on Mobile) */}
      <div className="hidden lg:flex lg:w-1/2 relative flex-col justify-between p-12 border-r border-white/5 bg-gradient-to-br from-[#0a0a0a] to-[#050505] z-10">
        <div className="absolute top-[-20%] left-[-10%] w-[50vw] h-[50vw] bg-blue-600/10 blur-[150px] rounded-full pointer-events-none" />
        
        {/* Brand Header */}
        <div>
          <h1 className="text-3xl font-bold tracking-tight flex items-center gap-2">
            <Activity className="text-blue-500" />
            FinIntel<span className="text-blue-500">.</span>
          </h1>
        </div>

        {/* Hero Copy & Decorative Chart */}
        <div className="relative z-10 space-y-8">
          <motion.div initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.2 }} className="space-y-4">
            <div className="inline-flex items-center space-x-2 bg-emerald-500/10 text-emerald-400 px-3 py-1 rounded-full text-xs font-semibold tracking-wider border border-emerald-500/20">
              <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
              SYSTEM OPERATIONAL
            </div>
            <h2 className="text-5xl font-bold leading-tight tracking-tight">
              Institutional-Grade <br />
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-cyan-300">Market Intelligence</span>
            </h2>
            <p className="text-gray-400 text-lg max-w-md leading-relaxed">
              Access real-time predictive analytics, macro insights, and algorithmic sentiment tracking in one unified terminal.
            </p>
          </motion.div>

          {/* Abstract Financial UI Element */}
          {mounted && (
            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.4 }} className="w-full max-w-md h-40 bg-white/[0.02] border border-white/5 rounded-2xl p-6 relative overflow-hidden backdrop-blur-sm">
              <div className="absolute top-0 right-0 p-6 opacity-20">
                <TrendingUp size={100} className="text-blue-500" />
              </div>
              <div className="space-y-4 relative z-10">
                <div className="w-24 h-2 bg-white/10 rounded-full" />
                <div className="w-48 h-2 bg-white/10 rounded-full" />
                <div className="w-32 h-2 bg-white/10 rounded-full" />
                <div className="flex items-end gap-2 pt-4">
                  {[40, 70, 45, 90, 65, 110, 85].map((height, i) => (
                    <motion.div 
                      key={i}
                      initial={{ height: 0 }}
                      animate={{ height }}
                      transition={{ duration: 1, delay: 0.5 + (i * 0.1) }}
                      className="w-4 bg-blue-500/50 rounded-t-sm"
                    />
                  ))}
                </div>
              </div>
            </motion.div>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center gap-4 text-sm text-gray-500 font-medium">
          <ShieldCheck size={18} className="text-emerald-500" />
          <span>Bank-level AES-256 Encryption</span>
        </div>
      </div>

      {/* 🔐 RIGHT PANEL - AUTHENTICATION */}
      <div className="w-full lg:w-1/2 flex items-center justify-center p-6 relative z-10">
        
        {/* Mobile Background Elements */}
        <div className="absolute top-0 right-0 w-[50vw] h-[50vw] bg-purple-600/10 blur-[120px] rounded-full pointer-events-none lg:hidden" />

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="w-full max-w-[420px] p-8 sm:p-10 rounded-[2rem] bg-[#0a0a0a]/80 backdrop-blur-2xl border border-white/10 shadow-[0_20px_80px_rgba(0,0,0,0.5)] space-y-8"
        >
          {/* LOGO (Mobile Only) */}
          <div className="text-left lg:hidden mb-8">
             <h1 className="text-2xl font-bold tracking-tight flex items-center gap-2">
              <Activity className="text-blue-500" size={24} />
              FinIntel<span className="text-blue-500">.</span>
            </h1>
          </div>

          <div className="text-left">
            <h2 className="text-2xl font-semibold tracking-tight text-white">
              {isLogin ? "Access Terminal" : "Initialize Account"}
            </h2>
            <p className="text-sm text-gray-400 mt-2">
              {isLogin ? "Enter your credentials to connect to the market." : "Set up your secure access key."}
            </p>
          </div>

          {/* INPUTS */}
          <div className="space-y-5">
            <div className="space-y-1">
              <label className="text-xs font-semibold text-gray-400 uppercase tracking-wider ml-1">Email Address</label>
              <div className="relative">
                <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
                  <Mail size={18} className="text-gray-500" />
                </div>
                <input
                  type="email"
                  value={email} // 🔥 Added explicit value binding
                  placeholder="investor@domain.com"
                  className="w-full pl-11 pr-4 py-3.5 rounded-xl bg-white/[0.03] border border-white/10 focus:border-blue-500/50 focus:bg-blue-500/5 focus:ring-4 focus:ring-blue-500/10 outline-none transition-all text-white placeholder-gray-600"
                  onChange={(e) => setEmail(e.target.value)}
                />
              </div>
            </div>

            <div className="space-y-1">
              <div className="flex items-center justify-between ml-1">
                <label className="text-xs font-semibold text-gray-400 uppercase tracking-wider">Secure Password</label>
                {/* 🔥 Updated Forgot Password Button */}
                {isLogin && (
                  <button 
                    onClick={handlePasswordReset}
                    disabled={resetLoading}
                    className="text-xs text-blue-400 hover:text-blue-300 transition-colors disabled:opacity-50"
                  >
                    {resetLoading ? "Sending..." : "Recover access?"}
                  </button>
                )}
              </div>
              <div className="relative">
                <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
                  <Lock size={18} className="text-gray-500" />
                </div>
                <input
                  type="password"
                  value={password} // 🔥 Added explicit value binding
                  placeholder="••••••••"
                  className="w-full pl-11 pr-4 py-3.5 rounded-xl bg-white/[0.03] border border-white/10 focus:border-blue-500/50 focus:bg-blue-500/5 focus:ring-4 focus:ring-blue-500/10 outline-none transition-all text-white placeholder-gray-600"
                  onChange={(e) => setPassword(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && handleEmailAuth()} // 🔥 Added enter key support
                />
              </div>
            </div>
          </div>

          {/* PRIMARY BUTTON */}
          <button
            onClick={handleEmailAuth}
            disabled={loading}
            className="w-full py-4 rounded-xl font-semibold bg-gradient-to-r from-blue-600 to-blue-500 hover:from-blue-500 hover:to-blue-400 text-white transition-all shadow-[0_0_20px_rgba(59,130,246,0.3)] hover:shadow-[0_0_30px_rgba(59,130,246,0.5)] flex items-center justify-center gap-2 active:scale-[0.98] disabled:opacity-70"
          >
            {loading ? (
              <><Loader2 size={18} className="animate-spin" /> Authenticating...</>
            ) : (
              <>{isLogin ? "Connect securely" : "Generate credentials"}</>
            )}
          </button>

          {/* DIVIDER */}
          <div className="flex items-center gap-4 text-xs text-gray-500 font-semibold tracking-widest">
            <div className="flex-1 h-px bg-white/10" />
            OR
            <div className="flex-1 h-px bg-white/10" />
          </div>

          {/* GOOGLE BUTTON */}
          <button
            onClick={handleGoogleLogin}
            disabled={loading}
            className="w-full flex items-center justify-center gap-3 py-3.5 rounded-xl bg-white/[0.05] border border-white/10 text-white font-medium hover:bg-white/10 transition-colors active:scale-[0.98] disabled:opacity-70"
          >
            <img
              src="https://www.svgrepo.com/show/475656/google-color.svg"
              className="w-5 h-5"
              alt="Google logo"
            />
            Continue with Google
          </button>

          {/* TOGGLE */}
          <p className="text-center text-sm text-gray-400 mt-8">
            {isLogin ? "No access key yet?" : "Already authorized?"}{" "}
            <button
              onClick={() => setIsLogin(!isLogin)}
              className="text-blue-400 hover:text-blue-300 font-medium transition-colors"
            >
              {isLogin ? "Create account" : "Log in here"}
            </button>
          </p>

        </motion.div>
      </div>
    </main>
  );
}