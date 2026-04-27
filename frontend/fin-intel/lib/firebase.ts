import { initializeApp, getApps, getApp } from "firebase/app";
import { getAuth } from "firebase/auth";

// ─────────────────────────────────────────────
// Firebase configuration
// Add your Firebase project credentials to .env.local:
//
//   NEXT_PUBLIC_FIREBASE_API_KEY=...
//   NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN=...
//   NEXT_PUBLIC_FIREBASE_PROJECT_ID=...
//   NEXT_PUBLIC_FIREBASE_STORAGE_BUCKET=...
//   NEXT_PUBLIC_FIREBASE_MESSAGING_SENDER_ID=...
//   NEXT_PUBLIC_FIREBASE_APP_ID=...
// ─────────────────────────────────────────────
const firebaseConfig = { apiKey: "AIzaSyDxLjjuaMPwDPXbcaXTyKZAUlqDoWxzLvw", authDomain: "finintel-7ffec.firebaseapp.com", projectId: "finintel-7ffec", storageBucket: "finintel-7ffec.firebasestorage.app", messagingSenderId: "671404986977", appId: "1:671404986977:web:1a9cac77da63bc18179020", measurementId: "G-HF8PCW6D56" };

// Prevent duplicate initialization in Next.js dev (hot reload)
const app = getApps().length ? getApp() : initializeApp(firebaseConfig);

export const auth = getAuth(app);
export default app;