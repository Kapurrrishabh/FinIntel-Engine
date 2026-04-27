"use client";

import { usePathname } from "next/navigation";
import Navbar from "./Navbar";

export default function NavbarWrapper() {
  const pathname = usePathname();

  // Define the routes where the Navbar should NOT appear
  const hiddenRoutes = ["/login", "/register", "/forgot-password"];

  // If the current URL matches a hidden route, render absolutely nothing
  if (hiddenRoutes.includes(pathname)) {
    return null;
  }

  // Otherwise, render your normal Navbar
  return <Navbar />;
}