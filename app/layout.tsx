import type { Metadata } from "next";
import "./globals.css";

/**
 * NOTE ON FONTS: this project is designed around Space Grotesk (display),
 * Inter (body), and JetBrains Mono (telemetry) via next/font/google, which
 * self-hosts the font files at build time. The sandbox this was built in
 * has no route to fonts.googleapis.com, so next/font/google fails there.
 * We fall back to close system-font stacks (see app/globals.css) so the
 * build succeeds everywhere. On a host with normal internet access (e.g.
 * Vercel), swap the CSS var fallbacks below for the real next/font/google
 * imports — see docs/TROUBLESHOOTING.md for the exact diff.
 */

export const metadata: Metadata = {
  title: "Sign \u21c4 Speech Translator",
  description:
    "Real-time continuous Arabic Sign Language alphabet to text, and speech to text, running locally in your browser.",
};

export const viewport = {
  width: "device-width",
  initialScale: 1,
  maximumScale: 1,
  userScalable: false,
  themeColor: "#0B1220",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body className="antialiased">
        {children}
      </body>
    </html>
  );
}
