/* Stroke Detection System — Header Component */

import { Activity, ExternalLink } from 'lucide-react';

export default function Header() {
  return (
    <header className="sticky top-0 z-50 glass-elevated border-b border-[var(--color-border)]">
      <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
        {/* Logo */}
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-500 to-cyan-400 flex items-center justify-center shadow-lg">
            <Activity className="w-5 h-5 text-white" />
          </div>
          <div>
            <h1 className="text-lg font-bold tracking-tight bg-gradient-to-r from-indigo-400 to-cyan-400 bg-clip-text text-transparent">
              NeuroScan AI
            </h1>
            <p className="text-[10px] text-[var(--color-text-muted)] uppercase tracking-widest">
              Stroke Detection System
            </p>
          </div>
        </div>

        {/* Status Badge */}
        <div className="flex items-center gap-4">
          <div className="hidden sm:flex items-center gap-2 px-3 py-1.5 rounded-full bg-emerald-500/10 border border-emerald-500/20">
            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
            <span className="text-xs font-medium text-emerald-400">System Online</span>
          </div>
          <a
            href="https://github.com"
            target="_blank"
            rel="noopener"
            className="w-9 h-9 rounded-lg flex items-center justify-center text-[var(--color-text-muted)] hover:text-white hover:bg-white/5 transition-all"
          >
            <ExternalLink className="w-4 h-4" />
          </a>
        </div>
      </div>
    </header>
  );
}
