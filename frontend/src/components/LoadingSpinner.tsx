/* Stroke Detection System — Loading Spinner */

import { Search } from 'lucide-react';

interface Props {
  message?: string;
}

export default function LoadingSpinner({ message = 'Analysing scan…' }: Props) {
  return (
    <div className="glass-elevated p-10 flex flex-col items-center justify-center gap-5 animate-fade-in">
      <div className="relative">
        <div className="w-16 h-16 rounded-full border-2 border-indigo-500/20 border-t-indigo-400 animate-spin" />
        <div className="absolute inset-0 flex items-center justify-center">
          <Search className="w-6 h-6 text-indigo-400" />
        </div>
      </div>
      <div className="text-center">
        <p className="font-semibold text-white">{message}</p>
        <p className="text-xs text-[var(--color-text-muted)] mt-1">
          Running AI models — this may take a moment
        </p>
      </div>
      {/* Shimmer bar */}
      <div className="w-48 h-1 rounded-full overflow-hidden bg-white/5">
        <div
          className="h-full rounded-full w-1/3"
          style={{
            background: 'linear-gradient(90deg, transparent, #6366f1, transparent)',
            backgroundSize: '200% 100%',
            animation: 'shimmer 1.5s infinite',
          }}
        />
      </div>
    </div>
  );
}
