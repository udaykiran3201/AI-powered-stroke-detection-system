/* Stroke Detection System — CT Scan Viewer with Overlay */

import { useState } from 'react';
import { ZoomIn, ZoomOut, RotateCw, Layers } from 'lucide-react';

interface Props {
  originalUrl: string | null;
  overlayUrl: string | null;
  maskUrl: string | null;
}

export default function ScanViewer({ originalUrl, overlayUrl, maskUrl }: Props) {
  const [zoom, setZoom] = useState(1);
  const [rotation, setRotation] = useState(0);
  const [activeView, setActiveView] = useState<'original' | 'overlay' | 'mask'>('overlay');

  if (!originalUrl && !overlayUrl) {
    return (
      <div className="glass-elevated aspect-square flex items-center justify-center">
        <div className="text-center">
          <div className="w-20 h-20 mx-auto mb-4 rounded-2xl bg-gradient-to-br from-slate-700/40 to-slate-600/20 flex items-center justify-center">
            <Layers className="w-8 h-8 text-[var(--color-text-muted)]" />
          </div>
          <p className="text-[var(--color-text-muted)] text-sm">
            Upload a scan to view it here
          </p>
        </div>
      </div>
    );
  }

  const currentSrc =
    activeView === 'overlay' && overlayUrl
      ? overlayUrl
      : activeView === 'mask' && maskUrl
      ? maskUrl
      : originalUrl;

  const views = [
    { key: 'original' as const, label: 'Original', available: !!originalUrl },
    { key: 'overlay' as const, label: 'Overlay', available: !!overlayUrl },
    { key: 'mask' as const, label: 'Mask', available: !!maskUrl },
  ];

  return (
    <div className="glass-elevated overflow-hidden animate-fade-in">
      {/* Toolbar */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-[var(--color-border)]">
        <div className="flex gap-1">
          {views.filter(v => v.available).map((v) => (
            <button
              key={v.key}
              onClick={() => setActiveView(v.key)}
              className={`
                px-3 py-1.5 rounded-lg text-xs font-medium transition-all
                ${activeView === v.key
                  ? 'bg-indigo-500/20 text-indigo-300 border border-indigo-500/30'
                  : 'text-[var(--color-text-muted)] hover:text-white hover:bg-white/5'}
              `}
            >
              {v.label}
            </button>
          ))}
        </div>

        <div className="flex items-center gap-1">
          <button
            onClick={() => setZoom((z) => Math.max(0.5, z - 0.25))}
            className="w-8 h-8 rounded-lg flex items-center justify-center text-[var(--color-text-muted)] hover:text-white hover:bg-white/5 transition-all"
          >
            <ZoomOut className="w-4 h-4" />
          </button>
          <span className="text-xs text-[var(--color-text-muted)] w-10 text-center font-mono">
            {Math.round(zoom * 100)}%
          </span>
          <button
            onClick={() => setZoom((z) => Math.min(3, z + 0.25))}
            className="w-8 h-8 rounded-lg flex items-center justify-center text-[var(--color-text-muted)] hover:text-white hover:bg-white/5 transition-all"
          >
            <ZoomIn className="w-4 h-4" />
          </button>
          <button
            onClick={() => setRotation((r) => r + 90)}
            className="w-8 h-8 rounded-lg flex items-center justify-center text-[var(--color-text-muted)] hover:text-white hover:bg-white/5 transition-all"
          >
            <RotateCw className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Canvas */}
      <div className="relative aspect-square bg-black/40 flex items-center justify-center overflow-hidden">
        {currentSrc && (
          <img
            src={currentSrc}
            alt="CT brain scan"
            className="max-w-full max-h-full object-contain transition-transform duration-300"
            style={{
              transform: `scale(${zoom}) rotate(${rotation}deg)`,
            }}
            draggable={false}
          />
        )}
      </div>
    </div>
  );
}
