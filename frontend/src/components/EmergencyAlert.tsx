/* Stroke Detection System — Emergency Alert Banner */

import { AlertTriangle, AlertOctagon } from 'lucide-react';
import type { SeverityLevel } from '../types';

interface Props {
  severity: SeverityLevel;
  recommendation: string;
  visible: boolean;
}

const severityConfig: Record<
  SeverityLevel,
  { label: string; color: string; bg: string; border: string; icon: typeof AlertTriangle; glow: boolean }
> = {
  critical: {
    label: 'CRITICAL EMERGENCY',
    color: 'text-red-300',
    bg: 'bg-red-500/10',
    border: 'border-red-500/40',
    icon: AlertOctagon,
    glow: true,
  },
  high: {
    label: 'HIGH SEVERITY',
    color: 'text-orange-300',
    bg: 'bg-orange-500/10',
    border: 'border-orange-500/30',
    icon: AlertTriangle,
    glow: true,
  },
  moderate: {
    label: 'MODERATE',
    color: 'text-yellow-300',
    bg: 'bg-yellow-500/10',
    border: 'border-yellow-500/20',
    icon: AlertTriangle,
    glow: false,
  },
  low: {
    label: 'LOW SEVERITY',
    color: 'text-blue-300',
    bg: 'bg-blue-500/10',
    border: 'border-blue-500/20',
    icon: AlertTriangle,
    glow: false,
  },
  normal: {
    label: 'NORMAL',
    color: 'text-emerald-300',
    bg: 'bg-emerald-500/10',
    border: 'border-emerald-500/20',
    icon: AlertTriangle,
    glow: false,
  },
};

export default function EmergencyAlert({ severity, recommendation, visible }: Props) {
  if (!visible) return null;

  const config = severityConfig[severity];
  const Icon = config.icon;

  return (
    <div
      className={`
        rounded-2xl border p-5 animate-slide-up
        ${config.bg} ${config.border}
        ${config.glow ? 'animate-pulse-glow' : ''}
      `}
    >
      <div className="flex items-start gap-4">
        <div className={`
          w-12 h-12 rounded-xl flex items-center justify-center shrink-0
          ${severity === 'critical' ? 'bg-red-500/20' : severity === 'high' ? 'bg-orange-500/20' : 'bg-white/5'}
        `}>
          <Icon className={`w-6 h-6 ${config.color}`} />
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-3 mb-2">
            <span className={`text-xs font-bold tracking-widest uppercase ${config.color}`}>
              {config.label}
            </span>
            {config.glow && (
              <span className="relative flex h-2.5 w-2.5">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75" />
                <span className="relative inline-flex h-2.5 w-2.5 rounded-full bg-red-500" />
              </span>
            )}
          </div>
          <p className="text-sm text-[var(--color-text)] leading-relaxed">
            {recommendation}
          </p>
        </div>
      </div>
    </div>
  );
}
