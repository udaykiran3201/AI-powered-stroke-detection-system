/* Stroke Detection System — Diagnosis Results Panel */

import { Brain, Clock, Target, TrendingUp, Percent } from 'lucide-react';
import type { DiagnosisReport } from '../types';

interface Props {
  report: DiagnosisReport | null;
}

const strokeTypeLabels: Record<string, string> = {
  hemorrhagic: 'Hemorrhagic Stroke',
  ischemic: 'Ischemic Stroke',
  none: 'No Stroke Detected',
};

const subtypeLabels: Record<string, string> = {
  epidural: 'Epidural',
  intraparenchymal: 'Intraparenchymal',
  intraventricular: 'Intraventricular',
  subarachnoid: 'Subarachnoid',
  subdural: 'Subdural',
  ischemic: 'Ischemic',
};

export default function DiagnosisResults({ report }: Props) {
  if (!report) {
    return (
      <div className="glass-elevated p-8 text-center">
        <Brain className="w-12 h-12 mx-auto mb-4 text-[var(--color-text-muted)] opacity-40" />
        <p className="text-[var(--color-text-muted)]">
          Analysis results will appear here after diagnosis
        </p>
      </div>
    );
  }

  const { classification, segmentation } = report;
  const sortedProbs = Object.entries(classification.subtype_probabilities)
    .sort((a, b) => b[1] - a[1]);

  return (
    <div className="space-y-4 animate-slide-up">
      {/* Primary result */}
      <div className="glass-elevated p-6">
        <div className="flex items-center gap-3 mb-5">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-500/20 to-cyan-500/20 flex items-center justify-center">
            <Brain className="w-5 h-5 text-indigo-400" />
          </div>
          <div>
            <h3 className="font-semibold">Classification</h3>
            <p className="text-xs text-[var(--color-text-muted)]">
              <Clock className="w-3 h-3 inline mr-1" />
              {classification.inference_time_ms.toFixed(1)} ms
            </p>
          </div>
        </div>

        <div className="flex items-center justify-between mb-5">
          <div>
            <p className="text-2xl font-bold bg-gradient-to-r from-indigo-400 to-cyan-400 bg-clip-text text-transparent">
              {strokeTypeLabels[classification.stroke_type] || classification.stroke_type}
            </p>
            <p className="text-sm text-[var(--color-text-muted)] mt-0.5">
              Severity: <span className={`font-semibold ${
                classification.severity === 'critical' ? 'text-red-400' :
                classification.severity === 'high' ? 'text-orange-400' :
                classification.severity === 'moderate' ? 'text-yellow-400' :
                'text-emerald-400'
              }`}>{classification.severity.toUpperCase()}</span>
            </p>
          </div>
          <div className="text-right">
            <p className="text-3xl font-bold font-mono text-white">
              {(classification.confidence * 100).toFixed(1)}%
            </p>
            <p className="text-xs text-[var(--color-text-muted)]">Confidence</p>
          </div>
        </div>

        {/* Probability bars */}
        <div className="space-y-2.5">
          {sortedProbs.map(([key, prob]) => (
            <div key={key}>
              <div className="flex justify-between items-center mb-1">
                <span className="text-xs font-medium text-[var(--color-text-muted)]">
                  {subtypeLabels[key] || key}
                </span>
                <span className="text-xs font-mono text-[var(--color-text-muted)]">
                  {(prob * 100).toFixed(1)}%
                </span>
              </div>
              <div className="h-1.5 rounded-full bg-white/5 overflow-hidden">
                <div
                  className="h-full rounded-full transition-all duration-700 ease-out"
                  style={{
                    width: `${Math.max(prob * 100, 1)}%`,
                    background: prob > 0.7
                      ? 'linear-gradient(90deg, #ef4444, #f97316)'
                      : prob > 0.4
                      ? 'linear-gradient(90deg, #f59e0b, #eab308)'
                      : 'linear-gradient(90deg, #6366f1, #06b6d4)',
                  }}
                />
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Segmentation card */}
      <div className="glass-elevated p-6">
        <div className="flex items-center gap-3 mb-4">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-cyan-500/20 to-emerald-500/20 flex items-center justify-center">
            <Target className="w-5 h-5 text-cyan-400" />
          </div>
          <div>
            <h3 className="font-semibold">Segmentation</h3>
            <p className="text-xs text-[var(--color-text-muted)]">
              <Clock className="w-3 h-3 inline mr-1" />
              {segmentation.inference_time_ms.toFixed(1)} ms
            </p>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div className="p-4 rounded-xl bg-white/[0.03] border border-[var(--color-border)]">
            <Percent className="w-5 h-5 text-cyan-400 mb-2" />
            <p className="text-2xl font-bold font-mono">
              {segmentation.lesion_area_percentage.toFixed(1)}%
            </p>
            <p className="text-xs text-[var(--color-text-muted)] mt-1">Affected Area</p>
          </div>
          <div className="p-4 rounded-xl bg-white/[0.03] border border-[var(--color-border)]">
            <TrendingUp className="w-5 h-5 text-indigo-400 mb-2" />
            <p className="text-2xl font-bold font-mono">
              {(classification.inference_time_ms + segmentation.inference_time_ms).toFixed(0)}ms
            </p>
            <p className="text-xs text-[var(--color-text-muted)] mt-1">Total Inference</p>
          </div>
        </div>
      </div>
    </div>
  );
}
