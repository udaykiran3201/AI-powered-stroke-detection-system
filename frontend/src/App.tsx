/* Stroke Detection System — Main Application */

import { useState, useCallback } from 'react';
import { Toaster, toast } from 'react-hot-toast';
import { Zap, RotateCcw, ShieldCheck } from 'lucide-react';

import Header from './components/Header';
import UploadZone from './components/UploadZone';
import ScanViewer from './components/ScanViewer';
import DiagnosisResults from './components/DiagnosisResults';
import EmergencyAlert from './components/EmergencyAlert';
import LoadingSpinner from './components/LoadingSpinner';
import { runFullDiagnosis, BASE_URL } from './services/api';
import type { DiagnosisReport } from './types';

export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [report, setReport] = useState<DiagnosisReport | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleFileSelect = useCallback((selected: File) => {
    setFile(selected);
    setReport(null);
    // Create local preview for standard images
    const ext = selected.name.split('.').pop()?.toLowerCase();
    if (ext && ['png', 'jpg', 'jpeg', 'tif', 'tiff'].includes(ext)) {
      setPreviewUrl(URL.createObjectURL(selected));
    } else {
      setPreviewUrl(null);
    }
  }, []);

  const handleClear = useCallback(() => {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setFile(null);
    setPreviewUrl(null);
    setReport(null);
  }, [previewUrl]);

  const handleDiagnose = async () => {
    if (!file) return;
    setIsLoading(true);
    setReport(null);

    try {
      const result = await runFullDiagnosis(file);
      setReport(result);

      if (result.emergency_alert) {
        toast.error('⚠️ Emergency: Stroke detected — immediate action required', {
          duration: 8000,
          style: {
            background: '#1e293b',
            color: '#f1f5f9',
            border: '1px solid rgba(239,68,68,0.4)',
          },
        });
      } else if (result.classification.stroke_type === 'none') {
        toast.success('No stroke detected — scan appears normal', {
          style: {
            background: '#1e293b',
            color: '#f1f5f9',
            border: '1px solid rgba(16,185,129,0.3)',
          },
        });
      }
    } catch (err: any) {
      const msg = err?.response?.data?.detail || err?.message || 'Diagnosis failed';
      toast.error(msg, {
        style: {
          background: '#1e293b',
          color: '#f1f5f9',
          border: '1px solid rgba(239,68,68,0.3)',
        },
      });
    } finally {
      setIsLoading(false);
    }
  };

  const overlayUrl = report?.segmentation.overlay_url 
    ? (report.segmentation.overlay_url.startsWith('/') ? `${BASE_URL}${report.segmentation.overlay_url}` : report.segmentation.overlay_url) 
    : null;
    
  const maskUrl = report?.segmentation.mask_url 
    ? (report.segmentation.mask_url.startsWith('/') ? `${BASE_URL}${report.segmentation.mask_url}` : report.segmentation.mask_url) 
    : null;

  return (
    <div className="min-h-screen flex flex-col">
      <Toaster position="top-right" />
      <Header />

      {/* Hero section */}
      <div className="relative overflow-hidden">
        {/* Background gradient blobs */}
        <div className="absolute inset-0 -z-10">
          <div className="absolute top-0 left-1/4 w-96 h-96 bg-indigo-500/10 rounded-full blur-[120px]" />
          <div className="absolute bottom-0 right-1/4 w-80 h-80 bg-cyan-500/10 rounded-full blur-[100px]" />
        </div>

        <main className="max-w-7xl mx-auto px-6 py-8">
          {/* Title area */}
          <div className="text-center mb-10 animate-fade-in">
            <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-indigo-500/10 border border-indigo-500/20 mb-4">
              <ShieldCheck className="w-3.5 h-3.5 text-indigo-400" />
              <span className="text-xs font-medium text-indigo-300">
                AI-Powered Diagnostic Assistance
              </span>
            </div>
            <h2 className="text-3xl sm:text-4xl font-bold tracking-tight mb-3">
              <span className="bg-gradient-to-r from-white via-slate-200 to-slate-400 bg-clip-text text-transparent">
                Rapid Stroke Detection
              </span>
            </h2>
            <p className="text-[var(--color-text-muted)] max-w-xl mx-auto text-sm leading-relaxed">
              Upload a CT brain scan for instant AI analysis. Our system identifies hemorrhagic
              and ischemic strokes, segments affected regions, and generates emergency alerts.
            </p>
          </div>

          {/* Emergency Alert */}
          {report && (
            <div className="mb-6">
              <EmergencyAlert
                severity={report.classification.severity}
                recommendation={report.recommendation}
                visible={report.emergency_alert}
              />
            </div>
          )}

          {/* Main grid */}
          <div className="grid lg:grid-cols-2 gap-6 mb-6">
            {/* Left — Viewer */}
            <div>
              <ScanViewer
                originalUrl={previewUrl}
                overlayUrl={overlayUrl}
                maskUrl={maskUrl}
              />
            </div>

            {/* Right — Results or upload */}
            <div className="flex flex-col gap-6">
              <UploadZone
                onFileSelect={handleFileSelect}
                selectedFile={file}
                onClear={handleClear}
                isLoading={isLoading}
              />

              {isLoading ? (
                <LoadingSpinner />
              ) : (
                <DiagnosisResults report={report} />
              )}
            </div>
          </div>

          {/* Action buttons */}
          <div className="flex flex-wrap items-center justify-center gap-4 mb-10">
            <button
              onClick={handleDiagnose}
              disabled={!file || isLoading}
              className={`
                inline-flex items-center gap-2.5 px-8 py-3.5 rounded-xl font-semibold text-sm
                transition-all duration-200
                ${!file || isLoading
                  ? 'bg-slate-700/40 text-slate-500 cursor-not-allowed'
                  : 'bg-gradient-to-r from-indigo-500 to-cyan-500 text-white shadow-lg shadow-indigo-500/25 hover:shadow-indigo-500/40 hover:scale-[1.02] active:scale-[0.98]'
                }
              `}
            >
              <Zap className="w-4 h-4" />
              {isLoading ? 'Analysing…' : 'Run Diagnosis'}
            </button>

            {report && (
              <button
                onClick={handleClear}
                className="inline-flex items-center gap-2 px-6 py-3.5 rounded-xl font-semibold text-sm
                  border border-[var(--color-border)] text-[var(--color-text-muted)]
                  hover:text-white hover:bg-white/5 transition-all"
              >
                <RotateCcw className="w-4 h-4" />
                New Scan
              </button>
            )}
          </div>

          {/* Info footer */}
          {!report && !isLoading && (
            <div className="grid sm:grid-cols-3 gap-4 max-w-3xl mx-auto animate-fade-in">
              {[
                { title: 'Fast Detection', desc: 'Results in under 5 seconds', icon: '⚡' },
                { title: 'Multi-class', desc: '6 hemorrhage subtypes + ischemic', icon: '🧠' },
                { title: 'Region Mapping', desc: 'U-Net lesion segmentation', icon: '🎯' },
              ].map((item) => (
                <div
                  key={item.title}
                  className="glass p-5 text-center hover:bg-white/[0.04] transition-colors"
                >
                  <div className="text-2xl mb-2">{item.icon}</div>
                  <p className="font-semibold text-sm mb-1">{item.title}</p>
                  <p className="text-xs text-[var(--color-text-muted)]">{item.desc}</p>
                </div>
              ))}
            </div>
          )}
        </main>
      </div>

      {/* Footer */}
      <footer className="mt-auto border-t border-[var(--color-border)] py-6">
        <p className="text-center text-xs text-[var(--color-text-muted)]">
          NeuroScan AI · For research & educational purposes only · Not a substitute for professional medical diagnosis
        </p>
      </footer>
    </div>
  );
}
