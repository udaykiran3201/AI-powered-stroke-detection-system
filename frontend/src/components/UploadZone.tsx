/* Stroke Detection System — File Upload / Dropzone */

import { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, FileImage, X } from 'lucide-react';

interface Props {
  onFileSelect: (file: File) => void;
  selectedFile: File | null;
  onClear: () => void;
  isLoading: boolean;
}

const ACCEPTED = {
  'image/png': ['.png'],
  'image/jpeg': ['.jpg', '.jpeg'],
  'image/tiff': ['.tif', '.tiff'],
  'application/dicom': ['.dcm'],
};

export default function UploadZone({ onFileSelect, selectedFile, onClear, isLoading }: Props) {
  const onDrop = useCallback(
    (accepted: File[]) => {
      if (accepted.length > 0) onFileSelect(accepted[0]);
    },
    [onFileSelect]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: ACCEPTED,
    maxFiles: 1,
    disabled: isLoading,
  });

  const formatSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  return (
    <div className="animate-fade-in">
      {!selectedFile ? (
        <div
          {...getRootProps()}
          className={`
            relative group cursor-pointer rounded-2xl border-2 border-dashed
            transition-all duration-300 p-12 text-center
            ${isDragActive
              ? 'border-indigo-400 bg-indigo-500/10 scale-[1.01]'
              : 'border-[var(--color-border)] hover:border-indigo-400/50 hover:bg-white/[0.02]'
            }
          `}
        >
          <input {...getInputProps()} id="ct-scan-upload" />

          <div className="mx-auto w-16 h-16 rounded-2xl bg-gradient-to-br from-indigo-500/20 to-cyan-500/20 flex items-center justify-center mb-5 group-hover:scale-110 transition-transform">
            <Upload className="w-7 h-7 text-indigo-400" />
          </div>

          <p className="text-lg font-semibold text-[var(--color-text)] mb-2">
            {isDragActive ? 'Drop your scan here' : 'Upload CT Brain Scan'}
          </p>
          <p className="text-sm text-[var(--color-text-muted)] max-w-md mx-auto">
            Drag & drop a DICOM, PNG, JPEG, or TIFF file, or click to browse.
            <br />
            <span className="text-xs opacity-70">Maximum file size: 50 MB</span>
          </p>
        </div>
      ) : (
        <div className="glass-elevated p-6 flex items-center gap-4">
          <div className="w-14 h-14 rounded-xl bg-indigo-500/10 flex items-center justify-center shrink-0">
            <FileImage className="w-6 h-6 text-indigo-400" />
          </div>
          <div className="flex-1 min-w-0">
            <p className="font-semibold truncate">{selectedFile.name}</p>
            <p className="text-sm text-[var(--color-text-muted)]">
              {formatSize(selectedFile.size)} · {selectedFile.type || 'application/dicom'}
            </p>
          </div>
          <button
            onClick={(e) => { e.stopPropagation(); onClear(); }}
            disabled={isLoading}
            className="w-9 h-9 rounded-lg flex items-center justify-center text-[var(--color-text-muted)] hover:text-red-400 hover:bg-red-500/10 transition-all disabled:opacity-40"
          >
            <X className="w-5 h-5" />
          </button>
        </div>
      )}
    </div>
  );
}
