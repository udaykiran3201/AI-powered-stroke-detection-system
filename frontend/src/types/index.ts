/* Stroke Detection System — TypeScript Types */

export type StrokeType = 'hemorrhagic' | 'ischemic' | 'none';

export type SeverityLevel = 'critical' | 'high' | 'moderate' | 'low' | 'normal';

export interface UploadResponse {
  scan_id: string;
  filename: string;
  file_size_bytes: number;
  upload_time: string;
  message: string;
}

export interface ClassificationResult {
  scan_id: string;
  stroke_type: StrokeType;
  subtype_probabilities: Record<string, number>;
  confidence: number;
  severity: SeverityLevel;
  inference_time_ms: number;
}

export interface SegmentationResult {
  scan_id: string;
  mask_url: string;
  overlay_url: string;
  lesion_area_percentage: number;
  inference_time_ms: number;
}

export interface DiagnosisReport {
  scan_id: string;
  classification: ClassificationResult;
  segmentation: SegmentationResult;
  emergency_alert: boolean;
  recommendation: string;
  timestamp: string;
}

export interface HealthResponse {
  status: string;
  version: string;
  models_loaded: boolean;
  environment: string;
}
