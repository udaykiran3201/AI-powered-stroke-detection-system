/* Stroke Detection System — API Service */

import axios from 'axios';
import type { DiagnosisReport, HealthResponse, UploadResponse } from '../types';

const api = axios.create({
  baseURL: '/api/v1',
  timeout: 120_000, // 2 min — model inference can be slow
});

export async function checkHealth(): Promise<HealthResponse> {
  const { data } = await api.get<HealthResponse>('/health');
  return data;
}

export async function uploadScan(file: File): Promise<UploadResponse> {
  const form = new FormData();
  form.append('file', file);
  const { data } = await api.post<UploadResponse>('/upload/', form);
  return data;
}

export async function runFullDiagnosis(file: File): Promise<DiagnosisReport> {
  const form = new FormData();
  form.append('file', file);
  const { data } = await api.post<DiagnosisReport>('/diagnosis/full', form);
  return data;
}

export default api;
