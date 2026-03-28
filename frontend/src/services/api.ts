/* Stroke Detection System — API Service */

import axios from 'axios';
import type { DiagnosisReport, HealthResponse, UploadResponse } from '../types';

const BASE_URL = import.meta.env.VITE_API_BASE_URL || '';

const api = axios.create({
  baseURL: `${BASE_URL}/api/v1`,
  timeout: 120_000, 
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
