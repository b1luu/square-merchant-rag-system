import type { AnswerResponse, HealthResponse } from './types';

const DEFAULT_API_BASE_URL = 'http://127.0.0.1:8000';
const API_BASE_URL = (
  import.meta.env.VITE_RAG_API_BASE_URL?.trim() || DEFAULT_API_BASE_URL
).replace(/\/$/, '');

function buildUrl(path: string): string {
  return `${API_BASE_URL}${path}`;
}

async function parseError(response: Response): Promise<string> {
  const contentType = response.headers.get('content-type') || '';

  if (contentType.includes('application/json')) {
    const payload = (await response.json()) as { error?: string };
    if (payload.error) {
      return payload.error;
    }
  }

  const text = await response.text();
  return text || `${response.status} ${response.statusText}`;
}

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(buildUrl(path), {
    ...init,
    headers: {
      'Content-Type': 'application/json',
      ...(init?.headers || {}),
    },
  });

  if (!response.ok) {
    throw new Error(await parseError(response));
  }

  return (await response.json()) as T;
}

export function getApiBaseUrl(): string {
  return API_BASE_URL;
}

export function fetchHealth(): Promise<HealthResponse> {
  return requestJson<HealthResponse>('/health');
}

export function answerQuery(
  query: string,
  options?: { allowLowConfidence?: boolean; topK?: number }
): Promise<AnswerResponse> {
  return requestJson<AnswerResponse>('/answer', {
    method: 'POST',
    body: JSON.stringify({
      query,
      top_k: options?.topK ?? 3,
      allow_low_confidence: options?.allowLowConfidence ?? false,
    }),
  });
}
