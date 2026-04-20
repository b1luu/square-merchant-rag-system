export interface RetrievalConfidence {
  score: number;
  level: 'low' | 'medium' | 'high';
  should_abstain: boolean;
  top_score: number;
  score_margin: number;
  supporting_hits: number;
  anchor_ratio: number;
  reasons: string[];
}

export interface RetrievedRecord {
  rank: number;
  score: number;
  id: string;
  title: string;
  type: string;
  source_file: string;
  source_page: number;
  retrieval_text: string;
}

export interface AnswerVerification {
  supported: boolean;
  coverage_score: number;
  checked_sentences: number;
  unsupported_sentences: string[];
}

export interface HealthResponse {
  status: string;
  records: number;
  jsonl: string;
  embedding_model: string;
  ollama_provider: string | null;
  ollama_model: string;
  uptime_seconds: number;
}

export interface AnswerResponse {
  query: string;
  top_k: number;
  raw_query: boolean;
  latency_ms: number;
  answer: string;
  abstained: boolean;
  verification?: AnswerVerification;
  retrieval_confidence: RetrievalConfidence;
  results: RetrievedRecord[];
}

export interface AssistantMeta {
  query: string;
  abstained: boolean;
  latencyMs: number;
  confidence: RetrievalConfidence;
  verification?: AnswerVerification;
  results: RetrievedRecord[];
}

interface BaseMessage {
  id: string;
  content: string;
  createdAt: string;
}

export interface UserMessage extends BaseMessage {
  role: 'user';
}

export interface AssistantMessage extends BaseMessage {
  role: 'assistant';
  error?: string;
  meta?: AssistantMeta;
  retryableLowConfidence?: boolean;
}

export type Message = UserMessage | AssistantMessage;

export interface Chat {
  id: string;
  title: string;
  messages: Message[];
}

export interface PendingRequest {
  chatId: string;
  query: string;
  allowLowConfidence: boolean;
}

export type HealthState =
  | { kind: 'loading' }
  | { kind: 'connected'; data: HealthResponse }
  | { kind: 'error'; error: string };
