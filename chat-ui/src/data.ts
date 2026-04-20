import type {
  AnswerResponse,
  AssistantMessage,
  Chat,
  UserMessage,
} from './types';

const TITLE_LIMIT = 36;
let nextId = 1;

export function uid(prefix = 'id'): string {
  nextId += 1;
  return `${prefix}-${nextId}`;
}

export function deriveChatTitle(text: string): string {
  const normalized = text.trim().replace(/\s+/g, ' ');
  if (!normalized) {
    return 'New lookup';
  }
  if (normalized.length <= TITLE_LIMIT) {
    return normalized;
  }
  return `${normalized.slice(0, TITLE_LIMIT - 1).trimEnd()}…`;
}

export function createWelcomeChat(): Chat {
  return {
    id: uid('chat'),
    title: 'Mosa Ops lookup',
    messages: [
      {
        id: uid('assistant'),
        role: 'assistant',
        content:
          'Ask about recipes, policies, or procedures. Each answer will show retrieval confidence and the records it used.',
        createdAt: new Date().toISOString(),
      },
    ],
  };
}

export function createChat(): Chat {
  return {
    id: uid('chat'),
    title: 'New lookup',
    messages: [],
  };
}

export function createUserMessage(content: string): UserMessage {
  return {
    id: uid('user'),
    role: 'user',
    content,
    createdAt: new Date().toISOString(),
  };
}

export function createAssistantMessage(response: AnswerResponse): AssistantMessage {
  return {
    id: uid('assistant'),
    role: 'assistant',
    content: response.answer,
    createdAt: new Date().toISOString(),
    meta: {
      query: response.query,
      abstained: response.abstained,
      answerMode: response.answer_mode,
      latencyMs: response.latency_ms,
      confidence: response.retrieval_confidence,
      verification: response.verification,
      validation: response.validation,
      results: response.results,
    },
    retryableLowConfidence: response.abstained,
  };
}

export function createErrorMessage(message: string): AssistantMessage {
  return {
    id: uid('assistant'),
    role: 'assistant',
    content:
      'The UI could not reach the RAG server. Start `python serve_mosa_rag.py` and verify the API base URL.',
    createdAt: new Date().toISOString(),
    error: message,
  };
}
