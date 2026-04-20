import { useEffect, useRef } from 'react';
import type { Message, PendingRequest, RetrievedRecord } from '../types';

interface MessageListProps {
  messages: Message[];
  pendingRequest: PendingRequest | null;
  onPromptSelect: (query: string) => void;
  onRetryLowConfidence: (query: string) => void;
}

const STARTER_PROMPTS = [
  'What happens if I am sick?',
  'How do I request time off in Square Team?',
  'On weekends what is the last batch time for boba?',
];

function formatTimestamp(value: string): string {
  return new Date(value).toLocaleTimeString([], {
    hour: 'numeric',
    minute: '2-digit',
  });
}

function confidenceClass(level: 'low' | 'medium' | 'high'): string {
  if (level === 'high') {
    return 'border-emerald-200 bg-emerald-50 text-emerald-800';
  }
  if (level === 'medium') {
    return 'border-amber-200 bg-amber-50 text-amber-800';
  }
  return 'border-rose-200 bg-rose-50 text-rose-700';
}

function verificationClass(supported: boolean): string {
  return supported
    ? 'border-emerald-200 bg-emerald-50 text-emerald-800'
    : 'border-rose-200 bg-rose-50 text-rose-700';
}

function SourceList({ results }: { results: RetrievedRecord[] }) {
  if (!results.length) {
    return null;
  }

  return (
    <div className="mt-4 space-y-2">
      <p className="text-xs font-semibold uppercase tracking-[0.16em] text-stone-500">
        Retrieved records
      </p>
      <div className="grid gap-2">
        {results.map((result) => (
          <div
            key={`${result.id}-${result.rank}`}
            className="rounded-2xl border border-stone-200 bg-white px-4 py-3"
          >
            <div className="flex flex-wrap items-center justify-between gap-2">
              <p className="font-medium text-slate-900">{result.title}</p>
              <span className="text-xs text-stone-500">
                score {result.score.toFixed(4)}
              </span>
            </div>
            <p className="mt-1 text-sm text-stone-600">
              {result.source_file} · page {result.source_page || 0}
            </p>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function MessageList({
  messages,
  pendingRequest,
  onPromptSelect,
  onRetryLowConfidence,
}: MessageListProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' });
  }, [messages, pendingRequest]);

  if (messages.length === 0 && !pendingRequest) {
    return (
      <div className="flex flex-1 items-center justify-center px-6 py-10">
        <div className="max-w-2xl rounded-[32px] border border-stone-200 bg-white p-8 shadow-[0_20px_60px_rgba(15,23,42,0.08)]">
          <p className="text-xs font-semibold uppercase tracking-[0.18em] text-stone-500">
            Start with a real ops question
          </p>
          <h3 className="mt-3 text-3xl font-semibold tracking-tight text-slate-900">
            This UI talks directly to the Mosa RAG server.
          </h3>
          <p className="mt-3 text-base leading-relaxed text-stone-600">
            Answers can abstain when retrieval confidence is too weak. Each completed response
            shows the confidence level, latency, and source records it used.
          </p>

          <div className="mt-6 grid gap-3">
            {STARTER_PROMPTS.map((prompt) => (
              <button
                key={prompt}
                type="button"
                onClick={() => onPromptSelect(prompt)}
                className="rounded-2xl border border-stone-200 bg-stone-50 px-4 py-4 text-left text-sm font-medium text-slate-900 transition hover:border-emerald-300 hover:bg-emerald-50"
              >
                {prompt}
              </button>
            ))}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-y-auto px-4 py-6 md:px-8">
      <div className="mx-auto flex max-w-4xl flex-col gap-5">
        {messages.map((msg) => (
          <div key={msg.id} className={msg.role === 'user' ? 'ml-auto max-w-2xl' : 'max-w-3xl'}>
            <div
              className={
                msg.role === 'user'
                  ? 'rounded-[28px] bg-emerald-950 px-5 py-4 text-emerald-50 shadow-[0_18px_50px_rgba(6,78,59,0.18)]'
                  : 'rounded-[28px] border border-stone-200 bg-white px-5 py-4 text-slate-900 shadow-[0_18px_50px_rgba(15,23,42,0.08)]'
              }
            >
              <div className="mb-3 flex items-center justify-between gap-3">
                <span className="text-xs font-semibold uppercase tracking-[0.16em] text-stone-500">
                  {msg.role === 'user' ? 'You' : 'Assistant'}
                </span>
                <span
                  className={
                    msg.role === 'user'
                      ? 'text-xs text-emerald-200/80'
                      : 'text-xs text-stone-400'
                  }
                >
                  {formatTimestamp(msg.createdAt)}
                </span>
              </div>

              <div
                className={
                  msg.role === 'user'
                    ? 'whitespace-pre-wrap text-[15px] leading-relaxed'
                    : 'whitespace-pre-wrap text-[15px] leading-relaxed text-slate-800'
                }
              >
                {msg.content}
              </div>

              {msg.role === 'assistant' && msg.error ? (
                <div className="mt-4 rounded-2xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700">
                  {msg.error}
                </div>
              ) : null}

              {msg.role === 'assistant' && msg.meta ? (
                <div className="mt-4 rounded-[24px] border border-stone-200 bg-stone-50 px-4 py-4">
                  <div className="flex flex-wrap items-center gap-2">
                    <span
                      className={`inline-flex items-center rounded-full border px-3 py-1 text-xs font-medium ${confidenceClass(
                        msg.meta.confidence.level
                      )}`}
                    >
                      {msg.meta.confidence.level} confidence
                    </span>
                    <span className="inline-flex items-center rounded-full border border-stone-200 bg-white px-3 py-1 text-xs text-stone-600">
                      {msg.meta.latencyMs.toFixed(0)} ms
                    </span>
                    {msg.meta.abstained ? (
                      <span className="inline-flex items-center rounded-full border border-amber-200 bg-amber-50 px-3 py-1 text-xs font-medium text-amber-800">
                        abstained
                      </span>
                    ) : null}
                    {msg.meta.verification ? (
                      <span
                        className={`inline-flex items-center rounded-full border px-3 py-1 text-xs font-medium ${verificationClass(
                          msg.meta.verification.supported
                        )}`}
                      >
                        {msg.meta.verification.supported ? 'verified' : 'unsupported details'}
                      </span>
                    ) : null}
                  </div>

                  {msg.meta.confidence.reasons.length ? (
                    <ul className="mt-4 space-y-2 text-sm text-stone-600">
                      {msg.meta.confidence.reasons.map((reason) => (
                        <li key={reason} className="rounded-2xl bg-white px-4 py-3">
                          {reason}
                        </li>
                      ))}
                    </ul>
                  ) : null}

                  {msg.meta.verification?.unsupported_sentences.length ? (
                    <div className="mt-4 rounded-2xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-800">
                      <p className="font-semibold">Unsupported generated details</p>
                      <ul className="mt-2 space-y-2">
                        {msg.meta.verification.unsupported_sentences.map((sentence) => (
                          <li key={sentence}>{sentence}</li>
                        ))}
                      </ul>
                    </div>
                  ) : null}

                  <SourceList results={msg.meta.results} />

                  {msg.retryableLowConfidence && msg.meta ? (
                    <button
                      type="button"
                      onClick={() => onRetryLowConfidence(msg.meta!.query)}
                      className="mt-4 rounded-2xl border border-slate-300 bg-slate-900 px-4 py-3 text-sm font-medium text-stone-50 transition hover:bg-slate-800"
                    >
                      Retry with low-confidence override
                    </button>
                  ) : null}
                </div>
              ) : null}
            </div>
          </div>
        ))}

        {pendingRequest ? (
          <div className="max-w-3xl">
            <div className="rounded-[28px] border border-stone-200 bg-white px-5 py-4 shadow-[0_18px_50px_rgba(15,23,42,0.06)]">
              <div className="mb-3 flex items-center justify-between gap-3">
                <span className="text-xs font-semibold uppercase tracking-[0.16em] text-stone-500">
                  Assistant
                </span>
                <span className="text-xs text-stone-400">
                  {pendingRequest.allowLowConfidence ? 'override' : 'retrieval first'}
                </span>
              </div>

              <p className="text-sm text-stone-600">
                {pendingRequest.allowLowConfidence
                  ? 'Retrying with low-confidence override enabled.'
                  : 'Searching the corpus and assembling the grounded answer.'}
              </p>
              <p className="mt-2 text-base font-medium text-slate-900">{pendingRequest.query}</p>

              <div className="mt-4 flex items-center gap-2">
                <span className="h-2 w-2 animate-pulse rounded-full bg-emerald-500" />
                <span className="h-2 w-2 animate-pulse rounded-full bg-emerald-400 [animation-delay:140ms]" />
                <span className="h-2 w-2 animate-pulse rounded-full bg-emerald-300 [animation-delay:280ms]" />
              </div>
            </div>
          </div>
        ) : null}

        <div ref={bottomRef} />
      </div>
    </div>
  );
}
