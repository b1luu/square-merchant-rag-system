import type { HealthState } from '../types';

interface ChatHeaderProps {
  title: string;
  apiBaseUrl: string;
  health: HealthState;
  onRefreshHealth: () => void;
}

function renderHealthLabel(health: HealthState): { text: string; className: string } {
  if (health.kind === 'loading') {
    return {
      text: 'Checking server',
      className: 'border-stone-300 bg-stone-100 text-stone-600',
    };
  }

  if (health.kind === 'error') {
    return {
      text: 'Server offline',
      className: 'border-rose-200 bg-rose-50 text-rose-700',
    };
  }

  return {
    text: `${health.data.records} records ready`,
    className: 'border-emerald-200 bg-emerald-50 text-emerald-800',
  };
}

export default function ChatHeader({
  title,
  apiBaseUrl,
  health,
  onRefreshHealth,
}: ChatHeaderProps) {
  const status = renderHealthLabel(health);
  const provider =
    health.kind === 'connected'
      ? health.data.ollama_provider || 'retrieval only'
      : 'server unavailable';

  return (
    <header className="border-b border-stone-200/80 bg-stone-50/90 px-5 py-4 backdrop-blur md:px-8">
      <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
        <div className="min-w-0">
          <p className="text-xs font-semibold uppercase tracking-[0.18em] text-stone-500">
            Mosa Ops RAG
          </p>
          <h2 className="mt-1 truncate text-2xl font-semibold tracking-tight text-slate-900">
            {title}
          </h2>
          <p className="mt-1 text-sm text-stone-600">
            Grounded answers with retrieval confidence and cited source records.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <span
            className={`inline-flex items-center rounded-full border px-3 py-1 text-xs font-medium ${status.className}`}
          >
            {status.text}
          </span>
          <span className="inline-flex items-center rounded-full border border-stone-300 bg-white px-3 py-1 text-xs font-medium text-stone-700">
            {provider}
          </span>
          <span className="inline-flex items-center rounded-full border border-stone-300 bg-white px-3 py-1 text-xs text-stone-600">
            {apiBaseUrl}
          </span>
          <button
            type="button"
            onClick={onRefreshHealth}
            className="inline-flex items-center rounded-full border border-stone-300 bg-white px-3 py-1 text-xs font-medium text-stone-700 transition hover:border-stone-400 hover:text-slate-900"
          >
            Refresh health
          </button>
        </div>
      </div>
    </header>
  );
}
