import { useEffect, useRef, useState } from 'react';
import type { KeyboardEvent } from 'react';

interface MessageInputProps {
  onSend: (text: string) => void;
  disabled: boolean;
}

export default function MessageInput({ onSend, disabled }: MessageInputProps) {
  const [value, setValue] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (!disabled) {
      textareaRef.current?.focus();
    }
  }, [disabled]);

  // Auto-resize textarea
  useEffect(() => {
    const el = textareaRef.current;
    if (el) {
      el.style.height = 'auto';
      el.style.height = Math.min(el.scrollHeight, 160) + 'px';
    }
  }, [value]);

  function handleSubmit() {
    const trimmed = value.trim();
    if (!trimmed || disabled) return;
    onSend(trimmed);
    setValue('');
  }

  function handleKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  }

  const canSend = value.trim().length > 0 && !disabled;

  return (
    <div className="shrink-0 border-t border-stone-200/80 bg-stone-50/95 px-4 py-4 backdrop-blur md:px-8">
      <div className="mx-auto max-w-4xl">
        <div className="rounded-[28px] border border-stone-300 bg-white px-4 py-3 shadow-[0_14px_40px_rgba(15,23,42,0.06)] transition focus-within:border-emerald-400/70 focus-within:shadow-[0_16px_48px_rgba(16,185,129,0.08)] md:px-5">
          <div className="flex items-end gap-3">
          <textarea
            ref={textareaRef}
            value={value}
            onChange={(e) => setValue(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask about recipes, policies, or procedures…"
            disabled={disabled}
            rows={1}
            className="min-h-[30px] flex-1 resize-none bg-transparent text-[15px] leading-relaxed text-slate-900 outline-none placeholder:text-stone-400"
          />
          <button
            type="button"
            onClick={handleSubmit}
            disabled={!canSend}
            className={`shrink-0 rounded-2xl px-4 py-2 text-sm font-medium transition ${
              canSend
                ? 'bg-emerald-900 text-emerald-50 hover:bg-emerald-800'
                : 'cursor-default bg-stone-200 text-stone-400'
            }`}
            aria-label="Send query"
          >
            Send
          </button>
          </div>

          <div className="mt-3 flex flex-wrap items-center justify-between gap-2 border-t border-stone-100 pt-3 text-xs text-stone-500">
            <span>Enter to send. Shift+Enter for a newline.</span>
            <span>{disabled ? 'Waiting for the current lookup to finish…' : 'Answers show confidence and sources.'}</span>
          </div>
        </div>
      </div>
    </div>
  );
}
