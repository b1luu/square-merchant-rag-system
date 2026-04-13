import { useEffect, useRef } from 'react';
import type { Message } from '../types';

interface MessageListProps {
  messages: Message[];
  isLoading: boolean;
}

export default function MessageList({ messages, isLoading }: MessageListProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  if (messages.length === 0) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <p className="text-neutral-400 text-sm">Start a conversation</p>
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-y-auto">
      <div className="max-w-2xl mx-auto px-6 py-6 space-y-6">
        {messages.map((msg) => (
          <div key={msg.id}>
            <div className="mb-1">
              <span className="text-xs font-medium text-neutral-400 uppercase tracking-wide">
                {msg.role === 'user' ? 'You' : 'Assistant'}
              </span>
            </div>
            <div className="text-sm text-neutral-800 leading-relaxed whitespace-pre-wrap">
              {msg.content}
            </div>
          </div>
        ))}

        {isLoading && (
          <div>
            <div className="mb-1">
              <span className="text-xs font-medium text-neutral-400 uppercase tracking-wide">
                Assistant
              </span>
            </div>
            <div className="flex items-center gap-1 py-1">
              <span className="w-1.5 h-1.5 bg-neutral-300 rounded-full animate-pulse" />
              <span className="w-1.5 h-1.5 bg-neutral-300 rounded-full animate-pulse [animation-delay:150ms]" />
              <span className="w-1.5 h-1.5 bg-neutral-300 rounded-full animate-pulse [animation-delay:300ms]" />
            </div>
          </div>
        )}

        <div ref={bottomRef} />
      </div>
    </div>
  );
}
