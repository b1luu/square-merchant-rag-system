import type { Chat } from '../types';

interface SidebarProps {
  chats: Chat[];
  activeChatId: string;
  onSelectChat: (id: string) => void;
  onNewChat: () => void;
}

export default function Sidebar({ chats, activeChatId, onSelectChat, onNewChat }: SidebarProps) {
  return (
    <aside className="flex w-full shrink-0 flex-col border-b border-slate-800 bg-slate-950 text-stone-100 md:h-full md:w-72 md:border-b-0 md:border-r">
      <div className="border-b border-slate-800 px-4 py-5">
        <p className="text-xs font-semibold uppercase tracking-[0.18em] text-emerald-300/80">
          Staff assistant
        </p>
        <h1 className="mt-1 text-xl font-semibold tracking-tight">Mosa Ops</h1>
        <p className="mt-2 text-sm leading-relaxed text-slate-300">
          Recipes, policies, and procedures with confidence-aware retrieval.
        </p>
      </div>

      <div className="px-3 pb-2 pt-3">
        <button
          type="button"
          onClick={onNewChat}
          className="w-full rounded-2xl border border-emerald-400/30 bg-emerald-300/10 px-4 py-3 text-left text-sm font-medium text-emerald-100 transition hover:border-emerald-300/60 hover:bg-emerald-300/15"
        >
          + New lookup
        </button>
      </div>

      <nav className="flex gap-2 overflow-x-auto px-3 pb-4 pt-2 md:flex-1 md:flex-col md:overflow-y-auto md:overflow-x-hidden md:pb-3">
        {chats.map((chat) => (
          <button
            key={chat.id}
            type="button"
            onClick={() => onSelectChat(chat.id)}
            className={`min-w-[13rem] rounded-2xl border px-4 py-3 text-left text-sm transition md:min-w-0 ${
              chat.id === activeChatId
                ? 'border-emerald-300/40 bg-emerald-300/14 text-white shadow-[0_10px_30px_rgba(16,185,129,0.08)]'
                : 'border-slate-800 bg-slate-900/80 text-slate-300 hover:border-slate-700 hover:bg-slate-900 hover:text-stone-100'
            }`}
          >
            <div className="truncate font-medium">{chat.title}</div>
            <div className="mt-1 truncate text-xs text-slate-400">
              {chat.messages.length} message{chat.messages.length === 1 ? '' : 's'}
            </div>
          </button>
        ))}
      </nav>

      <div className="border-t border-slate-800 px-4 py-3">
        <p className="text-xs uppercase tracking-[0.14em] text-slate-400">
          Retrieval first
        </p>
        <p className="mt-1 text-sm text-slate-300">
          Low-confidence answers abstain until someone explicitly overrides them.
        </p>
      </div>
    </aside>
  );
}
