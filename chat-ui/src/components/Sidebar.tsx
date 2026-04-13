import type { Chat } from '../types';

interface SidebarProps {
  chats: Chat[];
  activeChatId: string;
  onSelectChat: (id: string) => void;
  onNewChat: () => void;
}

export default function Sidebar({ chats, activeChatId, onSelectChat, onNewChat }: SidebarProps) {
  return (
    <aside className="w-56 shrink-0 border-r border-neutral-200 bg-neutral-50 flex flex-col h-full">
      {/* Logo / App name */}
      <div className="px-4 py-5 border-b border-neutral-200">
        <h1 className="text-base font-semibold text-neutral-800 tracking-tight">Mosa Chat</h1>
      </div>

      {/* New Chat button */}
      <div className="px-3 pt-3 pb-1">
        <button
          onClick={onNewChat}
          className="w-full text-left px-3 py-2 text-sm text-neutral-600 border border-neutral-200 rounded-lg hover:bg-neutral-100 hover:text-neutral-800 transition-colors cursor-pointer"
        >
          + New chat
        </button>
      </div>

      {/* Chat list */}
      <nav className="flex-1 overflow-y-auto px-3 py-2 space-y-0.5">
        {chats.map((chat) => (
          <button
            key={chat.id}
            onClick={() => onSelectChat(chat.id)}
            className={`w-full text-left px-3 py-2 text-sm rounded-lg transition-colors truncate cursor-pointer ${
              chat.id === activeChatId
                ? 'bg-neutral-200 text-neutral-900 font-medium'
                : 'text-neutral-500 hover:bg-neutral-100 hover:text-neutral-700'
            }`}
          >
            {chat.title}
          </button>
        ))}
      </nav>

      {/* Bottom area */}
      <div className="px-4 py-3 border-t border-neutral-200">
        <p className="text-xs text-neutral-400">Ollama · llama3.2</p>
      </div>
    </aside>
  );
}
