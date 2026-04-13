import { useState, useCallback } from 'react';
import Sidebar from './components/Sidebar';
import ChatHeader from './components/ChatHeader';
import MessageList from './components/MessageList';
import MessageInput from './components/MessageInput';
import type { Chat } from './types';
import { initialChats, uid } from './data';
import { sendMessage } from './api';

export default function App() {
  const [chats, setChats] = useState<Chat[]>(initialChats);
  const [activeChatId, setActiveChatId] = useState(initialChats[0].id);
  const [isLoading, setIsLoading] = useState(false);

  const activeChat = chats.find((c) => c.id === activeChatId)!;

  const handleNewChat = useCallback(() => {
    const newChat: Chat = {
      id: uid(),
      title: 'New chat',
      messages: [],
    };
    setChats((prev) => [newChat, ...prev]);
    setActiveChatId(newChat.id);
  }, []);

  const handleSend = useCallback(
    async (text: string) => {
      const userMsg = { id: uid(), role: 'user' as const, content: text };
      const chatId = activeChatId;

      setChats((prev) =>
        prev.map((chat) => {
          if (chat.id !== chatId) return chat;
          return {
            ...chat,
            messages: [...chat.messages, userMsg],
            title: chat.messages.length === 0 ? text.slice(0, 30) : chat.title,
          };
        })
      );

      setIsLoading(true);

      try {
        const response = await sendMessage(text);
        const assistantMsg = {
          id: uid(),
          role: 'assistant' as const,
          content: response,
        };
        setChats((prev) =>
          prev.map((chat) =>
            chat.id === chatId
              ? { ...chat, messages: [...chat.messages, assistantMsg] }
              : chat
          )
        );
      } catch (err) {
        const errorMsg = {
          id: uid(),
          role: 'assistant' as const,
          content: `Error: ${err instanceof Error ? err.message : 'Failed to get response. Is the server running?'}`,
        };
        setChats((prev) =>
          prev.map((chat) =>
            chat.id === chatId
              ? { ...chat, messages: [...chat.messages, errorMsg] }
              : chat
          )
        );
      } finally {
        setIsLoading(false);
      }
    },
    [activeChatId]
  );

  return (
    <div className="flex h-screen bg-white text-neutral-800 antialiased">
      <Sidebar
        chats={chats}
        activeChatId={activeChatId}
        onSelectChat={setActiveChatId}
        onNewChat={handleNewChat}
      />
      <main className="flex-1 flex flex-col min-w-0">
        <ChatHeader title={activeChat.title} />
        <MessageList messages={activeChat.messages} isLoading={isLoading} />
        <MessageInput onSend={handleSend} disabled={isLoading} />
      </main>
    </div>
  );
}
