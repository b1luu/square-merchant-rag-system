import { startTransition, useEffect, useRef, useState } from 'react';
import { answerQuery, fetchHealth, getApiBaseUrl } from './api';
import Sidebar from './components/Sidebar';
import ChatHeader from './components/ChatHeader';
import MessageList from './components/MessageList';
import MessageInput from './components/MessageInput';
import {
  createAssistantMessage,
  createChat,
  createErrorMessage,
  createUserMessage,
  createWelcomeChat,
  deriveChatTitle,
} from './data';
import type { Chat, HealthState, PendingRequest } from './types';

export default function App() {
  const initialChatRef = useRef<Chat>(createWelcomeChat());
  const [chats, setChats] = useState<Chat[]>(() => [initialChatRef.current]);
  const [activeChatId, setActiveChatId] = useState<string>(() => initialChatRef.current.id);
  const [pendingRequest, setPendingRequest] = useState<PendingRequest | null>(null);
  const [health, setHealth] = useState<HealthState>({ kind: 'loading' });

  const activeChat = chats.find((c) => c.id === activeChatId)!;
  const activePendingRequest = pendingRequest?.chatId === activeChatId ? pendingRequest : null;

  useEffect(() => {
    void refreshHealth();
  }, []);

  async function refreshHealth() {
    try {
      const data = await fetchHealth();
      setHealth({ kind: 'connected', data });
    } catch (error) {
      setHealth({
        kind: 'error',
        error: error instanceof Error ? error.message : 'Unknown error while checking the server.',
      });
    }
  }

  function updateChat(chatId: string, updater: (chat: Chat) => Chat) {
    startTransition(() => {
      setChats((prev) => prev.map((chat) => (chat.id === chatId ? updater(chat) : chat)));
    });
  }

  function handleNewChat() {
    const newChat: Chat = {
      ...createChat(),
      messages: [],
    };
    startTransition(() => {
      setChats((prev) => [newChat, ...prev]);
      setActiveChatId(newChat.id);
    });
  }

  async function runQuery(chatId: string, query: string, options?: { allowLowConfidence?: boolean; appendUserMessage?: boolean }) {
    const allowLowConfidence = options?.allowLowConfidence ?? false;
    const appendUserMessage = options?.appendUserMessage ?? true;

    if (pendingRequest) {
      return;
    }

    if (appendUserMessage) {
      const userMessage = createUserMessage(query);
      updateChat(chatId, (chat) => {
        const hasUserMessages = chat.messages.some((message) => message.role === 'user');
        return {
          ...chat,
          title: hasUserMessages ? chat.title : deriveChatTitle(query),
          messages: [...chat.messages, userMessage],
        };
      });
    }

    setPendingRequest({ chatId, query, allowLowConfidence });

    try {
      const response = await answerQuery(query, {
        allowLowConfidence,
      });
      updateChat(chatId, (chat) => ({
        ...chat,
        messages: [...chat.messages, createAssistantMessage(response)],
      }));
    } catch (error) {
      const message =
        error instanceof Error ? error.message : 'Unknown error while contacting the RAG server.';
      updateChat(chatId, (chat) => ({
        ...chat,
        messages: [...chat.messages, createErrorMessage(message)],
      }));
    } finally {
      setPendingRequest((current) => {
        if (
          current &&
          current.chatId === chatId &&
          current.query === query &&
          current.allowLowConfidence === allowLowConfidence
        ) {
          return null;
        }
        return current;
      });
    }
  }

  function handleSend(text: string) {
    void runQuery(activeChatId, text, {
      appendUserMessage: true,
    });
  }

  function handlePromptSelect(text: string) {
    void runQuery(activeChatId, text, {
      appendUserMessage: true,
    });
  }

  function handleRetryLowConfidence(query: string) {
    void runQuery(activeChatId, query, {
      allowLowConfidence: true,
      appendUserMessage: false,
    });
  }

  return (
    <div className="flex h-dvh flex-col bg-stone-100 text-slate-900 antialiased md:flex-row">
      <Sidebar
        chats={chats}
        activeChatId={activeChatId}
        onSelectChat={setActiveChatId}
        onNewChat={handleNewChat}
      />
      <main className="flex min-w-0 flex-1 flex-col bg-stone-100">
        <ChatHeader
          title={activeChat.title}
          apiBaseUrl={getApiBaseUrl()}
          health={health}
          onRefreshHealth={refreshHealth}
        />
        <MessageList
          messages={activeChat.messages}
          pendingRequest={activePendingRequest}
          onPromptSelect={handlePromptSelect}
          onRetryLowConfidence={handleRetryLowConfidence}
        />
        <MessageInput onSend={handleSend} disabled={pendingRequest !== null} />
      </main>
    </div>
  );
}
