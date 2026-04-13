import type { Chat } from './types';

let nextId = 100;
export function uid(): string {
  return String(nextId++);
}

export const initialChats: Chat[] = [
  {
    id: '1',
    title: 'New chat',
    messages: [],
  },
];
