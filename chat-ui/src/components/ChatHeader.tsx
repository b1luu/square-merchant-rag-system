interface ChatHeaderProps {
  title: string;
}

export default function ChatHeader({ title }: ChatHeaderProps) {
  return (
    <header className="h-14 shrink-0 border-b border-neutral-200 flex items-center px-6">
      <h2 className="text-sm font-medium text-neutral-700 truncate">{title}</h2>
    </header>
  );
}
