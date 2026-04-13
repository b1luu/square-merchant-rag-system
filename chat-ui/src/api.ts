const API_BASE = '/api';

export async function sendMessage(message: string): Promise<string> {
  const res = await fetch(`${API_BASE}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message }),
  });

  if (!res.ok) {
    const detail = await res.text().catch(() => 'Unknown error');
    throw new Error(`Server error (${res.status}): ${detail}`);
  }

  const data = await res.json();
  return data.response;
}
