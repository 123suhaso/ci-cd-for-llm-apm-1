import React, { useEffect, useRef, useState } from 'react';

export default function ChatWindow({ chat, onSend }) {
  const [input, setInput] = useState('');
  const [sending, setSending] = useState(false);
  const messagesRef = useRef(null);

  useEffect(() => {
    setInput('');
  }, [chat?.id]);

  useEffect(() => {
    const el = messagesRef.current;
    if (!el) return;
    const t = setTimeout(() => {
      el.scrollTop = el.scrollHeight;
    }, 50);
    return () => clearTimeout(t);
  }, [chat?.messages?.length]);

  async function handleSend() {
    const text = String(input ?? '').trim();
    if (!text || sending) return;
    setSending(true);
    try {
      await onSend(text);
      setInput('');
    } catch (err) {
      console.error('send error', err);
    } finally {
      setSending(false);
    }
  }

  function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }

  return (
    <div
      className="chat-window-inner"
      style={{ display: 'flex', flexDirection: 'column', height: '100%' }}
    >
      <div
        className="messages"
        ref={messagesRef}
        style={{
          flex: 1,
          overflow: 'auto',
          padding: '8px 12px',
          display: 'flex',
          flexDirection: 'column',
          gap: 14,
        }}
      >
        {chat?.messages?.length ? (
          chat.messages.map((m, idx) => (
            <div
              key={idx}
              className={`chat-message ${m.sender === 'user' ? 'user' : 'bot'}`}
            >
              <div className={`chat-bubble ${m.sender === 'user' ? 'user' : 'bot'}`}>
                {m.text}
              </div>
            </div>
          ))
        ) : (
          <div className="chat-empty" style={{ padding: 20 }}>
            Select or create a chat to begin.
          </div>
        )}
      </div>

      <div className="chat-input-area" style={{ marginTop: 12 }}>
        <textarea
          className="prompt-input"
          placeholder="Type a prompt..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={sending}
          rows={1}
          style={{ resize: 'none' }}
        />
        <button className="send-btn" onClick={handleSend} disabled={sending}>
          {sending ? 'Sending...' : 'Send'}
        </button>
      </div>
    </div>
  );
}
