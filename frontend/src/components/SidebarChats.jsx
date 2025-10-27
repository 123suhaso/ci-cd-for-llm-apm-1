import React from 'react';

export default function SidebarChats({ chats = [], activeId, onSelect }) {
  return (
    <div>
      {chats.length === 0 && (
        <div style={{ color: '#666' }}>No chats yet. Click New chat to start.</div>
      )}
      {chats.map((c) => (
        <div
          key={c.id}
          className={`chat-item ${c.id === activeId ? 'active' : ''}`}
          onClick={() => onSelect(c.id)}
        >
          <div style={{ fontWeight: 600 }}>{c.title}</div>
          <div style={{ fontSize: 12, color: '#666' }}>{c.model || 'model'}</div>
        </div>
      ))}
    </div>
  );
}
