import React, { useEffect, useState } from 'react';

export default function Modal({ open, title, children, onClose, expiryMinutes }) {
  const [remainingMs, setRemainingMs] = useState(null);
  const [expiryTs, setExpiryTs] = useState(null);

  useEffect(() => {
    if (!open) {
      setRemainingMs(null);
      setExpiryTs(null);
      return;
    }
    if (!expiryMinutes) return;

    const now = Date.now();
    const expiry = now + parseInt(expiryMinutes, 10) * 60 * 1000;
    setExpiryTs(expiry);
    setRemainingMs(expiry - now);

    const interval = setInterval(() => {
      const rem = expiry - Date.now();
      setRemainingMs(rem > 0 ? rem : 0);
      if (rem <= 0) clearInterval(interval);
    }, 1000);

    return () => clearInterval(interval);
  }, [open, expiryMinutes]);

  function formatMs(ms) {
    if (ms == null) return '';
    if (ms <= 0) return '00:00';
    const totalSec = Math.floor(ms / 1000);
    const mm = String(Math.floor(totalSec / 60)).padStart(2, '0');
    const ss = String(totalSec % 60).padStart(2, '0');
    return `${mm}:${ss}`;
  }

  if (!open) return null;
  return (
    <div
      style={{
        position: 'fixed',
        inset: 0,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        background: 'rgba(0,0,0,0.35)',
        zIndex: 9999,
      }}
    >
      <div
        style={{
          width: 460,
          maxWidth: '95%',
          background: 'white',
          borderRadius: 10,
          padding: 20,
          boxShadow: '0 10px 40px rgba(0,0,0,0.2)',
        }}
      >
        <div
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            marginBottom: 12,
          }}
        >
          <strong>{title}</strong>
          <button
            onClick={onClose}
            style={{
              border: 'none',
              background: 'transparent',
              cursor: 'pointer',
              fontSize: 18,
            }}
          >
            ✕
          </button>
        </div>

        <div style={{ marginBottom: 12 }}>{children}</div>

        {expiryMinutes ? (
          <div
            style={{
              marginBottom: 12,
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
            }}
          >
            <div style={{ color: '#444' }}>
              OTP expires in:{' '}
              <strong style={{ marginLeft: 8 }}>{formatMs(remainingMs)}</strong>
            </div>
            <div style={{ fontSize: 12, color: '#666' }}>
              Expires at: {expiryTs ? new Date(expiryTs).toLocaleTimeString() : '-'}
            </div>
          </div>
        ) : null}
      </div>
    </div>
  );
}
