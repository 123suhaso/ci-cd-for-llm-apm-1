import React from 'react';

export default function Topbar({ title, user, onLogout }) {
  // Use runtime env for Grafana URL
  const GRAFANA_URL = window.__env?.VITE_GRAFANA_URL || 'http://localhost:3000';

  return (
    <div className="topbar">
      <div className="left">
        <div className="user-pill">{user?.username}</div>
      </div>

      <div className="title">{title}</div>

      <div className="right">
        <a className="btn" href={GRAFANA_URL} target="_blank" rel="noopener noreferrer">
          Dashboard
        </a>
        <button className="btn" onClick={onLogout}>
          Logout
        </button>
      </div>
    </div>
  );
}
