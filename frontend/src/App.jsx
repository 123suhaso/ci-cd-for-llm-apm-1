import React, { useEffect, useState } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import Login from './pages/Login';
import ChatPage from './pages/ChatPage';
import client from './api/axiosClient';

export default function App() {
  const [uiConfig, setUiConfig] = useState(null);
  const [token, setToken] = useState(null);
  const [userInfo, setUserInfo] = useState(null);
  const [loadingConfig, setLoadingConfig] = useState(true);

  useEffect(() => {
    let mounted = true;
    async function loadConfig() {
      try {
        const res = await client.get('/auth/ui-config');
        if (!mounted) return;
        setUiConfig(res.data);
      } catch (e) {
        console.warn('ui-config failed, using defaults', e);
        if (!mounted) return;
        setUiConfig({ model_name: 'gpt-4o-mini', otp_expire_minutes: 10 });
      } finally {
        if (mounted) setLoadingConfig(false);
      }
    }
    loadConfig();
    return () => {
      mounted = false;
    };
  }, []);

  useEffect(() => {
    try {
      const t = localStorage.getItem('llm_token');
      const u = localStorage.getItem('llm_user');
      if (t) setToken(t);
      if (u) setUserInfo(JSON.parse(u));
    } catch (e) {}
  }, []);

  useEffect(() => {
    if (token) {
      client.defaults.headers.Authorization = `Bearer ${token}`;
      try {
        localStorage.setItem('llm_token', token);
      } catch {}
    } else {
      delete client.defaults.headers.Authorization;
      try {
        localStorage.removeItem('llm_token');
      } catch {}
    }
  }, [token]);

  useEffect(() => {
    try {
      if (userInfo) localStorage.setItem('llm_user', JSON.stringify(userInfo));
      else localStorage.removeItem('llm_user');
    } catch {}
  }, [userInfo]);

  if (loadingConfig || !uiConfig) {
    return <div style={{ padding: 40 }}>Loading configuration...</div>;
  }

  return (
    <BrowserRouter>
      <Routes>
        {!token ? (
          <>
            <Route
              path="/"
              element={
                <Login
                  apiClient={client}
                  onLogin={(t, u) => {
                    setToken(t);
                    setUserInfo(u);
                  }}
                  uiConfig={uiConfig}
                />
              }
            />
            <Route
              path="/signup"
              element={
                <Login
                  apiClient={client}
                  onLogin={(t, u) => {
                    setToken(t);
                    setUserInfo(u);
                  }}
                  uiConfig={uiConfig}
                />
              }
            />
            <Route
              path="/forgot"
              element={
                <Login
                  apiClient={client}
                  onLogin={(t, u) => {
                    setToken(t);
                    setUserInfo(u);
                  }}
                  uiConfig={uiConfig}
                />
              }
            />
            <Route
              path="/verify"
              element={
                <Login
                  apiClient={client}
                  onLogin={(t, u) => {
                    setToken(t);
                    setUserInfo(u);
                  }}
                  uiConfig={uiConfig}
                />
              }
            />
            <Route path="*" element={<Navigate to="/" replace />} />
          </>
        ) : (
          <>
            <Route
              path="/chat"
              element={
                <ChatPage
                  apiClient={client}
                  token={token}
                  onLogout={() => {
                    setToken(null);
                    setUserInfo(null);
                  }}
                  uiConfig={uiConfig}
                  userInfo={userInfo}
                />
              }
            />
            <Route
              path="/chat/:id"
              element={
                <ChatPage
                  apiClient={client}
                  token={token}
                  onLogout={() => {
                    setToken(null);
                    setUserInfo(null);
                  }}
                  uiConfig={uiConfig}
                  userInfo={userInfo}
                />
              }
            />
            <Route path="/" element={<Navigate to="/chat" replace />} />
            <Route path="*" element={<Navigate to="/chat" replace />} />
          </>
        )}
      </Routes>
    </BrowserRouter>
  );
}
