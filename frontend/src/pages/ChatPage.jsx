import React, { useEffect, useState } from 'react';
import { useNavigate, useParams, useLocation } from 'react-router-dom';
import Topbar from '../components/Topbar';
import SidebarChats from '../components/SidebarChats';
import ChatWindow from '../components/ChatWindow';
import ModelSelector from '../components/ModelSelector';

/**
 * ChatPage - updated to:
 * - fetch /available_models
 * - auto-apply provider/model when selected
 * - auto-create chat when selection made and no active chat
 * - validation: block ollama + gpt-4o combos (inline error, no repeated bot messages)
 *
 * Props:
 * - apiClient (axios-like instance)
 * - token, onLogout, uiConfig, userInfo (same as before)
 */
export default function ChatPage({ apiClient, token, onLogout, uiConfig, userInfo }) {
  const navigate = useNavigate();
  const params = useParams();
  const location = useLocation();

  const [chats, setChats] = useState([]);
  const [activeId, setActiveId] = useState(null);
  const [available, setAvailable] = useState({
    providers: {},
    default_provider: null,
  });
  const [selectedProvider, setSelectedProvider] = useState('');
  const [selectedModel, setSelectedModel] = useState('');
  const [error, setError] = useState(null);

  const storageKey = 'llm_chats';
  const defaultModel = uiConfig?.model_name || 'gpt-4o-mini';

  // load chats from sessionStorage
  useEffect(() => {
    const stored = sessionStorage.getItem(storageKey);
    if (stored) {
      try {
        const parsed = JSON.parse(stored);
        setChats(parsed);
        if (params?.id) {
          const exists = parsed.find((c) => c.id === params.id);
          if (exists) setActiveId(params.id);
          else if (parsed.length) setActiveId(parsed[0].id);
        } else {
          if (parsed.length) setActiveId(parsed[0].id);
        }
      } catch {
        setChats([]);
        setActiveId(null);
      }
    }
  }, [params?.id]);

  // persist chats
  useEffect(() => {
    try {
      sessionStorage.setItem(storageKey, JSON.stringify(chats));
    } catch {}
  }, [chats]);

  // fetch available models from backend
  useEffect(() => {
    let mounted = true;
    async function fetchModels() {
      try {
        const res = await apiClient.get('/available_models');
        if (!mounted) return;
        const provs = (res && res.data && res.data.providers) || {};
        setAvailable({
          providers: provs,
          default_provider: (res && res.data && res.data.default_provider) || null,
        });
      } catch (e) {
        console.error('Failed to fetch available models', e);
        setAvailable({ providers: {}, default_provider: null });
      }
    }
    fetchModels();
    return () => (mounted = false);
  }, [apiClient]);

  function newChat(customProps = {}) {
    const id = Date.now().toString();
    setChats((prev) => {
      const title =
        customProps.title || `Chat ${prev && prev.length ? prev.length + 1 : 1}`;
      const c = {
        id,
        title,
        provider: customProps.provider || available.default_provider || '',
        model: customProps.model || defaultModel,
        messages: customProps.messages || [],
        last_metrics: customProps.last_metrics || null,
      };
      return [c, ...prev];
    });
    setActiveId(id);
    navigate(`/chat/${id}`);
    return id;
  }

  function addMessageToChatId(chatId, sender, text) {
    setChats((prev) => {
      const found = prev.find((c) => c.id === chatId);
      if (found) {
        return prev.map((c) =>
          c.id === chatId
            ? { ...c, messages: [...(c.messages || []), { sender, text }] }
            : c
        );
      } else {
        const newChatObj = {
          id: chatId,
          title: `Chat ${prev.length + 1}`,
          provider: available.default_provider || '',
          model: defaultModel,
          messages: [{ sender, text }],
          last_metrics: null,
        };
        return [newChatObj, ...prev];
      }
    });
  }

  // helper: safely update active chat provider/model
  function updateActiveChatProviderModel(provider, model) {
    setChats((prev) =>
      prev.map((c) =>
        c.id === activeId ? { ...c, provider: provider || '', model: model || '' } : c
      )
    );
  }

  // Called when user selects provider/model in ModelSelector
  function handleProviderModelSelect(provider, model) {
    setError(null);

    // normalize
    const p = (provider || '').toString();
    let m = (model || '').toString();

    // If user changed provider, auto-pick first model for that provider
    if (p && p !== selectedProvider) {
      const pModels = available.providers[p] || [];
      m = pModels.length ? pModels[0] : '';
    }

    setSelectedProvider(p);
    setSelectedModel(m);

    // If no active chat, create one automatically with selection
    if (!activeId) {
      const id = newChat({ provider: p, model: m, messages: [] });
      setSelectedProvider(p);
      setSelectedModel(m);
      setActiveId(id);
      return;
    }

    // Validation: if ollama + gpt-4o -> block and choose fallback
    if (p && m && p.toLowerCase() === 'ollama' && m.toLowerCase().includes('gpt-4o')) {
      setError(
        '❌ Ollama cannot use GPT-4o models. Please choose a different model or provider.'
      );
      // try fallback: pick first non-gpt-4o model for ollama if any
      const ollamaModels = (available.providers['ollama'] || []).filter(
        (mm) => !String(mm).toLowerCase().includes('gpt-4o')
      );
      if (ollamaModels.length) {
        m = ollamaModels[0];
        setSelectedModel(m);
        updateActiveChatProviderModel(p, m);
      }
      // if no fallback, keep provider but clear model to force user choice
      else {
        setSelectedModel('');
        updateActiveChatProviderModel(p, '');
      }
      return;
    }

    // OK -> clear error and update chat
    setError(null);
    updateActiveChatProviderModel(p, m);
  }

  // send prompt (construct payload with provider & model)
  async function sendPrompt(promptText) {
    const text = String(promptText ?? '').trim();
    if (!text) return;

    let chatId = activeId;
    // determine provider + model to send (from active chat)
    const activeChat = chats.find((c) => c.id === chatId);
    const providerToUse =
      (activeChat && activeChat.provider) || available.default_provider || '';
    const modelToUse = (activeChat && activeChat.model) || defaultModel;

    // Validate before sending: block ollama+gpt-4o
    if (
      providerToUse &&
      modelToUse &&
      providerToUse.toLowerCase() === 'ollama' &&
      modelToUse.toLowerCase().includes('gpt-4o')
    ) {
      // show inline error only
      setError(
        '❌ Cannot send: Ollama is not compatible with GPT-4o models. Change provider or model.'
      );
      return;
    }

    // Add user message to UI
    addMessageToChatId(chatId, 'user', text);

    const payload = {
      prompt: text,
      max_tokens: uiConfig?.default_max_tokens ?? 150,
      temperature: uiConfig?.default_temperature ?? 0.7,
      session_id: chatId,
      provider: providerToUse,
      model: modelToUse,
    };

    try {
      const res = await apiClient.post('/generate', payload);
      const botText = res?.data?.response ?? '(no response)';
      addMessageToChatId(chatId, 'bot', botText);
      // store metrics on chat
      setChats((prev) =>
        prev.map((c) =>
          c.id === chatId ? { ...c, last_metrics: res.data.metrics || {} } : c
        )
      );
    } catch (e) {
      const errMsg = e?.response?.data?.detail || e?.message || 'Error';
      addMessageToChatId(chatId, 'bot', `Error: ${errMsg}`);
      setChats((prev) =>
        prev.map((c) =>
          c.id === chatId ? { ...c, last_metrics: { error: errMsg } } : c
        )
      );
    }
  }

  // sync selectedProvider/model when active chat changes
  useEffect(() => {
    const active = chats.find((c) => c.id === activeId);
    if (active) {
      setSelectedProvider(active.provider || available.default_provider || '');
      setSelectedModel(active.model || '');
      setError(null);
      // ensure URL matches active chat
      if (!params?.id || params.id !== activeId) {
        navigate(`/chat/${activeId}`, { replace: false });
      }
    } else {
      setSelectedProvider(available.default_provider || '');
      setSelectedModel('');
      if (location.pathname !== '/chat') {
        navigate('/chat', { replace: true });
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeId, chats, available]);

  const activeChat = chats.find((c) => c.id === activeId) || null;

  return (
    <div>
      <Topbar
        title="LLM-APM (LLM Application Performance Monitor)"
        user={userInfo}
        onLogout={onLogout}
      />

      <div className="container" style={{ display: 'flex', gap: 12 }}>
        <div className="sidebar" style={{ width: 280 }}>
          <button
            className="btn"
            onClick={() => newChat()}
            style={{ width: '100%', marginBottom: 8 }}
          >
            + New chat
          </button>
          <SidebarChats
            chats={chats}
            activeId={activeId}
            onSelect={(id) => setActiveId(id)}
          />
        </div>

        <div className="chat-main" style={{ flex: 1 }}>
          <ModelSelector
            providersMap={available.providers}
            selectedProvider={selectedProvider}
            selectedModel={selectedModel}
            onSelect={(p, m) => handleProviderModelSelect(p, m)}
          />

          {error && <div style={{ color: '#b00020', padding: 8 }}>{error}</div>}

          <div
            style={{
              border: '1px solid #eee',
              borderRadius: 8,
              overflow: 'hidden',
              height: '70vh',
              display: 'flex',
              flexDirection: 'column',
            }}
          >
            <ChatWindow chat={activeChat} onSend={sendPrompt} />
          </div>

          {activeChat?.last_metrics && (
            <div className="metrics" style={{ marginTop: 12 }}>
              <h4>Last response metrics</h4>
              <div style={{ overflowX: 'auto' }}>
                <table className="metrics-table">
                  <tbody>
                    {Object.entries(activeChat.last_metrics).map(([k, v]) => (
                      <tr key={k}>
                        <th style={{ textAlign: 'left', paddingRight: 8 }}>{k}</th>
                        <td>
                          {typeof v === 'object'
                            ? JSON.stringify(v, null, 2)
                            : String(v)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
