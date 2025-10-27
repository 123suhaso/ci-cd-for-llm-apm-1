import React from 'react';

/**
 * ModelSelector (no Apply button)
 * Props:
 * - providersMap: { providerName: [modelName, ...], ... }
 * - selectedProvider, selectedModel
 * - onSelect(provider, model) -> called immediately when user changes provider or model
 */
export default function ModelSelector({
  providersMap = {},
  selectedProvider,
  selectedModel,
  onSelect,
}) {
  const providerNames = Object.keys(providersMap || {});
  const modelsForProvider = (providersMap[selectedProvider] || []).slice();

  function handleProviderChange(p) {
    const firstModel = (providersMap[p] && providersMap[p][0]) || '';
    onSelect(p || '', firstModel || '');
  }

  function handleModelChange(m) {
    onSelect(selectedProvider || '', m || '');
  }

  return (
    <div
      style={{
        display: 'flex',
        gap: 8,
        alignItems: 'center',
        padding: 8,
        borderBottom: '1px solid #eee',
      }}
    >
      <label style={{ fontSize: 13 }}>Provider:</label>
      <select
        value={selectedProvider || ''}
        onChange={(e) => handleProviderChange(e.target.value)}
      >
        <option value="">-- select provider --</option>
        {providerNames.map((p) => (
          <option key={p} value={p}>
            {p}
          </option>
        ))}
      </select>

      <label style={{ fontSize: 13 }}>Model:</label>
      <select
        value={selectedModel || ''}
        onChange={(e) => handleModelChange(e.target.value)}
        disabled={!selectedProvider}
      >
        <option value="">-- select model --</option>
        {modelsForProvider.map((m) => (
          <option key={m} value={m}>
            {m}
          </option>
        ))}
      </select>
    </div>
  );
}
