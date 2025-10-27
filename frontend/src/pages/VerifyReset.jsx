import React, { useState } from 'react';
import PasswordField from '../components/PasswordField';

export default function VerifyReset({ apiClient, uiConfig, onBack }) {
  const [email, setEmail] = useState('');
  const [otp, setOtp] = useState('');
  const [newPw, setNewPw] = useState('');
  const [msg, setMsg] = useState(null);
  const [loading, setLoading] = useState(false);

  const [fieldErrors, setFieldErrors] = useState({
    email: null,
    otp: null,
    newPw: null,
  });

  function extractErrorMessage(err) {
    const detail = err?.response?.data?.detail;

    if (detail) {
      if (Array.isArray(detail)) {
        const msgs = detail
          .map((d) => (typeof d === 'string' ? d : d?.msg || JSON.stringify(d)))
          .filter(Boolean);
        return msgs.join(' • ') || JSON.stringify(detail);
      }
      if (typeof detail === 'string') return detail;
      if (typeof detail === 'object') {
        if (detail.message) return detail.message;
        if (detail.msg) return detail.msg;
        try {
          return JSON.stringify(detail);
        } catch {
          return 'An error occurred';
        }
      }
    }

    if (err?.response?.statusText) return err.response.statusText;
    if (err?.message) return err.message;
    return 'Failed to reset';
  }

  function isValidEmail(e) {
    return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(String(e).toLowerCase());
  }

  function validateAll() {
    const errs = { email: null, otp: null, newPw: null };
    let ok = true;

    if (!String(email || '').trim()) {
      errs.email = 'Email is required';
      ok = false;
    } else if (!isValidEmail(email)) {
      errs.email = 'Enter a valid email';
      ok = false;
    }

    if (!String(otp || '').trim()) {
      errs.otp = 'OTP is required';
      ok = false;
    }

    if (!String(newPw || '').trim()) {
      errs.newPw = 'Password is required';
      ok = false;
    } else if ((newPw || '').length < 6) {
      errs.newPw = 'Password must be at least 6 characters';
      ok = false;
    }

    setFieldErrors(errs);
    return ok;
  }

  const submit = async (e) => {
    e.preventDefault();
    setMsg(null);

    const ok = validateAll();
    if (!ok) return;

    setLoading(true);
    try {
      const res = await apiClient.post('/auth/forgot-verify-otp', {
        email,
        otp,
        new_password: newPw,
      });

      const successMsg = res?.data?.message || 'Password updated successfully';
      setMsg({ type: 'success', text: successMsg });
    } catch (e) {
      const errorText = extractErrorMessage(e);
      setMsg({ type: 'error', text: errorText });
    } finally {
      setLoading(false);
    }
  };

  return (
    <form onSubmit={submit} noValidate>
      <h2>Verify OTP & reset</h2>

      <div className="field">
        <input
          id="verify-email"
          className="input"
          placeholder=" "
          value={email}
          onChange={(e) => {
            setEmail(e.target.value);
            setFieldErrors((s) => ({ ...s, email: null }));
          }}
          required
        />
        <label htmlFor="verify-email">Email</label>
        {fieldErrors.email && (
          <div style={{ color: 'red', marginTop: 6 }}>{fieldErrors.email}</div>
        )}
      </div>

      <div className="field">
        <input
          id="verify-otp"
          className="input"
          placeholder=" "
          value={otp}
          onChange={(e) => {
            setOtp(e.target.value);
            setFieldErrors((s) => ({ ...s, otp: null }));
          }}
          required
        />
        <label htmlFor="verify-otp">OTP</label>
        {fieldErrors.otp && (
          <div style={{ color: 'red', marginTop: 6 }}>{fieldErrors.otp}</div>
        )}
      </div>

      <PasswordField
        id="verify-password"
        value={newPw}
        onChange={(e) => {
          setNewPw(e.target.value);
          setFieldErrors((s) => ({ ...s, newPw: null }));
        }}
        required
      />
      {fieldErrors.newPw && (
        <div style={{ color: 'red', marginTop: 6 }}>{fieldErrors.newPw}</div>
      )}

      <button className="btn" type="submit" disabled={loading}>
        {loading ? 'Resetting...' : 'Reset password'}
      </button>

      {msg && (
        <div
          style={{
            marginTop: 8,
            color: msg.type === 'error' ? '#b33' : '#0a7a2e',
            whiteSpace: 'pre-wrap',
          }}
        >
          {msg.text}
        </div>
      )}

      <div style={{ marginTop: 12 }}>
        <span className="text-link" onClick={onBack}>
          Back to login
        </span>
      </div>
    </form>
  );
}
