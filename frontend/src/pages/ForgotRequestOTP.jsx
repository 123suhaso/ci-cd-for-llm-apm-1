import React, { useState } from 'react';

export default function ForgotRequestOTP({
  apiClient,
  uiConfig,
  onBack,
  onVerify,
  onOtpSent,
}) {
  const [email, setEmail] = useState('');
  const [msg, setMsg] = useState(null);
  const [loading, setLoading] = useState(false);

  const [fieldErrors, setFieldErrors] = useState({ email: null });

  function isValidEmail(e) {
    return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(String(e || '').toLowerCase());
  }

  function validate() {
    const errs = { email: null };
    let ok = true;

    if (!String(email || '').trim()) {
      errs.email = 'Email is required';
      ok = false;
    } else if (!isValidEmail(email)) {
      errs.email = 'Enter a valid email';
      ok = false;
    }

    setFieldErrors(errs);
    return ok;
  }

  const submit = async (e) => {
    e.preventDefault();
    setMsg(null);

    if (!validate()) return;

    setLoading(true);
    try {
      await apiClient.post('/auth/forgot-request-otp', { email });

      const popupMsg = `If an account exists, an OTP was sent to ${email}. It expires in ${
        uiConfig?.otp_expire_minutes ?? 'N/A'
      } minutes.`;
      setMsg(popupMsg);

      if (typeof onOtpSent === 'function') {
        onOtpSent(popupMsg, uiConfig?.otp_expire_minutes);
      }
    } catch (e) {
      const detail = e?.response?.data?.detail;
      const fallback = 'Failed to request OTP';
      if (detail) {
        if (typeof detail === 'string') setMsg(detail);
        else if (Array.isArray(detail))
          setMsg(
            detail
              .map((d) => (typeof d === 'string' ? d : d?.msg || JSON.stringify(d)))
              .join(' • ')
          );
        else if (detail.message) setMsg(detail.message);
        else setMsg(JSON.stringify(detail));
      } else if (e?.message) setMsg(e.message);
      else setMsg(fallback);
    } finally {
      setLoading(false);
    }
  };

  return (
    <form onSubmit={submit} noValidate>
      <h2>Reset password</h2>
      <p>Enter the email to receive an OTP</p>

      <div className="field">
        <input
          id="forgot-email"
          className="input"
          placeholder=" "
          value={email}
          onChange={(e) => {
            setEmail(e.target.value);
            setFieldErrors((s) => ({ ...s, email: null }));
          }}
          required
        />
        <label htmlFor="forgot-email">Email</label>
        {fieldErrors.email && (
          <div style={{ color: 'red', marginTop: 6 }}>{fieldErrors.email}</div>
        )}
      </div>

      <button className="btn" type="submit" disabled={loading}>
        {loading ? 'Sending...' : 'Send OTP'}
      </button>

      {msg && <div style={{ marginTop: 8, whiteSpace: 'pre-wrap' }}>{msg}</div>}

      <div style={{ marginTop: 12 }}>
        <span className="text-link" onClick={onVerify}>
          I have an OTP (verify & reset)
        </span>
        <div style={{ marginTop: 8 }}>
          <span className="text-link" onClick={onBack}>
            Back to login
          </span>
        </div>
      </div>
    </form>
  );
}
