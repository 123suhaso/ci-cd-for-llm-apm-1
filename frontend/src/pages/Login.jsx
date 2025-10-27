import React, { useState, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import ForgotRequestOTP from './ForgotRequestOTP';
import VerifyReset from './VerifyReset';
import Signup from './Signup';
import Modal from '../components/Modal';
import PasswordField from '../components/PasswordField';

export default function Login({ apiClient, onLogin, uiConfig }) {
  const navigate = useNavigate();
  const location = useLocation();

  const routeMode = (() => {
    if (location.pathname === '/signup') return 'signup';
    if (location.pathname === '/forgot') return 'forgot';
    if (location.pathname === '/verify') return 'verify';
    return 'login';
  })();

  const [mode, setMode] = useState(routeMode);
  const [otpPopupOpen, setOtpPopupOpen] = useState(false);
  const [otpPopupMessage, setOtpPopupMessage] = useState('');
  const [otpExpiryMinutes, setOtpExpiryMinutes] = useState(null);
  const [modalTitle, setModalTitle] = useState('');

  useEffect(() => {
    setMode(
      location.pathname === '/signup'
        ? 'signup'
        : location.pathname === '/forgot'
          ? 'forgot'
          : location.pathname === '/verify'
            ? 'verify'
            : 'login'
    );
  }, [location.pathname]);

  const handleOtpSent = (message, expiryMinutes) => {
    setModalTitle('OTP sent');
    setOtpPopupMessage(
      message ||
        `If an account exists, an OTP was sent. Expires in ${expiryMinutes} minutes.`
    );
    setOtpExpiryMinutes(expiryMinutes || null);
    setOtpPopupOpen(true);
  };

  const handleSignupSuccess = (res) => {
    setModalTitle('User created successfully');
    setOtpPopupMessage('Account created. Please login.');
    setOtpExpiryMinutes(null);
    setOtpPopupOpen(true);
    setMode('login');
    navigate('/');
  };

  return (
    <div className="auth-outer">
      <div className="left-illustration">
        <div className="illustration-wrapper">
          <img src="/illustration.png" alt="LLM APM Illustration" />
          <div className="illustration-text">
            <h1>LLM APM</h1>
            <p>Secure & monitored</p>
          </div>
        </div>
      </div>
      <div className="right-panel">
        <div className="card">
          {mode === 'login' && (
            <LoginForm
              apiClient={apiClient}
              onLogin={(t, u) => {
                onLogin(t, u);
                navigate('/chat');
              }}
            />
          )}
          {mode === 'forgot' && (
            <ForgotRequestOTP
              apiClient={apiClient}
              uiConfig={uiConfig}
              onBack={() => {
                navigate('/');
                setMode('login');
              }}
              onVerify={() => {
                navigate('/verify');
                setMode('verify');
              }}
              onOtpSent={handleOtpSent}
            />
          )}
          {mode === 'verify' && (
            <VerifyReset
              apiClient={apiClient}
              uiConfig={uiConfig}
              onBack={() => {
                navigate('/');
                setMode('login');
              }}
            />
          )}
          {mode === 'signup' && (
            <Signup
              apiClient={apiClient}
              onSignupSuccess={handleSignupSuccess}
              onBack={() => {
                navigate('/');
                setMode('login');
              }}
            />
          )}
        </div>
      </div>

      <Modal
        open={otpPopupOpen}
        title={modalTitle}
        onClose={() => setOtpPopupOpen(false)}
        expiryMinutes={otpExpiryMinutes}
      >
        <div style={{ marginBottom: 12 }}>{otpPopupMessage}</div>
        <div style={{ textAlign: 'right' }}>
          <button className="btn" onClick={() => setOtpPopupOpen(false)}>
            OK
          </button>
        </div>
      </Modal>
    </div>
  );
}

function LoginForm({ apiClient, onLogin }) {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [err, setErr] = useState(null);
  const [loading, setLoading] = useState(false);
  const [fieldErrors, setFieldErrors] = useState({
    username: null,
    password: null,
  });

  const validate = () => {
    const errors = { username: null, password: null };
    let ok = true;
    if (!String(username || '').trim()) {
      errors.username = 'Username or email is required';
      ok = false;
    }
    if (!String(password || '').trim()) {
      errors.password = 'Password is required';
      ok = false;
    }
    setFieldErrors(errors);
    return ok;
  };

  const submit = async (e) => {
    e.preventDefault();
    setErr(null);
    if (!validate()) return;
    setLoading(true);
    try {
      const fd = new URLSearchParams();
      fd.append('username', username);
      fd.append('password', password);
      const res = await apiClient.post('/auth/login', fd.toString(), {
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      });
      onLogin(res.data.access_token, res.data.user || null);
    } catch (e) {
      const detail = e?.response?.data?.detail;
      setErr(
        typeof detail === 'string'
          ? detail
          : detail?.msg || 'Login failed. Please try again.'
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <form onSubmit={submit} noValidate>
      <h2>Welcome back</h2>
      <p>Sign in to your account</p>

      <div className="field">
        <input
          id="login-username"
          className="input"
          placeholder=" "
          value={username}
          onChange={(e) => {
            setUsername(e.target.value);
            setFieldErrors((s) => ({ ...s, username: null }));
          }}
          required
        />
        <label htmlFor="login-username">Email or username</label>
        {fieldErrors.username && (
          <div style={{ color: 'red', marginTop: 6 }}>{fieldErrors.username}</div>
        )}
      </div>

      <PasswordField
        id="login-password"
        value={password}
        onChange={(e) => {
          setPassword(e.target.value);
          setFieldErrors((s) => ({ ...s, password: null }));
        }}
        required
      />
      {fieldErrors.password && (
        <div style={{ color: 'red', marginTop: 6 }}>{fieldErrors.password}</div>
      )}
      {err && <div style={{ color: 'red', marginBottom: 8 }}>{err}</div>}
      <button className="btn" type="submit" disabled={loading}>
        {loading ? 'Signing in...' : 'Sign in'}
      </button>

      {/* Removed "Forgot password?" and "Sign up" links as requested */}
    </form>
  );
}
