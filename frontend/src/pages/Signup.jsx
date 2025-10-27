import React, { useState } from 'react';
import PasswordField from '../components/PasswordField';

export default function Signup({ apiClient, onSignupSuccess, onBack }) {
  const [name, setName] = useState('');
  const [username, setUsername] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [role, setRole] = useState('user');
  const [err, setErr] = useState(null);
  const [loading, setLoading] = useState(false);

  const [fieldErrors, setFieldErrors] = useState({
    name: null,
    username: null,
    email: null,
    password: null,
    role: null,
  });

  function toText(v) {
    if (v === undefined || v === null) return '';
    if (typeof v === 'string') return v;
    try {
      return JSON.stringify(v, null, 2);
    } catch {
      return String(v);
    }
  }

  function isValidEmail(e) {
    return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(String(e).toLowerCase());
  }

  function validateAll() {
    const errs = {
      name: null,
      username: null,
      email: null,
      password: null,
      role: null,
    };
    let ok = true;

    if (!String(name || '').trim()) {
      errs.name = 'Full name is required';
      ok = false;
    }
    if (!String(username || '').trim()) {
      errs.username = 'Username is required';
      ok = false;
    }
    if (!String(email || '').trim()) {
      errs.email = 'Email is required';
      ok = false;
    } else if (!isValidEmail(email)) {
      errs.email = 'Enter a valid email address';
      ok = false;
    }
    if (!String(password || '').trim()) {
      errs.password = 'Password is required';
      ok = false;
    } else if ((password || '').length < 6) {
      errs.password = 'Password must be at least 6 characters';
      ok = false;
    }
    if (!String(role || '').trim()) {
      errs.role = 'Role is required';
      ok = false;
    }

    setFieldErrors(errs);
    return ok;
  }

  const submit = async (e) => {
    e.preventDefault();
    setErr(null);
    setLoading(true);

    const ok = validateAll();
    if (!ok) {
      setLoading(false);
      return;
    }

    try {
      const payload = { name, email, username, password, role };
      const res = await apiClient.post('/auth/users', payload);

      onSignupSuccess && onSignupSuccess(res.data);
    } catch (e) {
      const detail = e?.response?.data?.detail;
      let msg = 'Signup failed';
      if (detail) {
        if (typeof detail === 'string') msg = detail;
        else if (Array.isArray(detail)) {
          msg = detail
            .map((d) => (typeof d === 'string' ? d : d.msg || JSON.stringify(d)))
            .join(' • ');
        } else if (detail.message) msg = detail.message;
        else msg = JSON.stringify(detail);
      } else if (e?.response?.data?.message) {
        msg = e.response.data.message;
      } else if (e?.message) {
        msg = e.message;
      }
      setErr(msg);

      const data = e?.response?.data;
      if (data && typeof data === 'object') {
        const newFieldErrs = { ...fieldErrors };
        let foundFieldErr = false;
        for (const key of ['name', 'username', 'email', 'password', 'role']) {
          if (data[key]) {
            foundFieldErr = true;
            const val = data[key];
            newFieldErrs[key] =
              typeof val === 'string'
                ? val
                : Array.isArray(val)
                  ? val.join(', ')
                  : JSON.stringify(val);
          }
        }
        if (foundFieldErr) setFieldErrors(newFieldErrs);
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <form onSubmit={submit} noValidate>
      <h2>Create account</h2>
      <p>Sign up to use LLM APM</p>

      <div className="field">
        <input
          id="signup-name"
          className="input"
          placeholder=" "
          value={name}
          onChange={(e) => {
            setName(e.target.value);
            setFieldErrors((s) => ({ ...s, name: null }));
          }}
          required
        />
        <label htmlFor="signup-name">Full name</label>
        {fieldErrors.name && (
          <div style={{ color: 'red', marginTop: 6 }}>{fieldErrors.name}</div>
        )}
      </div>

      <div className="field">
        <input
          id="signup-username"
          className="input"
          placeholder=" "
          value={username}
          onChange={(e) => {
            setUsername(e.target.value);
            setFieldErrors((s) => ({ ...s, username: null }));
          }}
          required
        />
        <label htmlFor="signup-username">Username</label>
        {fieldErrors.username && (
          <div style={{ color: 'red', marginTop: 6 }}>{fieldErrors.username}</div>
        )}
      </div>

      <div className="field">
        <input
          id="signup-email"
          className="input"
          placeholder=" "
          value={email}
          onChange={(e) => {
            setEmail(e.target.value);
            setFieldErrors((s) => ({ ...s, email: null }));
          }}
          required
        />
        <label htmlFor="signup-email">Email</label>
        {fieldErrors.email && (
          <div style={{ color: 'red', marginTop: 6 }}>{fieldErrors.email}</div>
        )}
      </div>

      <PasswordField
        id="signup-password"
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

      <div className="field">
        <select
          id="signup-role"
          className="input"
          value={role}
          onChange={(e) => {
            setRole(e.target.value);
            setFieldErrors((s) => ({ ...s, role: null }));
          }}
          required
        >
          <option value="user">User</option>
          <option value="admin">Admin</option>
        </select>
        {fieldErrors.role && (
          <div style={{ color: 'red', marginTop: 6 }}>{fieldErrors.role}</div>
        )}
      </div>

      {err && <div style={{ color: 'red', marginBottom: 8 }}>{err}</div>}
      <button className="btn" type="submit" disabled={loading}>
        {loading ? 'Creating...' : 'Create account'}
      </button>

      <div style={{ marginTop: 12 }}>
        <span className="text-link" onClick={onBack}>
          Back to login
        </span>
      </div>
    </form>
  );
}
