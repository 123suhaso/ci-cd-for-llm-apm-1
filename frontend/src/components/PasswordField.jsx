import React, { useState } from 'react';

export default function PasswordField({
  id = 'password',
  value,
  onChange,
  placeholder = ' ',
  required = false,
}) {
  const [show, setShow] = useState(false);
  return (
    <div className="field password">
      <input
        id={id}
        className="input"
        type={show ? 'text' : 'password'}
        value={value}
        onChange={onChange}
        placeholder={placeholder}
        required={required}
        autoComplete="current-password"
      />
      <label htmlFor={id}>Password</label>

      <button
        type="button"
        className="password-toggle"
        onClick={() => setShow((s) => !s)}
        aria-label={show ? 'Hide password' : 'Show password'}
      >
        {show ? (
          <svg
            className="password-icon"
            xmlns="http://www.w3.org/2000/svg"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            strokeWidth="1.6"
          >
            <path d="M1.5 12s4-7.5 10.5-7.5S22.5 12 22.5 12s-4 7.5-10.5 7.5S1.5 12 1.5 12z" />
            <circle cx="12" cy="12" r="3" />
          </svg>
        ) : (
          <svg
            className="password-icon"
            xmlns="http://www.w3.org/2000/svg"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            strokeWidth="1.6"
          >
            <path d="M3 3l18 18" />
            <path d="M9.9 9.9A3 3 0 0112 9c1.66 0 3 1.34 3 3 0 .39-.07.76-.21 1.1" />
            <path d="M2.5 12C4 9.75 8 6 12 6c4 0 8 3.75 9.5 6-1.5 2.25-5.5 6-9.5 6-1.25 0-2.43-.38-3.43-1.04" />
          </svg>
        )}
      </button>
    </div>
  );
}
