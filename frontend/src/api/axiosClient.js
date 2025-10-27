import axios from 'axios';

// Read from runtime env first, fallback to localhost
const API_BASE = window.__env?.VITE_API_BASE_URL || 'http://localhost:8000';

const client = axios.create({
  baseURL: API_BASE,
  withCredentials: false,
  headers: { 'Content-Type': 'application/json' },
});

export default client;
