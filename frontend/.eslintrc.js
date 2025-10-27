module.exports = {
  env: { browser: true, es2021: true, node: true },
  extends: [
    'eslint:recommended',
    'plugin:react/recommended',
    'plugin:react-hooks/recommended'
  ],
  parserOptions: { ecmaVersion: 2021, sourceType: 'module', ecmaFeatures: { jsx: true } },
  settings: { react: { version: 'detect' } },
  plugins: ['react', 'react-hooks'],
  rules: {
    'react/prop-types': 'warn',
    'no-empty': ['warn', { 'allowEmptyCatch': true }],
    'no-unused-vars': ['warn', { 'argsIgnorePattern': '^_' }],
    'react/react-in-jsx-scope': 'off'
  }
};
