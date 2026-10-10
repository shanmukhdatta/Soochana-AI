import React from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';
import './styles.css';
import './mono.css';

// The authentic full frontend is always rendered. The demo build uses an
// evidence-only, explicitly unavailable-model API adapter instead of FastAPI.
createRoot(document.getElementById('root')!).render(
  <React.StrictMode><App/></React.StrictMode>
);
