import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import ReviewerDemo from './ReviewerDemo';
import './styles.css';

const demo = import.meta.env.VITE_STATIC_DEMO === 'true';
ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>{demo ? <ReviewerDemo/> : <App/>}</React.StrictMode>
);
