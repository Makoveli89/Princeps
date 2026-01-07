import React from 'react';

export const PrincepsLogo = ({ className = "w-12 h-12" }: { className?: string }) => (
  <svg viewBox="0 0 100 100" className={className} fill="none" xmlns="http://www.w3.org/2000/svg">
    <path d="M10 80 H 40 Q 50 80 50 60 V 10 H 50 V 60 Q 50 80 60 80 H 90 V 90 H 10 V 80 Z" fill="#1a1a1a" stroke="#00f3ff" strokeWidth="2" />
    <circle cx="50" cy="75" r="4" fill="#00f3ff" className="animate-pulse">
      <animate attributeName="opacity" values="0.5;1;0.5" dur="2s" repeatCount="indefinite" />
    </circle>
    <path d="M50 75 L 50 55" stroke="#00f3ff" strokeWidth="2" strokeOpacity="0.6" />
    <path d="M50 55 L 40 45" stroke="#00f3ff" strokeWidth="1" strokeOpacity="0.4" />
    <path d="M50 55 L 60 45" stroke="#00f3ff" strokeWidth="1" strokeOpacity="0.4" />
  </svg>
);
