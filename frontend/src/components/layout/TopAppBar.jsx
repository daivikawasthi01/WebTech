import React from 'react';

export default function TopAppBar({ activeTabLabel, systemActive }) {
  return (
    <header className="h-16 pl-72 pr-8 flex border-b border-outline-variant glass-panel items-center justify-between sticky top-0 z-40">
      <div className="flex items-center text-on-surface-variant">
        <span className="text-sm font-medium">
          Core Platform / <span className="text-on-surface">{activeTabLabel}</span>
        </span>
      </div>
      <div className="flex items-center gap-4">
        <div className="flex items-center px-3 py-1 rounded-full bg-surface-container-high border border-outline-variant">
          <span className={`w-2 h-2 rounded-full mr-2 ${systemActive ? 'bg-primary pulse-glow' : 'bg-surface-variant'}`}></span>
          <span className={`text-xs font-mono ${systemActive ? 'text-primary' : 'text-on-surface-variant'}`}>
            {systemActive ? 'SYSTEM RUNNING' : 'SYSTEM IDLE'}
          </span>
        </div>
        <button className="w-8 h-8 rounded-full bg-surface-container-high flex items-center justify-center text-on-surface-variant hover:text-primary transition-colors border border-outline-variant">
          <span className="material-symbols-outlined text-sm">notifications</span>
        </button>
      </div>
    </header>
  );
}
