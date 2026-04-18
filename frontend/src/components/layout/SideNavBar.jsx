import React from 'react';

const TABS = [
  { id: 'pipeline',    label: 'Pipeline', icon: 'route' },
  { id: 'ga',         label: 'GA Results', icon: 'troubleshoot' },
  { id: 'baselines',  label: 'Baselines', icon: 'ssid_chart' },
  { id: 'ablation',   label: 'Ablation', icon: 'cut' },
  { id: 'filerisk',   label: 'File Risk', icon: 'gpp_bad' },
  { id: 'shap',       label: 'SHAP Explanations', icon: 'psychology' },
  { id: 'chromeditor',label: 'Chromosome Editor', icon: 'family_history' },
  { id: 'advanced',   label: 'Advanced Settings', icon: 'build_circle' },
  { id: 'multirepo',  label: 'Multi-Repo Stats', icon: 'hub' },
  { id: 'sensitivity',label: 'Sensitivity Sweep', icon: 'tune' },
];

export default function SideNavBar({ activeTab, setActiveTab, toggleSettings }) {
  return (
    <nav className="w-64 h-screen fixed left-0 top-0 glass-panel border-r border-outline-variant flex flex-col z-50">
      {/* Brand */}
      <div className="h-16 flex items-center px-6 border-b border-outline-variant shrink-0">
        <span className="material-symbols-outlined text-primary mr-3 text-2xl">neurology</span>
        <span className="font-bold text-on-surface tracking-wider">
          NEURO<span className="text-primary">CORE</span>
        </span>
      </div>

      {/* Nav Links */}
      <div className="flex-1 py-6 flex flex-col gap-2 px-4 overflow-y-auto">
        <div className="text-xs font-semibold text-on-surface-variant tracking-wider mb-2 px-4">MODULES</div>
        {TABS.map(tab => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`flex items-center px-4 py-3 rounded-lg transition-colors group ${
              activeTab === tab.id
                ? 'bg-[rgba(186,18,36,0.1)] text-primary border border-[rgba(186,18,36,0.2)]'
                : 'text-on-surface-variant hover:text-primary hover:bg-surface-container-high'
            }`}
          >
            <span className="material-symbols-outlined mr-3 text-[20px]">{tab.icon}</span>
            <span className="font-medium text-sm">{tab.label}</span>
          </button>
        ))}
      </div>

      {/* Settings toggle */}
      <div className="p-4 border-t border-outline-variant shrink-0">
        <button 
          onClick={toggleSettings}
          className="w-full flex items-center px-4 py-3 text-on-surface-variant hover:text-primary hover:bg-surface-container-high rounded-lg transition-colors group"
        >
          <span className="material-symbols-outlined mr-3">settings</span>
          <span className="font-medium text-sm">Configuration</span>
        </button>
      </div>
    </nav>
  );
}
