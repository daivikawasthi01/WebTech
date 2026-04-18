import React from 'react';

function Slider({ label, value, min, max, step = 1, onChange }) {
  return (
    <div className="mb-3">
      <div className="flex justify-between text-xs text-on-surface-variant mb-1">
        <span>{label}</span>
        <span className="text-primary font-mono">{value}</span>
      </div>
      <input
        type="range" min={min} max={max} step={step} value={value}
        onChange={(e) => onChange(step < 1 ? parseFloat(e.target.value) : parseInt(e.target.value))}
        className="w-full accent-primary"
      />
    </div>
  )
}

function Checkbox({ label, checked, onChange }) {
  return (
    <label className="flex items-center gap-2 text-sm cursor-pointer mb-1">
      <input type="checkbox" checked={checked} onChange={e => onChange(e.target.checked)}
             className="accent-primary w-4 h-4 bg-surface-container border-outline-variant rounded" />
      <span className="text-on-surface text-xs">{label}</span>
    </label>
  )
}

export default function SettingsDrawer({ isOpen, onClose, cfg, setCfg, fileStatus }) {
  const set = (key) => (val) => setCfg(c => ({ ...c, [key]: val }))

  const FILE_STATUS_LABELS = [
    ['raw_data',    'Raw data'],
    ['clean_data',  'Clean data'],
    ['hyperparams', 'Hyperparams'],
    ['ga_results',  'GA results'],
    ['baselines',   'Baselines'],
    ['ablation',    'Ablation'],
    ['stats',       'Stats'],
    ['multi_repo',  'Multi-repo'],
    ['sensitivity', 'Sensitivity'],
    ['report',      'HTML report'],
  ]

  return (
    <>
      {/* Backdrop */}
      {isOpen && (
        <div 
          className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 transition-opacity"
          onClick={onClose}
        />
      )}

      {/* Drawer */}
      <div className={`fixed inset-y-0 right-0 w-96 noir-glass border-l border-outline-variant z-50 transform transition-transform duration-300 flex flex-col ${isOpen ? 'translate-x-0' : 'translate-x-full'}`}>
        <div className="h-16 flex items-center justify-between px-6 border-b border-outline-variant">
          <div className="flex items-center text-on-surface">
            <span className="material-symbols-outlined mr-2">tune</span>
            <span className="font-bold tracking-wider text-sm">CONFIGURATION</span>
          </div>
          <button onClick={onClose} className="text-on-surface-variant hover:text-primary transition-colors">
            <span className="material-symbols-outlined">close</span>
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {/* Paths */}
          <section>
            <h2 className="text-[10px] font-bold text-primary tracking-widest uppercase mb-3">Paths</h2>
            {[
              ['Repo path',      'repo_path'],
              ['Raw file',       'raw_file'],
              ['Clean file',     'processed_file'],
            ].map(([label, key]) => (
              <div key={key} className="mb-3">
                <label className="text-xs text-on-surface-variant block mb-1">{label}</label>
                <input
                  value={cfg[key]}
                  onChange={e => set(key)(e.target.value)}
                  className="w-full bg-surface-container-high border border-outline-variant rounded px-3 py-2 text-xs text-on-surface font-mono focus:border-primary focus:outline-none transition-colors"
                />
              </div>
            ))}
          </section>

          {/* GA Settings */}
          <section>
            <h2 className="text-[10px] font-bold text-primary tracking-widest uppercase mb-3">GA Parameters</h2>
            <Slider label="Population size"  value={cfg.pop_size}      min={5}    max={50}  onChange={set('pop_size')} />
            <Slider label="Generations"      value={cfg.generations}   min={3}    max={50}  onChange={set('generations')} />
            <Slider label="Alpha (accuracy)" value={cfg.alpha}         min={0.1}  max={3.0} step={0.1} onChange={set('alpha')} />
            <Slider label="Beta (parsimony)" value={cfg.beta}          min={0.0}  max={2.0} step={0.1} onChange={set('beta')} />
            <Slider label="Mutation rate"    value={cfg.mutation_rate} min={0.05} max={0.40} step={0.01} onChange={set('mutation_rate')} />
            <Slider label="Min mutation"     value={cfg.min_mutation}  min={0.01} max={0.10} step={0.01} onChange={set('min_mutation')} />
            <Slider label="Stagnation"       value={cfg.stagnation}    min={2}    max={15}  onChange={set('stagnation')} />
          </section>

          {/* Research Modules */}
          <section>
            <h2 className="text-[10px] font-bold text-primary tracking-widest uppercase mb-3">Research Modules</h2>
            <Slider label="Trials per method" value={cfg.n_trials}    min={5}  max={50} onChange={set('n_trials')} />
            <Slider label="Optuna trials"     value={cfg.tune_trials} min={10} max={100} onChange={set('tune_trials')} />
            
            <div className="mt-4 grid grid-cols-2 gap-2">
              <Checkbox label="Optuna Tuning"         checked={cfg.run_tuning}      onChange={set('run_tuning')} />
              <Checkbox label="Baselines"             checked={cfg.run_baselines}   onChange={set('run_baselines')} />
              <Checkbox label="Ablation check"        checked={cfg.run_ablation}    onChange={set('run_ablation')} />
              <Checkbox label="Statistical tests"     checked={cfg.run_stats}       onChange={set('run_stats')} />
              <Checkbox label="Multi-repo check"      checked={cfg.run_multi}       onChange={set('run_multi')} />
              <Checkbox label="Sensitivity sweep"     checked={cfg.run_sensitivity} onChange={set('run_sensitivity')} />
              <Checkbox label="HTML report"           checked={cfg.run_report}      onChange={set('run_report')} />
            </div>
            
            {cfg.run_multi && (
              <div className="mt-4">
                <label className="text-xs text-on-surface-variant block mb-1">Repos (space-separated)</label>
                <input
                  value={cfg.repos.join(' ')}
                  onChange={e => set('repos')(e.target.value.split(/\s+/).filter(Boolean))}
                  className="w-full bg-surface-container-high border border-outline-variant rounded px-3 py-2 text-xs text-on-surface font-mono focus:border-primary focus:outline-none transition-colors"
                />
              </div>
            )}
          </section>

          {/* Flow Controls */}
          <section>
            <h2 className="text-[10px] font-bold text-primary tracking-widest uppercase mb-3">Flow Control</h2>
            <div className="grid grid-cols-2 gap-2">
              <Checkbox label="Force collect"  checked={cfg.force_collect}  onChange={set('force_collect')} />
              <Checkbox label="Force process"  checked={cfg.force_process}  onChange={set('force_process')} />
            </div>
          </section>

          {/* Validation Status */}
          <section>
            <h2 className="text-[10px] font-bold text-primary tracking-widest uppercase mb-3">Validation Status</h2>
            <div className="grid grid-cols-2 gap-y-2">
              {FILE_STATUS_LABELS.map(([key, label]) => (
                <div key={key} className="flex items-center text-xs">
                  <span className={`inline-block w-2 h-2 rounded-full mr-2 ${fileStatus[key] ? 'bg-primary pulse-glow' : 'bg-surface-variant'}`} />
                  <span className={fileStatus[key] ? 'text-on-surface' : 'text-on-surface-variant'}>{label}</span>
                </div>
              ))}
            </div>
          </section>
        </div>
      </div>
    </>
  );
}
