import { useState, useEffect } from 'react'
import Plot from 'react-plotly.js'
import { api } from '../utils/api.js'
import { PLOT_LAYOUT, GRID, COLORS, PLOTLY_CONFIG } from '../utils/plotTheme.js'

const METHOD_LABELS = {
  all_features:   'All Features Matrix',
  random_subset:  'Random Subset',
  ga_selected:    'GA Neural-Evo',
  xgb_ga:         'GA + XGBoost',
}
const METHOD_COLORS = {
  all_features:  COLORS.muted,
  random_subset: COLORS.yellow,
  ga_selected:   COLORS.accent,
  xgb_ga:        COLORS.teal,
}

export function BaselinesTab() {
  const [bl, setBl]         = useState(null)
  const [stats, setStats]   = useState(null)

  useEffect(() => {
    api.results('baseline_results').then(r => setBl(r.data)).catch(() => {})
    api.results('stats_results').then(r => setStats(r.data)).catch(() => {})
  }, [])

  if (!bl) return (
    <div className="flex flex-col items-center justify-center mt-20 text-on-surface-variant h-[60vh] glass-panel rounded-xl border border-outline-variant border-dashed">
      <span className="material-symbols-outlined text-4xl mb-3 text-outline-variant">ssid_chart</span>
      <p className="text-sm font-medium tracking-wide">NO BASELINE TELEMETRY COMPUTED</p>
      <p className="text-xs mt-1 font-mono">Toggle "Baselines" in Configuration and execute</p>
    </div>
  )

  const methods = ['all_features', 'random_subset', 'ga_selected', 'xgb_ga'].filter(k => bl[k])

  return (
    <div className="space-y-6 pb-20">
      <div>
        <h2 className="text-3xl font-bold text-on-surface mb-2">Performance Baselines</h2>
        <p className="text-sm text-on-surface-variant">Comparative analysis against standard feature selection strategies.</p>
      </div>

      {/* Summary cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        {methods.map(k => (
          <div key={k} className="glass-panel rounded-xl p-6 relative overflow-hidden group">
            <div className="absolute inset-0 bg-primary opacity-0 group-hover:opacity-5 transition-opacity"></div>
            <div className="text-[10px] font-bold text-on-surface-variant tracking-widest uppercase mb-2">{METHOD_LABELS[k]}</div>
            <div className="text-3xl font-mono mb-1" style={{ color: METHOD_COLORS[k] }}>
              {bl[k].mean?.toFixed(4)}
            </div>
            <div className="text-xs text-on-surface-variant font-mono">±{bl[k].std?.toFixed(4)}</div>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Bar chart */}
        <div className="glass-panel rounded-xl p-6">
           <h3 className="text-xs font-bold text-primary tracking-widest uppercase mb-4 flex items-center gap-2">
             <span className="material-symbols-outlined text-[16px]">bar_chart</span>
             Mean Error Deviance (Lower = Better)
           </h3>
          <div className="bg-surface-container-lowest border border-outline-variant rounded-lg">
              <Plot
                data={methods.map(k => ({
                  x: [METHOD_LABELS[k]],
                  y: [bl[k].mean],
                  error_y: { type: 'data', array: [bl[k].std], visible: true },
                  type: 'bar', marker: { color: METHOD_COLORS[k] }, name: METHOD_LABELS[k],
                }))}
                layout={{
                  ...PLOT_LAYOUT, height: 320,
                  yaxis: { ...GRID },
                  showlegend: false, bargap: 0.4,
                  margin: { l: 40, r: 10, t: 30, b: 60 }
                }}
                config={PLOTLY_CONFIG} style={{ width: '100%' }}
              />
          </div>
        </div>

        {/* Box plots */}
        <div className="glass-panel rounded-xl p-6">
          <h3 className="text-xs font-bold text-primary tracking-widest uppercase mb-4 flex items-center gap-2">
             <span className="material-symbols-outlined text-[16px]">candlestick_chart</span>
             Variance Distribution Surface
           </h3>
          <div className="bg-surface-container-lowest border border-outline-variant rounded-lg">
              <Plot
                data={methods.filter(k => bl[k].mses?.length).map(k => ({
                  y: bl[k].mses, name: METHOD_LABELS[k], type: 'box',
                  marker: { color: METHOD_COLORS[k] },
                  boxpoints: 'all', jitter: 0.3, pointpos: -1.8,
                }))}
                layout={{
                  ...PLOT_LAYOUT, height: 320,
                  yaxis: { ...GRID }, showlegend: false,
                  margin: { l: 40, r: 10, t: 30, b: 60 }
                }}
                config={PLOTLY_CONFIG} style={{ width: '100%' }}
              />
          </div>
        </div>
      </div>

      {/* Stats banner */}
      {stats && (
        <div className={`mt-2 rounded-xl p-4 text-sm border flex items-center gap-4 ${
          stats.significant
            ? 'glass-panel border-green-500/30 text-green-400 bg-green-500/5'
            : 'glass-panel border-yellow-500/30 text-yellow-400 bg-yellow-500/5'
        }`}>
          <span className="material-symbols-outlined text-2xl">
            {stats.significant ? 'verified' : 'help_outline'}
          </span>
          <div>
            <div className="font-bold tracking-widest uppercase text-xs mb-1">
                {stats.significant ? 'Significant Divergence Detected' : 'No Significant Divergence'}
            </div>
            <div className="font-mono text-[11px] opacity-80">
                Wilcoxon p={stats.wilcoxon_p_value?.toFixed(4)} &nbsp;|&nbsp;
                Cohen's d={stats.cohens_d?.toFixed(3)} ({stats.effect_size.toUpperCase()}) &nbsp;|&nbsp;
                Improvement: {stats.pct_improvement?.toFixed(1)}%
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

// AblationTab.jsx
export function AblationTab() {
  const [abl, setAbl] = useState(null)

  useEffect(() => {
    api.results('ablation_results').then(r => setAbl(r.data)).catch(() => {})
  }, [])

  if (!abl) return (
    <div className="flex flex-col items-center justify-center mt-20 text-on-surface-variant h-[60vh] glass-panel rounded-xl border border-outline-variant border-dashed">
      <span className="material-symbols-outlined text-4xl mb-3 text-outline-variant">cut</span>
      <p className="text-sm font-medium tracking-wide">NO ABLATION DATA</p>
      <p className="text-xs mt-1 font-mono">Toggle "Ablation" in Configuration and execute</p>
    </div>
  )

  const sorted = Object.entries(abl)
    .filter(([, v]) => typeof v === 'object' && 'mean' in v)
    .sort(([, a], [, b]) => a.mean - b.mean)

  const COMBO_COLORS = {
    'A + B + C (Full)':       COLORS.accent,
    'C only (Evolutionary)':  COLORS.teal,
    'B only (Textual)':       COLORS.blue,
    'A only (Structural)':    COLORS.yellow,
  }

  const a_only = abl['A only (Structural)']?.mean
  const abc    = abl['A + B + C (Full)']?.mean
  const improvement = a_only && abc
    ? ((a_only - abc) / (a_only + 1e-9) * 100).toFixed(1)
    : null

  return (
    <div className="space-y-6 pb-20">
      <div>
        <h2 className="text-3xl font-bold text-on-surface mb-2">Ablation Study</h2>
        <p className="text-sm text-on-surface-variant">Component isolations and category performance degradation.</p>
      </div>

      {improvement && (
        <div className="glass-panel border-blue-500/30 rounded-xl p-4 text-sm text-blue-400 bg-blue-500/5 flex items-center gap-4">
          <span className="material-symbols-outlined text-2xl">insights</span>
          <span>
              Integrating textual and evolutionary vectors to standard structural metrics reduces MSE by <strong>{improvement}%</strong>.
          </span>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Horizontal bar */}
        <div className="glass-panel rounded-xl p-6">
          <h3 className="text-xs font-bold text-primary tracking-widest uppercase mb-4 flex items-center gap-2">
             <span className="material-symbols-outlined text-[16px]">align_horizontal_left</span>
             Category Impact Matrix
           </h3>
          <div className="bg-surface-container-lowest border border-outline-variant rounded-lg">
              <Plot
                data={[{
                  x: sorted.map(([, v]) => v.mean),
                  y: sorted.map(([k]) => k.replace(/ \(/, '<br>(')),
                  error_x: { type: 'data', array: sorted.map(([, v]) => v.std), visible: true },
                  type: 'bar', orientation: 'h',
                  marker: { color: sorted.map(([k]) => COMBO_COLORS[k] || COLORS.muted) },
                }]}
                layout={{
                  ...PLOT_LAYOUT, height: 400,
                  xaxis: { ...GRID },
                  yaxis: { automargin: true },
                  margin: { l: 120, r: 20, t: 20, b: 40 }
                }}
                config={PLOTLY_CONFIG} style={{ width: '100%' }}
              />
          </div>
        </div>

        {/* Table */}
        <div className="glass-panel rounded-xl p-6 flex flex-col">
          <h3 className="text-xs font-bold text-primary tracking-widest uppercase mb-4 flex items-center gap-2">
             <span className="material-symbols-outlined text-[16px]">view_list</span>
             Isolation Results
          </h3>
          <div className="bg-surface-container-lowest border border-outline-variant rounded-lg flex-1 overflow-x-auto custom-scrollbar">
            <table className="text-xs w-full text-left border-collapse">
              <thead className="bg-surface-container-high border-b border-outline-variant">
                <tr>
                  <th className="px-4 py-3 text-on-surface-variant font-semibold">Vector Combination</th>
                  <th className="px-4 py-3 text-on-surface-variant font-semibold text-right">Features</th>
                  <th className="px-4 py-3 text-on-surface-variant font-semibold text-right">Mean Deviance</th>
                  <th className="px-4 py-3 text-on-surface-variant font-semibold text-right">Std</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-outline-variant/30">
                {sorted.map(([name, data]) => (
                  <tr key={name} className="hover:bg-surface-container-high transition-colors">
                    <td className="px-4 py-3 font-medium text-on-surface">{name}</td>
                    <td className="px-4 py-3 text-right text-on-surface-variant font-mono">{data.n_features}</td>
                    <td className="px-4 py-3 text-right font-mono text-primary font-bold">{data.mean?.toFixed(4)}</td>
                    <td className="px-4 py-3 text-right font-mono text-on-surface-variant">{data.std?.toFixed(4)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  )
}


// MultiRepoTab.jsx
export function MultiRepoTab() {
  const [mr, setMr] = useState(null)

  useEffect(() => {
    api.results('multi_repo_results').then(r => setMr(r.data)).catch(() => {})
  }, [])

  if (!mr) return (
     <div className="flex flex-col items-center justify-center mt-20 text-on-surface-variant h-[60vh] glass-panel rounded-xl border border-outline-variant border-dashed">
      <span className="material-symbols-outlined text-4xl mb-3 text-outline-variant">hub</span>
      <p className="text-sm font-medium tracking-wide">NO MULTI-REPO DATA</p>
      <p className="text-xs mt-1 font-mono">Toggle "Multi-repo" in Configuration and execute</p>
    </div>
  )

  const repos   = Object.keys(mr)
  const gaMses  = repos.map(r => mr[r].ga_ann_mse ?? mr[r].ga_best_mse)
  const allMses = repos.map(r => mr[r].all_features_mse)
  const reductions = repos.map(r => mr[r].reduction_pct)

  // Consensus features
  const allSelected = repos.map(r => new Set(mr[r].selected_features || []))
  const common = allSelected.length > 0
    ? [...allSelected.reduce((a, b) => new Set([...a].filter(x => b.has(x))))]
    : []
  const union  = allSelected.length > 0
    ? [...new Set(allSelected.flatMap(s => [...s]))]
    : []

  return (
    <div className="space-y-6 pb-20">
      <div>
        <h2 className="text-3xl font-bold text-on-surface mb-2">Cross-Repository Generalization</h2>
        <p className="text-sm text-on-surface-variant">Validation across distinct ecosystem domains.</p>
      </div>

      {/* Per-repo cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4">
        {repos.map(repo => {
          const d   = mr[repo]
          const mse = d.ga_ann_mse ?? d.ga_best_mse
          return (
            <div key={repo} className="glass-panel rounded-xl px-4 py-5 border-t-2 border-primary">
              <div className="text-lg font-bold text-primary capitalize tracking-widest">{repo}</div>
              <div className="text-[10px] text-on-surface-variant mt-2 uppercase tracking-widest">Deviance (MSE)</div>
              <div className="text-2xl font-mono text-on-surface">{mse?.toFixed(4) ?? 'N/A'}</div>
              <div className="text-xs font-mono text-secondary mt-1 bg-secondary/10 px-2 py-0.5 rounded inline-block">
                {d.n_selected}/{d.n_features_total} feats
              </div>
            </div>
          )
        })}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="glass-panel rounded-xl p-6">
           <h3 className="text-xs font-bold text-primary tracking-widest uppercase mb-4 flex items-center gap-2">
             <span className="material-symbols-outlined text-[16px]">compare_arrows</span>
             Optimization vs Baseline
           </h3>
          <div className="bg-surface-container-lowest border border-outline-variant rounded-lg">
              <Plot
                data={[
                  { x: repos, y: allMses, name: 'Baseline Matrix', type: 'bar', marker: { color: COLORS.muted } },
                  { x: repos, y: gaMses,  name: 'Optimized Matrix',  type: 'bar', marker: { color: COLORS.accent } },
                ]}
                layout={{
                  ...PLOT_LAYOUT, height: 320, barmode: 'group',
                  yaxis: { ...GRID },
                  legend: { orientation: 'h', y: -0.15 },
                  margin: { l: 40, r: 10, t: 20, b: 60 }
                }}
                config={PLOTLY_CONFIG} style={{ width: '100%' }}
              />
          </div>
        </div>
        <div className="glass-panel rounded-xl p-6">
           <h3 className="text-xs font-bold text-primary tracking-widest uppercase mb-4 flex items-center gap-2">
             <span className="material-symbols-outlined text-[16px]">compress</span>
             Dimensionality Reduction
           </h3>
          <div className="bg-surface-container-lowest border border-outline-variant rounded-lg">
              <Plot
                data={[{
                  x: repos, y: reductions, type: 'bar',
                  text: reductions.map(v => `${v?.toFixed(0)}%`),
                  textposition: 'auto',
                  marker: { color: COLORS.teal },
                }]}
                layout={{
                  ...PLOT_LAYOUT, height: 320,
                  yaxis: { ...GRID, range: [0, 100] },
                  margin: { l: 40, r: 10, t: 20, b: 60 }
                }}
                config={PLOTLY_CONFIG} style={{ width: '100%' }}
              />
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-2">
          {common.length > 0 && (
            <div className="glass-panel border-green-500/30 rounded-xl p-5 bg-green-500/5">
                <div className="flex flex-col h-full">
                   <div className="text-green-400 font-bold mb-2 tracking-widest uppercase text-xs flex items-center gap-2">
                       <span className="material-symbols-outlined text-[16px]">all_out</span>
                       Consensus Features
                   </div>
                   <p className="text-[13px] text-green-100/80 mb-3 flex-1">
                      <strong>{common.length}</strong> parameters were definitively selected across ALL environments.
                   </p>
                   <div className="flex flex-wrap gap-1.5">
                       {common.map(c => (
                           <span key={c} className="px-2 py-1 bg-green-500/20 rounded text-[10px] font-mono text-green-300">{c}</span>
                       ))}
                   </div>
                </div>
            </div>
          )}
          <div className="glass-panel border-blue-500/30 rounded-xl p-5 bg-blue-500/5">
            <div className="flex flex-col h-full">
                <div className="text-blue-400 font-bold mb-2 tracking-widest uppercase text-xs flex items-center gap-2">
                    <span className="material-symbols-outlined text-[16px]">device_hub</span>
                    Global Feature Domain
                </div>
                 <p className="text-[13px] text-blue-100/80 mb-3">
                    <strong>{union.length}</strong> distinct dimensions actively leveraged.
                 </p>
                 <div className="mt-auto opacity-70">
                     <span className="material-symbols-outlined text-4xl text-blue-400/50">layers</span>
                 </div>
            </div>
          </div>
      </div>
    </div>
  )
}

// SensitivityTab.jsx
export function SensitivityTab() {
  const [sens, setSens] = useState(null)
  const [popChoice, setPopChoice] = useState(null)
  const [metric, setMetric]       = useState('best_mse')

  useEffect(() => {
    api.results('sensitivity_results').then(r => {
      setSens(r.data)
      if(r.data.pop_sizes && r.data.pop_sizes.length > 0) {
          setPopChoice(String(r.data.pop_sizes[Math.floor(r.data.pop_sizes.length / 2)]))
      }
    }).catch(() => {})
  }, [])

  if (!sens) return (
     <div className="flex flex-col items-center justify-center mt-20 text-on-surface-variant h-[60vh] glass-panel rounded-xl border border-outline-variant border-dashed">
      <span className="material-symbols-outlined text-4xl mb-3 text-outline-variant">tune</span>
      <p className="text-sm font-medium tracking-wide">NO SENSITIVITY SWEEP LOCATED</p>
      <p className="text-xs mt-1 font-mono">Toggle "Sensitivity sweep" in Configuration and execute</p>
    </div>
  )

  const { alphas, betas, pop_sizes, results } = sens
  const pop = popChoice || String(pop_sizes[Math.floor(pop_sizes.length / 2)])

  const z = alphas.map(a =>
    betas.map(b => results[String(a)]?.[String(b)]?.[pop]?.[metric] ?? null)
  )

  return (
    <div className="space-y-6 pb-20">
      <div>
        <h2 className="text-3xl font-bold text-on-surface mb-2">Hyperparameter Sensitivity Matrix</h2>
        <p className="text-sm text-on-surface-variant">System stability across varying evolutionary pressures.</p>
      </div>

      <div className="glass-panel p-4 rounded-xl flex items-center gap-8">
        <div>
          <label className="text-[10px] font-bold text-primary tracking-widest uppercase mb-2 block">Population Constraints</label>
          <div className="flex gap-2 bg-surface-container-lowest p-1 rounded-lg border border-outline-variant w-fit">
            {pop_sizes.map(p => (
              <button key={p}
                onClick={() => setPopChoice(String(p))}
                className={`px-4 py-1.5 rounded-md text-xs font-mono font-bold transition-all ${
                  pop === String(p) ? 'bg-[rgba(186,18,36,0.1)] text-primary border border-primary crimson-glow' : 'text-on-surface-variant hover:text-on-surface hover:bg-surface-container-high border border-transparent'
                }`}
              >{p}</button>
            ))}
          </div>
        </div>
        <div>
          <label className="text-[10px] font-bold text-primary tracking-widest uppercase mb-2 block">Observed Metric</label>
          <div className="flex gap-2 bg-surface-container-lowest p-1 rounded-lg border border-outline-variant w-fit">
            {[['best_mse', 'Deviance (MSE)'], ['n_selected', 'Vector Size']].map(([val, label]) => (
              <button key={val}
                onClick={() => setMetric(val)}
                className={`px-4 py-1.5 rounded-md text-xs font-bold transition-all ${
                  metric === val ? 'bg-[rgba(186,18,36,0.1)] text-primary border border-primary crimson-glow' : 'text-on-surface-variant hover:text-on-surface hover:bg-surface-container-high border border-transparent'
                }`}
              >{label}</button>
            ))}
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="glass-panel rounded-xl p-6">
          <h3 className="text-xs font-bold text-primary tracking-widest uppercase mb-4 flex items-center gap-2">
             <span className="material-symbols-outlined text-[16px]">heatmap</span>
             Pressure Topography
          </h3>
          <div className="bg-surface-container-lowest border border-outline-variant rounded-lg">
              <Plot
                data={[{
                  z, type: 'heatmap',
                  x: betas.map(b => `β=${b}`),
                  y: alphas.map(a => `α=${a}`),
                  colorscale: 'Portland',
                  reversescale: metric === 'best_mse',
                  text: z.map(row => row.map(v =>
                    v === null ? '?' :
                    metric === 'best_mse' ? v.toFixed(4) : String(Math.round(v))
                  )),
                  texttemplate: '%{text}',
                  textfont: { size: 10, color: 'white' },
                }]}
                layout={{
                  ...PLOT_LAYOUT, height: 380,
                  xaxis: { title: 'Beta (parsimony)', ...GRID },
                  yaxis: { title: 'Alpha (accuracy)', ...GRID },
                  margin: { l: 60, r: 20, t: 20, b: 60 }
                }}
                config={PLOTLY_CONFIG} style={{ width: '100%' }}
              />
          </div>
        </div>

        {/* Raw table */}
        <div className="glass-panel rounded-xl p-6 flex flex-col">
          <h3 className="text-xs font-bold text-primary tracking-widest uppercase mb-4 flex items-center gap-2">
             <span className="material-symbols-outlined text-[16px]">table_rows</span>
             Log Matrix
          </h3>
          <div className="bg-surface-container-lowest border border-outline-variant rounded-lg flex-1 overflow-auto max-h-[380px] custom-scrollbar">
            <table className="text-xs w-full text-left border-collapse">
              <thead className="bg-surface-container-high border-b border-outline-variant sticky top-0 z-10 text-on-surface">
                <tr>
                  <th className="px-4 py-3 font-semibold">Alpha (α)</th>
                  <th className="px-4 py-3 font-semibold">Beta (β)</th>
                  <th className="px-4 py-3 font-semibold text-right">Deviance</th>
                  <th className="px-4 py-3 font-semibold text-right">Features</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-outline-variant/30">
                {alphas.flatMap(a => betas.map(b => {
                  const d = results[String(a)]?.[String(b)]?.[pop] || {}
                  return (
                    <tr key={`${a}-${b}`} className="hover:bg-surface-container-high transition-colors">
                      <td className="px-4 py-2 font-mono text-on-surface-variant">{a.toFixed(1)}</td>
                      <td className="px-4 py-2 font-mono text-on-surface-variant">{b.toFixed(1)}</td>
                      <td className="px-4 py-2 text-right font-mono text-primary font-bold">
                        {d.best_mse?.toFixed(4) ?? '—'}
                      </td>
                      <td className="px-4 py-2 text-right font-mono text-on-surface">
                        {d.n_selected ?? '—'}
                      </td>
                    </tr>
                  )
                }))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  )
}