import { useState, useEffect } from 'react'
import Plot from 'react-plotly.js'
import { api } from '../utils/api.js'
import { PLOT_LAYOUT, GRID, COLORS, CATEGORY_COLORS, PLOTLY_CONFIG } from '../utils/plotTheme.js'
import LiveGAProgress from './LiveGAProgress.jsx'

const FEATURE_CATEGORY = {
  avg_cyclomatic_complexity: 'Structural', halstead_volume: 'Structural',
  halstead_effort: 'Structural', depth_of_inheritance_tree: 'Structural',
  number_of_methods_per_class: 'Structural', weighted_methods_per_class: 'Structural',
  nesting_depth: 'Structural', class_coupling: 'Structural',
  comment_density: 'Textual', whitespace_ratio: 'Textual',
  docstring_presence: 'Textual', avg_identifier_length: 'Textual',
  code_duplication_pct: 'Textual',
  commit_frequency: 'Evolutionary', author_count: 'Evolutionary',
  bug_fix_ratio: 'Evolutionary', code_age_days: 'Evolutionary',
  code_churn: 'Evolutionary', added_deleted_ratio: 'Evolutionary',
}

function MetricCard({ label, value, sub }) {
  return (
    <div className="glass-panel rounded-xl p-6 border-l-2 border-primary relative overflow-hidden group">
        <div className="absolute inset-0 bg-primary opacity-0 group-hover:opacity-5 transition-opacity"></div>
        <div className="text-xs font-bold text-on-surface-variant tracking-widest uppercase mb-1">{label}</div>
        <div className="text-3xl font-mono text-on-surface">{value}</div>
        {sub && <div className="text-[10px] text-primary/80 mt-2 tracking-wider font-medium">{sub}</div>}
    </div>
  )
}

export default function GAResultsTab({ liveActive }) {
  const [ga,      setGa]      = useState(null)
  const [loading, setLoading] = useState(true)

  // Poll ga_results.json every 3s while a job is active
  useEffect(() => {
    const load = () => {
      api.results('ga_results').then(r => setGa(r.data)).catch(() => {}).finally(() => setLoading(false))
    }
    load()
    if (!liveActive) return
    const t = setInterval(load, 3000)
    return () => clearInterval(t)
  }, [liveActive])

  if (loading) return (
      <div className="flex flex-col items-center justify-center h-[60vh]">
          <span className="material-symbols-outlined text-primary text-6xl pulse-glow mb-4">troubleshoot</span>
          <p className="text-on-surface-variant text-sm tracking-widest font-mono">LOADING TELEMETRY...</p>
      </div>
  )
  if (!ga)     return (
    <div className="flex flex-col items-center justify-center mt-20 text-on-surface-variant h-[60vh] glass-panel rounded-xl border border-outline-variant border-dashed">
      <span className="material-symbols-outlined text-4xl mb-3 text-outline-variant">science</span>
      <p className="text-sm font-medium tracking-wide">NO OPTIMIZATION DATA AVAILABLE</p>
      <p className="text-xs mt-1 font-mono">Initiate sequence from Pipeline Execution</p>
    </div>
  )

  const history   = ga.history || []
  const chrom     = ga.chromosome || []
  const n_sel     = ga.n_selected || 0
  const n_tot     = ga.n_total   || 1
  const reduction = ((1 - n_sel / n_tot) * 100).toFixed(0)
  const elapsed   = ga.elapsed_s ?? null

  const selectedFeatures = (ga.feature_names || ga.selected_features || []).map(name => ({
    name,
    category: FEATURE_CATEGORY[name] || 'Unknown',
  }))

  const gens       = history.map(h => h.generation)
  const bestMse    = history.map(h => h.global_best_mse ?? h.best_mse)
  const genMse     = history.map(h => h.best_mse)
  const mutRate    = history.map(h => h.mutation_rate)
  const nFeats     = history.map(h => h.n_features)

  return (
    <div className="space-y-6 pb-20">
      <div>
        <h1 className="text-3xl font-bold text-on-surface mb-2">Genetic Algorithm Analytics</h1>
        <p className="text-sm text-on-surface-variant">Diagnostics and convergence telemetry of the neural-evolutionary process.</p>
      </div>

      {liveActive && <LiveGAProgress active={liveActive} />}

      {/* Primary Metrics Layer */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <MetricCard label="Best Fitness"   value={ga.best_fitness?.toFixed(4)} />
        <MetricCard label="Best MSE"       value={ga.best_mse?.toFixed(4)} />
        <MetricCard label="Feature Vector" value={`${n_sel}/${n_tot}`} sub={`${reduction}% DIMENSIONALITY REDUCTION`} />
        <MetricCard label="Elapsed Time"   value={elapsed !== null ? `${elapsed}s` : `${(ga.history?.length || 0)} gens`} />
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        {/* Analytics Core */}
        {history.length > 0 && (
          <div className="xl:col-span-2 glass-panel rounded-xl flex flex-col p-6">
            <h3 className="text-xs font-bold text-primary tracking-widest uppercase mb-6 flex items-center gap-2">
                <span className="material-symbols-outlined text-[16px]">monitoring</span> 
                Convergence Matrix
            </h3>
            <div className="flex-1 w-full bg-surface-container-lowest border border-outline-variant rounded-lg relative custom-scrollbar">
                <div className="absolute top-4 left-4 z-10 flex gap-4 text-[10px] font-mono">
                    <span className="flex items-center gap-1 text-primary"><span className="w-2 h-2 rounded-full bg-primary"></span> GLOBAL BEST MSE</span>
                    <span className="flex items-center gap-1 text-secondary"><span className="w-2 h-2 border border-secondary border-dashed"></span> GEN BEST MSE</span>
                </div>
                <Plot
                  data={[
                    {
                      x: gens, y: bestMse, name: 'Global Best MSE',
                      type: 'scatter', mode: 'lines+markers',
                      line: { color: COLORS.accent, width: 2, shape: 'spline' },
                      marker: { size: 6, color: COLORS.accent, line: { color: COLORS.accent, width: 2 } }
                    },
                    {
                      x: gens, y: genMse, name: 'Gen Best MSE',
                      type: 'scatter', mode: 'lines',
                      line: { color: COLORS.light, width: 1.5, dash: 'dot', shape: 'spline' },
                    },
                    {
                      x: gens, y: mutRate, name: 'Mutation Rate',
                      type: 'scatter', mode: 'lines', yaxis: 'y2',
                      line: { color: COLORS.yellow, width: 1, dash: 'dash' },
                    },
                  ]}
                  layout={{
                    ...PLOT_LAYOUT,
                    height: 280,
                    yaxis:  { title: '', ...GRID },
                    yaxis2: { title: 'Mutation Rate', overlaying: 'y', side: 'right', showgrid: false },
                    xaxis:  { title: 'Generations', ...GRID },
                    showlegend: false,
                    margin: { l: 40, r: 40, t: 30, b: 40 }
                  }}
                  config={PLOTLY_CONFIG}
                  style={{ width: '100%' }}
                />
            </div>
            
            <div className="mt-4 w-full bg-surface-container-lowest border border-outline-variant rounded-lg">
                <Plot
                  data={[{
                    x: gens, y: nFeats, type: 'bar',
                    marker: { color: 'rgba(255, 179, 177, 0.4)', line: { color: COLORS.accent, width: 1 } },
                    name: 'Features Selected',
                  }]}
                  layout={{
                    ...PLOT_LAYOUT,
                    height: 120,
                    yaxis:  { title: '', ...GRID },
                    xaxis:  { title: '', showticklabels: false, ...GRID },
                    margin: { l: 40, r: 40, t: 10, b: 10 },
                    showlegend: false
                  }}
                  config={PLOTLY_CONFIG}
                  style={{ width: '100%' }}
                />
            </div>
          </div>
        )}

        {/* Selected Features Panel */}
        <div className="glass-panel rounded-xl flex flex-col h-[500px]">
          <div className="p-4 border-b border-outline-variant flex items-center justify-between">
            <h3 className="text-xs font-bold text-primary tracking-widest uppercase">Evolved Vector ({n_sel})</h3>
          </div>
          <div className="flex-1 overflow-y-auto p-4 custom-scrollbar">
            <div className="space-y-2">
              {selectedFeatures.length > 0 ? selectedFeatures.map(({ name, category }) => (
                <div key={name} className="flex flex-col gap-1 p-3 rounded-lg bg-surface-container-lowest border border-outline-variant hover:border-primary transition-colors">
                  <div className="flex items-center justify-between">
                      <span className="text-xs text-on-surface font-mono font-bold truncate max-w-[80%]">{name}</span>
                      <span className="w-2 h-2 rounded-full" style={{ background: CATEGORY_COLORS[category] }}></span>
                  </div>
                  <span className="text-[10px] text-on-surface-variant font-medium tracking-wide uppercase">{category}</span>
                </div>
              )) : (
                <div className="font-mono text-xs grid grid-cols-8 gap-1">
                  {chrom.map((bit, i) => (
                    <div
                      key={i}
                      className={`flex aspect-square items-center justify-center rounded-md border ${
                        bit ? 'bg-[rgba(186,18,36,0.2)] text-primary border-primary' : 'bg-surface-container border-outline-variant text-on-surface-variant'
                      }`}
                      title={`Feature ${i}`}
                    >
                      {i}
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* Category breakdown Footer */}
          <div className="p-4 border-t border-outline-variant bg-surface-container-lowest">
            <h4 className="text-[10px] text-on-surface-variant mb-2 font-bold tracking-widest uppercase">Class Distribution</h4>
            <div className="grid grid-cols-3 gap-2">
                {['Structural', 'Textual', 'Evolutionary'].map(cat => {
                  const count = selectedFeatures.filter(f => f.category === cat).length
                  return (
                    <div key={cat} className="flex flex-col items-center p-2 rounded bg-surface-container border border-outline-variant">
                        <span className="text-[10px] text-on-surface-variant font-medium">{cat.substring(0,4)}</span>
                        <span className="text-sm font-bold" style={{ color: CATEGORY_COLORS[cat] }}>{count}</span>
                    </div>
                  )
                })}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}