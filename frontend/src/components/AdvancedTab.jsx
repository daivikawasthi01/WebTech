import { useState, useEffect } from 'react'
import Plot from 'react-plotly.js'
import axios from 'axios'
import { api } from '../utils/api.js'
import { PLOT_LAYOUT, GRID, COLORS, PLOTLY_CONFIG } from '../utils/plotTheme.js'

function Panel({ title, icon, children }) {
  return (
    <div className="glass-panel rounded-xl p-6 mb-6">
      <h3 className="text-xs font-bold text-primary tracking-widest uppercase mb-4 flex items-center gap-2">
         <span className="material-symbols-outlined text-[16px]">{icon}</span>
         {title}
      </h3>
      {children}
    </div>
  )
}

function ParetoPanel() {
  const [data,    setData]    = useState(null)
  const [loading, setLoading] = useState(false)

  const load = () => {
    setLoading(true)
    axios.get('/api/pareto').then(r => setData(r.data)).catch(() => {}).finally(() => setLoading(false))
  }

  useEffect(() => { load() }, [])

  return (
    <Panel title="Pareto Front — MSE vs Feature Volume" icon="show_chart">
      <p className="text-sm text-on-surface-variant mb-6">
        Non-dominated convergence vectors. Purple lines represent Pareto-efficient optimization paths; grey markers indicate sub-optimal dominance matrices.
      </p>
      {loading && <p className="text-primary font-mono text-xs">LOADING TELEMETRY...</p>}
      {data && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2 bg-surface-container-lowest border border-outline-variant rounded-lg">
            <Plot
              data={[
                {
                  x: data.all_points.map(p => p.n_features),
                  y: data.all_points.map(p => p.mse),
                  text: data.all_points.map(p => `Gen ${p.generation}`),
                  mode: 'markers', type: 'scatter', name: 'All iterations',
                  marker: { color: COLORS.muted, size: 6, opacity: 0.3, line: { width: 0 } },
                },
                {
                  x: data.pareto_front.map(p => p.n_features),
                  y: data.pareto_front.map(p => p.mse),
                  text: data.pareto_front.map(p => `Gen ${p.generation}`),
                  mode: 'lines+markers', type: 'scatter', name: 'Pareto Front',
                  line: { color: COLORS.accent, width: 2, shape: 'spline' },
                  marker: { color: COLORS.accent, size: 8, symbol: 'diamond', line: { color: 'white', width: 1 } },
                },
              ]}
              layout={{
                ...PLOT_LAYOUT, height: 320,
                xaxis: { title: 'Dimensionality (Features)', ...GRID },
                yaxis: { title: 'Deviance (MSE)',          ...GRID },
                legend: { orientation: 'h', y: -0.25 },
                margin: { l: 40, r: 20, t: 20, b: 60 }
              }}
              config={PLOTLY_CONFIG} style={{ width: '100%' }}
            />
          </div>

          <div className="flex flex-col gap-4">
             <div className="grid grid-cols-2 gap-4">
                <div className="bg-surface-container-lowest border border-outline-variant rounded-lg p-3">
                   <div className="text-[10px] text-on-surface-variant tracking-widest uppercase mb-1">Efficient Nodes</div>
                   <div className="text-xl font-mono text-primary font-bold">{data.n_pareto}</div>
                </div>
                <div className="bg-surface-container-lowest border border-outline-variant rounded-lg p-3">
                   <div className="text-[10px] text-on-surface-variant tracking-widest uppercase mb-1">Total Solved</div>
                   <div className="text-xl font-mono text-on-surface font-bold">{data.n_total}</div>
                </div>
             </div>
             
             {/* Pareto table */}
             <div className="flex-1 overflow-x-auto custom-scrollbar bg-surface-container-lowest border border-outline-variant rounded-lg max-h-48">
              <table className="text-xs w-full">
                <thead className="bg-surface-container-high border-b border-outline-variant sticky top-0">
                  <tr>
                    <th className="text-left py-2 px-3 font-semibold text-on-surface-variant">GEN</th>
                    <th className="text-right py-2 px-3 font-semibold text-on-surface-variant">DIM</th>
                    <th className="text-right py-2 px-3 font-semibold text-on-surface-variant">DEVIANCE</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-outline-variant/30">
                  {data.pareto_front.map((p, i) => (
                    <tr key={i} className="hover:bg-surface-container transition-colors">
                      <td className="py-2 px-3 text-on-surface-variant font-mono">{p.generation}</td>
                      <td className="py-2 px-3 text-right font-mono text-on-surface font-bold">{p.n_features}</td>
                      <td className="py-2 px-3 text-right font-mono text-primary font-bold">{p.mse.toFixed(4)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
             </div>
          </div>
        </div>
      )}
      {!data && !loading && (
        <div className="border border-outline-variant border-dashed p-6 rounded-lg text-center">
            <span className="material-symbols-outlined text-outline-variant text-4xl mb-2">trending_up</span>
            <p className="text-on-surface-variant text-sm font-medium">No Pareto generation data found. Run optimization sequence.</p>
        </div>
      )}
    </Panel>
  )
}

function DiversityPanel() {
  const [data,    setData]    = useState(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    setLoading(true)
    axios.get('/api/diversity').then(r => setData(r.data)).catch(() => {}).finally(() => setLoading(false))
  }, [])

  return (
    <Panel title="Evolutionary Diversity Dynamics" icon="diversity_3">
      <p className="text-sm text-on-surface-variant mb-6">
        Population variance tracked against global minimum deviance. Stagnation events trigger evolutionary bursts or early stops.
      </p>
      {loading && <p className="text-primary font-mono text-xs">LOADING TELEMETRY...</p>}
      {data && data.generations.length > 0 && (
         <div className="bg-surface-container-lowest border border-outline-variant rounded-lg">
            <Plot
            data={[
                {
                x: data.generations, y: data.diversity_proxy,
                name: 'Variance Index', type: 'scatter', mode: 'lines+markers',
                line: { color: COLORS.teal, width: 2 },
                marker: { size: 4, color: COLORS.teal },
                yaxis: 'y',
                },
                {
                x: data.generations, y: data.mse_trace,
                name: 'Min Deviance', type: 'scatter', mode: 'lines',
                line: { color: COLORS.accent, width: 2, dash: 'dot' },
                yaxis: 'y2',
                },
                {
                x: data.generations.filter((_, i) => data.stagnation_flags[i] === 1),
                y: data.diversity_proxy.filter((_, i) => data.stagnation_flags[i] === 1),
                name: 'Stagnation Vector', type: 'scatter', mode: 'markers',
                marker: { color: COLORS.red, size: 8, symbol: 'x', line: { width: 1, color: 'white' } },
                yaxis: 'y',
                },
            ]}
            layout={{
                ...PLOT_LAYOUT, height: 320,
                yaxis:  { title: 'Variance Index', ...GRID },
                yaxis2: { title: 'Deviance (MSE)', overlaying: 'y', side: 'right', gridcolor: 'rgba(255,255,255,0.05)' },
                xaxis:  { title: 'Evolutionary Iteration', ...GRID },
                legend: { orientation: 'h', y: -0.25 },
                margin: { l: 40, r: 40, t: 20, b: 60 }
            }}
            config={PLOTLY_CONFIG} style={{ width: '100%' }}
            />
         </div>
      )}
      {!data && !loading && (
        <div className="border border-outline-variant border-dashed p-6 rounded-lg text-center">
            <span className="material-symbols-outlined text-outline-variant text-4xl mb-2">all_inclusive</span>
            <p className="text-on-surface-variant text-sm font-medium">No diversity telemetry found. Ensure pipeline logging enabled.</p>
        </div>
      )}
    </Panel>
  )
}

function BootstrapPanel() {
  const [bl,      setBl]      = useState(null)
  const [result,  setResult]  = useState(null)
  const [loading, setLoading] = useState(false)
  const [nBoot,   setNBoot]   = useState(10000)
  const [ciLevel, setCiLevel] = useState(0.95)

  useEffect(() => {
    api.results('baseline_results').then(r => setBl(r.data)).catch(() => {})
  }, [])

  const run = async () => {
    if (!bl?.ga_selected?.mses || !bl?.all_features?.mses) return
    setLoading(true)
    try {
      const res = await axios.post('/api/stats/bootstrap', {
        ga_mses:  bl.ga_selected.mses,
        all_mses: bl.all_features.mses,
        n_boot:   nBoot,
        ci_level: ciLevel,
      })
      setResult(res.data)
    } finally {
      setLoading(false)
    }
  }

  return (
    <Panel title={`Bootstrap Confidence Testing (${(ciLevel * 100).toFixed(0)}%)`} icon="query_stats">
      <p className="text-sm text-on-surface-variant mb-6">
        Hypothesis validation: Ensure deviance contraction from vector selection is statistically robust against baseline parameters.
      </p>

      {!bl ? (
        <div className="border border-outline-variant border-dashed p-6 rounded-lg text-center">
            <span className="material-symbols-outlined text-outline-variant text-4xl mb-2">fact_check</span>
            <p className="text-on-surface-variant text-sm font-medium">No baseline telemetry to compare. Run Baseline module.</p>
        </div>
      ) : (
        <>
          <div className="flex flex-col md:flex-row items-end gap-6 mb-8 p-4 bg-surface-container-lowest border border-outline-variant rounded-lg">
            <div>
              <label className="text-[10px] font-bold text-primary tracking-widest uppercase mb-2 block">Resampling Iterations</label>
              <select value={nBoot} onChange={e => setNBoot(parseInt(e.target.value))}
                      className="bg-surface-container text-on-surface border border-outline-variant focus:border-primary px-3 py-2 rounded-md outline-none text-sm font-mono transition-colors w-32">
                {[1000, 5000, 10000, 50000].map(n =>
                  <option key={n} value={n}>{n.toLocaleString()}</option>
                )}
              </select>
            </div>
            <div>
              <label className="text-[10px] font-bold text-primary tracking-widest uppercase mb-2 block">Alpha Level</label>
              <select value={ciLevel} onChange={e => setCiLevel(parseFloat(e.target.value))}
                      className="bg-surface-container text-on-surface border border-outline-variant focus:border-primary px-3 py-2 rounded-md outline-none text-sm font-mono transition-colors w-32">
                {[0.90, 0.95, 0.99].map(v =>
                  <option key={v} value={v}>{(v*100).toFixed(0)}% CI</option>
                )}
              </select>
            </div>
            <button onClick={run} disabled={loading}
                    className="ml-auto px-6 py-2.5 bg-[rgba(186,18,36,0.1)] hover:bg-[rgba(186,18,36,0.2)] text-primary font-bold text-sm border border-primary transition-all flex items-center gap-2 crimson-glow disabled:opacity-40 disabled:cursor-not-allowed rounded-lg">
              <span className="material-symbols-outlined text-[20px]">{loading ? 'hourglass_empty' : 'bolt'}</span>
              {loading ? 'SIMULATING ITERATIONS...' : 'EXECUTE BOOTSTRAP'}
            </button>
          </div>

          {result && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Significance banner and numbers */}
               <div className="flex flex-col gap-4">
                  <div className={`p-5 rounded-xl border flex gap-4 ${
                    result.significant
                      ? 'bg-green-500/5 border-green-500/30'
                      : 'bg-yellow-500/5 border-yellow-500/30'
                  }`}>
                    <span className={`material-symbols-outlined text-3xl flex-none ${result.significant ? 'text-green-400' : 'text-yellow-400'}`}>{result.significant ? 'verified' : 'help_center'}</span>
                    <div>
                        <div className={`font-bold tracking-widest uppercase text-xs mb-2 ${result.significant ? 'text-green-400' : 'text-yellow-400'}`}>
                            {result.significant ? 'Validation Successful' : 'Inconclusive Advantage'}
                        </div>
                        <div className="flex flex-col gap-1 font-mono text-[11px] text-on-surface-variant">
                             <span>p-value: <strong className="text-on-surface">{result.p_value.toFixed(4)}</strong></span>
                             <span>Delta: <strong className="text-on-surface">{result.mean_diff.toFixed(4)}</strong> Dev Units</span>
                             <span>CI Range: <strong className="text-on-surface">[{result.diff_ci[0].toFixed(4)}, {result.diff_ci[1].toFixed(4)}]</strong></span>
                        </div>
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-4 flex-1">
                    {[
                      { label: 'Optimized Model', mean: result.ga_mean,  ci: result.ga_ci },
                      { label: 'Baseline Matrix',mean: result.all_mean, ci: result.all_ci },
                    ].map(({ label, mean, ci }) => (
                      <div key={label} className="bg-surface-container-lowest border border-outline-variant p-4 rounded-xl flex flex-col justify-center">
                        <div className="text-[10px] uppercase tracking-widest font-bold text-on-surface-variant mb-2">{label}</div>
                        <div className="text-2xl font-mono text-primary font-bold">{mean.toFixed(4)}</div>
                        <div className="text-[10px] font-mono text-on-surface-variant mt-2 border-t border-outline-variant pt-2">
                          CI: [{ci[0].toFixed(4)}, {ci[1].toFixed(4)}]
                        </div>
                      </div>
                    ))}
                  </div>
               </div>

              {/* CI plot */}
              <div className="bg-surface-container-lowest border border-outline-variant rounded-lg p-2">
                  <Plot
                    data={[
                      {
                        x: ['Optimized Model', 'Baseline Matrix'],
                        y: [result.ga_mean, result.all_mean],
                        error_y: {
                          type:    'data',
                          array:   [result.ga_mean - result.ga_ci[0],
                                    result.all_mean - result.all_ci[0]],
                          arrayminus: [result.ga_ci[1] - result.ga_mean,
                                      result.all_ci[1] - result.all_mean],
                          visible: true,
                          color: '#aa8986',
                          thickness: 2,
                          width: 8
                        },
                        type: 'bar',
                        marker: { color: [COLORS.accent, COLORS.muted] },
                      },
                    ]}
                    layout={{
                      ...PLOT_LAYOUT, height: 280,
                      yaxis: { ...GRID, title: 'Mean Deviance' },
                      showlegend: false,
                      margin: { l: 40, r: 10, t: 30, b: 40 }
                    }}
                    config={PLOTLY_CONFIG} style={{ width: '100%' }}
                  />
              </div>
            </div>
          )}
        </>
      )}
    </Panel>
  )
}

export default function AdvancedTab() {
  return (
    <div className="space-y-6 pb-20">
      <div>
        <h2 className="text-3xl font-bold text-on-surface mb-2">Deep Telemetry & Analytics</h2>
        <p className="text-sm text-on-surface-variant">
          Advanced statistical validation, convergence modeling, and topological analysis.
        </p>
      </div>
      <ParetoPanel />
      <DiversityPanel />
      <BootstrapPanel />
    </div>
  )
}