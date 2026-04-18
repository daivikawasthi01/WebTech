import { useState, useEffect } from 'react'
import Plot from 'react-plotly.js'
import { api } from '../utils/api.js'
import { PLOT_LAYOUT, GRID, COLORS, PLOTLY_CONFIG } from '../utils/plotTheme.js'

const RISK_COLORS = { High: COLORS.red, Medium: COLORS.yellow, Low: COLORS.teal }

export default function FileRiskTab({ cfg }) {
  const [ga,        setGa]        = useState(null)
  const [result,    setResult]    = useState(null)
  const [loading,   setLoading]   = useState(false)
  const [error,     setError]     = useState(null)
  const [filter,    setFilter]    = useState(['High', 'Medium', 'Low'])
  const [sortKey,   setSortKey]   = useState('predicted_score')
  const [sortAsc,   setSortAsc]   = useState(false)

  const [featureCount, setFeatureCount] = useState(null)

  useEffect(() => {
    api.results('ga_results').then(r => setGa(r.data)).catch(() => {})
    api.featureNames(cfg.processed_file).then(r => {
      setFeatureCount(r.data.n_features)
    }).catch(() => {})
  }, [cfg.processed_file])

  const runPrediction = async () => {
    if (!ga) return
    setLoading(true)
    setError(null)
    try {
      // Truncate chromosome to match actual feature count
      let chrom = ga.chromosome
      if (featureCount && chrom.length > featureCount) {
        chrom = chrom.slice(0, featureCount)
      }
      const res = await api.predict({
        processed_file: cfg.processed_file,
        chromosome:     chrom,
      })
      setResult(res.data)
    } catch (e) {
      setError(e.response?.data?.detail || e.message)
    } finally {
      setLoading(false)
    }
  }

  if (!ga) return (
     <div className="flex flex-col items-center justify-center mt-20 text-on-surface-variant h-[60vh] glass-panel rounded-xl border border-outline-variant border-dashed">
      <span className="material-symbols-outlined text-4xl mb-3 text-outline-variant">gpp_bad</span>
      <p className="text-sm font-medium tracking-wide">NO VECTOR DISCOVERED</p>
      <p className="text-xs mt-1 font-mono">Run the GA optimizer in Pipeline first.</p>
    </div>
  )

  const files = result?.files || []
  const filtered = files
    .filter(f => filter.includes(f.risk_level))
    .sort((a, b) => {
      const av = a[sortKey], bv = b[sortKey]
      return sortAsc ? (av > bv ? 1 : -1) : (av < bv ? 1 : -1)
    })

  const handleSort = (key) => {
    if (sortKey === key) setSortAsc(a => !a)
    else { setSortKey(key); setSortAsc(false) }
  }

  const maxVal = result
    ? Math.max(...files.map(f => Math.max(f.true_bugs, f.predicted_score))) + 1
    : 10

  return (
    <div className="space-y-6 pb-20">
      <div>
        <h2 className="text-3xl font-bold text-on-surface mb-2">Architectural Risk Surface</h2>
        <p className="text-sm text-on-surface-variant">Vulnerability localization driven by optimized telemetry vectors.</p>
      </div>

      <div className="glass-panel p-4 rounded-xl flex items-center justify-between gap-4">
        <button
          onClick={runPrediction}
          disabled={loading}
          className="py-3 px-6 rounded-lg bg-[rgba(186,18,36,0.1)] hover:bg-[rgba(186,18,36,0.2)] text-primary font-bold text-sm border border-primary transition-all flex items-center gap-2 crimson-glow disabled:opacity-40 disabled:cursor-not-allowed"
        >
          <span className="material-symbols-outlined text-[20px]">{loading ? 'hourglass_empty' : 'radar'}</span>
          {loading ? 'ANALYZING...' : 'SCAN ARCHITECTURE'}
        </button>
        {result && (
          <div className="flex gap-4 text-xs font-mono items-center bg-surface-container-lowest border border-outline-variant rounded-lg px-4 py-2">
            <span className="text-on-surface-variant flex gap-2 items-center"><span className="text-on-surface font-bold">MSE:</span> <strong className="text-secondary">{result.mse?.toFixed(4)}</strong></span>
            {Object.entries(result.summary).map(([k, v]) => (
              <span key={k} className="flex gap-2 items-center" style={{ color: RISK_COLORS[k.charAt(0).toUpperCase() + k.slice(1)] }}>
                <span className="text-on-surface-variant font-bold">{k.charAt(0).toUpperCase() + k.slice(1)}:</span> <strong>{v}</strong>
              </span>
            ))}
          </div>
        )}
        {error && <span className="text-primary text-xs font-mono border border-primary bg-primary/10 px-4 py-2 rounded-lg">{error}</span>}
      </div>

      {result && (
        <>
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Scatter: True vs Predicted */}
            <div className="lg:col-span-2 glass-panel rounded-xl p-6">
              <h3 className="text-xs font-bold text-primary tracking-widest uppercase mb-4 flex items-center gap-2">
                 <span className="material-symbols-outlined text-[16px]">scatter_plot</span>
                 Defect Projection Accuracy
               </h3>
              <div className="bg-surface-container-lowest border border-outline-variant rounded-lg">
                <Plot
                  data={[
                    ...['High', 'Medium', 'Low'].map(level => ({
                      x: files.filter(f => f.risk_level === level).map(f => f.true_bugs),
                      y: files.filter(f => f.risk_level === level).map(f => f.predicted_score),
                      text: files.filter(f => f.risk_level === level).map(f => f.file),
                      name: level, type: 'scatter', mode: 'markers',
                      marker: { color: RISK_COLORS[level], size: 8, opacity: 0.8, line: { color: '#000', width: 1 } },
                    })),
                    {
                      x: [0, maxVal], y: [0, maxVal], name: 'Perfect Resonance',
                      type: 'scatter', mode: 'lines',
                      line: { dash: 'dash', color: COLORS.muted, width: 1.5, shape: 'spline' },
                      showlegend: false,
                    },
                  ]}
                  layout={{
                    ...PLOT_LAYOUT, height: 320,
                    xaxis: { title: 'Ground Truth Defects', ...GRID },
                    yaxis: { title: 'Predicted Defect Volume', ...GRID },
                    legend: { orientation: 'h', y: -0.25 },
                    margin: { l: 40, r: 20, t: 20, b: 60 }
                  }}
                  config={PLOTLY_CONFIG} style={{ width: '100%' }}
                />
              </div>
            </div>

            {/* Top 10 riskiest */}
            <div className="glass-panel rounded-xl p-6 flex flex-col h-[400px]">
              <h3 className="text-xs font-bold text-primary tracking-widest uppercase mb-4 flex items-center gap-2">
                 <span className="material-symbols-outlined text-[16px]">warning</span>
                 Critical Subsystems (Top 10)
               </h3>
              <div className="flex-1 space-y-2 overflow-y-auto custom-scrollbar pr-2">
                {files.slice(0, 10).map((f, i) => (
                  <div key={f.file} className="flex items-center gap-3 p-3 text-xs bg-surface-container-lowest border border-outline-variant rounded-lg hover:border-primary transition-colors cursor-default group">
                    <span className="text-on-surface-variant font-mono w-4 font-bold">{i+1}</span>
                    <span
                      className="w-2 h-2 rounded-full flex-none shadow-[0_0_8px_currentColor]"
                      style={{ background: RISK_COLORS[f.risk_level], color: RISK_COLORS[f.risk_level] }}
                    />
                    <span className="text-on-surface font-mono flex-1 truncate" title={f.file}>
                      {f.file.split('/').pop()}
                    </span>
                    <span className="font-mono font-bold" style={{ color: RISK_COLORS[f.risk_level] }}>
                      {f.predicted_score.toFixed(2)}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Filterable table */}
          <div className="glass-panel rounded-xl p-6 flex flex-col">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-xs font-bold text-primary tracking-widest uppercase flex items-center gap-2">
                 <span className="material-symbols-outlined text-[16px]">folder_copy</span>
                 Full System Audit
               </h3>
              <div className="flex items-center gap-4">
                <div className="flex gap-2">
                  {['High', 'Medium', 'Low'].map(level => (
                    <button
                      key={level}
                      onClick={() => setFilter(f =>
                        f.includes(level) ? f.filter(x => x !== level) : [...f, level]
                      )}
                      className={`px-3 py-1.5 rounded-md text-[10px] font-bold uppercase tracking-wider transition-colors border ${
                        filter.includes(level)
                          ? 'border-transparent'
                          : 'bg-surface-container-lowest'
                      }`}
                      style={{
                        background: filter.includes(level) ? `${RISK_COLORS[level]}22` : undefined,
                        color: RISK_COLORS[level],
                        borderColor: RISK_COLORS[level],
                      }}
                    >
                      {level}
                    </button>
                  ))}
                </div>
                <span className="text-[10px] font-mono text-on-surface-variant bg-surface-container-high px-2 py-1 rounded">{filtered.length} NODES</span>
              </div>
            </div>

            <div className="bg-surface-container-lowest border border-outline-variant rounded-lg flex-1 overflow-x-auto max-h-96 custom-scrollbar">
              <table className="text-xs w-full text-left border-collapse">
                <thead className="sticky top-0 bg-surface-container-high z-10 border-b border-outline-variant shadow-md">
                  <tr>
                    {[
                      ['file',            'File Path'],
                      ['risk_level',      'Threat Level'],
                      ['predicted_score', 'Predicted Vol'],
                      ['true_bugs',       'Ground Truth'],
                    ].map(([key, label]) => (
                      <th
                        key={key}
                        onClick={() => handleSort(key)}
                        className="px-4 py-3 text-on-surface-variant font-semibold cursor-pointer hover:text-primary select-none transition-colors"
                      >
                        {label}
                        <span className="ml-1 text-[10px] text-primary">{sortKey === key ? (sortAsc ? '▲' : '▼') : ''}</span>
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-outline-variant/30">
                  {filtered.map((f, i) => (
                    <tr key={f.file} className="hover:bg-surface-container-high transition-colors">
                      <td className="px-4 py-2 font-mono text-on-surface max-w-xs truncate" title={f.file}>
                        {f.file}
                      </td>
                      <td className="px-4 py-2 font-semibold uppercase tracking-wider" style={{ color: RISK_COLORS[f.risk_level] }}>
                        {f.risk_level}
                      </td>
                      <td className="px-4 py-2 font-mono text-primary font-bold">{f.predicted_score.toFixed(3)}</td>
                      <td className="px-4 py-2 text-on-surface-variant font-mono">{f.true_bugs}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </div>
  )
}