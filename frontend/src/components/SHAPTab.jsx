import { useState, useEffect } from 'react'
import Plot from 'react-plotly.js'
import { api } from '../utils/api.js'
import { PLOT_LAYOUT, GRID, COLORS, CATEGORY_COLORS, PLOTLY_CONFIG } from '../utils/plotTheme.js'
import axios from 'axios'

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

export default function SHAPTab({ cfg }) {
  const [ga,       setGa]      = useState(null)
  const [result,   setResult]  = useState(null)
  const [loading,  setLoading] = useState(false)
  const [error,    setError]   = useState(null)
  const [nBg,      setNBg]     = useState(50)
  const [featureCount, setFeatureCount] = useState(null)

  useEffect(() => {
    api.results('ga_results').then(r => setGa(r.data)).catch(() => {})
    api.featureNames(cfg.processed_file).then(r => {
      setFeatureCount(r.data.n_features)
    }).catch(() => {})
  }, [cfg.processed_file])

  const run = async () => {
    if (!ga) return
    setLoading(true); setError(null)
    try {
      // Truncate chromosome to match actual feature count
      let chrom = ga.chromosome
      if (featureCount && chrom.length > featureCount) {
        chrom = chrom.slice(0, featureCount)
      }
      const res = await axios.post('/api/shap', {
        processed_file: cfg.processed_file,
        chromosome:     chrom,
        n_background:   nBg,
      })
      setResult(res.data)
    } catch (e) {
      const msg = e.response?.data?.detail || e.message
      setError(msg.includes('Missing dependency')
        ? 'SHAP not installed. Run: pip install shap'
        : msg)
    } finally {
      setLoading(false)
    }
  }

  if (!ga) return (
     <div className="flex flex-col items-center justify-center mt-20 text-on-surface-variant h-[60vh] glass-panel rounded-xl border border-outline-variant border-dashed">
      <span className="material-symbols-outlined text-4xl mb-3 text-outline-variant">psychology</span>
      <p className="text-sm font-medium tracking-wide">NO VECTOR SEQUENCE AVAILABLE</p>
      <p className="text-xs mt-1 font-mono">Run the GA optimizer first to extract features.</p>
    </div>
  )

  const ranked  = result?.ranked  || []
  const matrix  = result?.shap_matrix || []     // (n_samples, n_feats)
  const fvals   = result?.feature_values || []  // (n_samples, n_feats)
  const names   = result?.feature_names  || []

  const beeswarmTrace = (() => {
    if (!matrix.length || !names.length) return null
    const xs = [], ys = [], colors = []
    names.forEach((name, fi) => {
      const shapCol = matrix.map(row => row[fi])
      const valCol  = fvals.map(row => row[fi])
      const maxVal  = Math.max(...valCol.map(Math.abs)) || 1
      shapCol.forEach((sv, si) => {
        xs.push(sv)
        ys.push(name)
        colors.push(valCol[si] / maxVal)   // normalised to [-1, 1] -> colorscale maps it
      })
    })
    return {
      x: xs, y: ys,
      mode: 'markers',
      type: 'scatter',
      showlegend: false,
      marker: {
        color:        colors,
        colorscale:   'RdBu',
        reversescale: true,
        size:         4,
        opacity:      0.65,
        colorbar: {
          title:     'Feature Data',
          titleside: 'right',
          len:       0.5,
          thickness: 10,
          tickfont:  { color: '#aa8986', size: 10 },
          titlefont: { color: '#aa8986', size: 10 },
        },
      },
    }
  })()

  return (
    <div className="space-y-6 pb-20">
      <div>
        <h2 className="text-3xl font-bold text-on-surface mb-2">SHAP Explainability Matrix</h2>
        <p className="text-sm text-on-surface-variant">
          SHapley Additive exPlanations — identifying the causal weight of isolated features on output variance.
        </p>
      </div>

      <div className="glass-panel p-4 rounded-xl flex items-center gap-6">
        <div>
          <label className="text-[10px] font-bold text-primary tracking-widest uppercase mb-2 block">Background Nodes</label>
          <input
            type="number" value={nBg} min={10} max={200} step={10}
            onChange={e => setNBg(parseInt(e.target.value))}
            className="w-32 bg-surface-container-lowest rounded-lg px-4 py-2 text-sm text-on-surface font-mono border border-outline-variant focus:border-primary outline-none transition-colors"
          />
        </div>
        <button
          onClick={run} disabled={loading}
          className="mt-6 px-6 py-2.5 bg-[rgba(186,18,36,0.1)] hover:bg-[rgba(186,18,36,0.2)] text-primary font-bold text-sm border border-primary transition-all flex items-center gap-2 crimson-glow disabled:opacity-40 disabled:cursor-not-allowed rounded-lg"
        >
          <span className="material-symbols-outlined text-[20px]">{loading ? 'sync' : 'psychology'}</span>
          {loading ? 'COMPUTING SHAP...' : 'COMPUTE SHAP TENSORS'}
        </button>
        {error && <span className="text-primary text-xs font-mono border border-primary px-3 py-1 bg-primary/10 rounded mt-6 ml-auto">{error}</span>}
      </div>

      {result && (
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
          {/* Global importance bar */}
          <div className="glass-panel rounded-xl p-6">
            <h3 className="text-xs font-bold text-primary tracking-widest uppercase mb-4 flex items-center gap-2">
                 <span className="material-symbols-outlined text-[16px]">bar_chart</span>
                 Global Impact Matrix (Mean |SHAP|)
             </h3>
            <div className="bg-surface-container-lowest border border-outline-variant rounded-lg">
                <Plot
                  data={[{
                    x:           ranked.map(r => r.importance),
                    y:           ranked.map(r => r.feature),
                    type:        'bar',
                    orientation: 'h',
                    marker: {
                      color: ranked.map(r =>
                        CATEGORY_COLORS[FEATURE_CATEGORY[r.feature] || 'Unknown']
                      ),
                    },
                  }]}
                  layout={{
                    ...PLOT_LAYOUT,
                    height: Math.max(320, ranked.length * 30),
                    xaxis:  { ...GRID },
                    yaxis:  { automargin: true },
                    margin: { l: 200, r: 20, t: 20, b: 40 },
                  }}
                  config={PLOTLY_CONFIG} style={{ width: '100%' }}
                />
            </div>
            {/* Category legend */}
            <div className="flex gap-6 mt-4 flex-wrap bg-surface-container-highest border border-outline-variant rounded-lg p-3">
              {['Structural', 'Textual', 'Evolutionary'].map(cat => (
                <div key={cat} className="flex items-center gap-2 text-xs font-bold uppercase tracking-widest text-on-surface-variant">
                  <span className="w-3 h-3 rounded-md"
                        style={{ background: CATEGORY_COLORS[cat], display: 'inline-block' }} />
                  {cat}
                </div>
              ))}
            </div>
          </div>

          {/* Beeswarm */}
          {beeswarmTrace && (
            <div className="glass-panel rounded-xl p-6">
               <h3 className="text-xs font-bold text-primary tracking-widest uppercase mb-4 flex items-center gap-2">
                 <span className="material-symbols-outlined text-[16px]">scatter_plot</span>
                 Feature Distribution & Polarity
               </h3>
               <div className="bg-surface-container-lowest border border-outline-variant rounded-lg">
                  <Plot
                    data={[beeswarmTrace]}
                    layout={{
                      ...PLOT_LAYOUT,
                      height: Math.max(380, names.length * 30),
                      xaxis:  { title: 'SHAP value (Impact on Prediction)', ...GRID,
                                 zeroline: true, zerolinecolor: 'rgba(81,67,65,0.6)' },
                      yaxis:  { automargin: true, type: 'category', categoryorder: 'array',
                                 categoryarray: [...names].reverse() },
                      margin: { l: 200, r: 10, t: 20, b: 50 },
                    }}
                    config={PLOTLY_CONFIG} style={{ width: '100%' }}
                  />
               </div>
            </div>
          )}

          {/* Ranked table */}
          <div className="xl:col-span-2 glass-panel rounded-xl flex flex-col">
            <div className="p-4 border-b border-outline-variant bg-surface-container-lowest">
               <h3 className="text-xs font-bold text-primary tracking-widest uppercase flex items-center gap-2">
                   <span className="material-symbols-outlined text-[16px]">format_list_numbered</span>
                   Analyzed Hierarchy
               </h3>
            </div>
            
            <div className="flex-1 overflow-x-auto max-h-[400px] custom-scrollbar bg-surface-container-lowest rounded-b-xl">
              <table className="text-xs w-full text-left border-collapse">
                <thead className="sticky top-0 bg-surface-container-high z-10 border-b border-outline-variant shadow-sm text-on-surface-variant">
                  <tr>
                    <th className="px-6 py-3 font-semibold w-16">RANK</th>
                    <th className="px-4 py-3 font-semibold">FEATURE TENSOR</th>
                    <th className="px-4 py-3 font-semibold">VECTOR CATEGORY</th>
                    <th className="px-6 py-3 font-semibold text-right">GLOBAL WEIGHT (MEAN |SHAP|)</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-outline-variant/30">
                  {ranked.map((r, i) => {
                    const cat = FEATURE_CATEGORY[r.feature] || 'Unknown'
                    return (
                      <tr key={r.feature} className="hover:bg-surface-container transition-colors">
                        <td className="px-6 py-3 text-on-surface-variant font-mono font-bold text-lg">{i + 1}</td>
                        <td className="px-4 py-3 font-mono text-on-surface font-bold">{r.feature}</td>
                        <td className="px-4 py-3">
                          <span className="text-[10px] font-bold uppercase tracking-widest px-2 py-1 rounded"
                                style={{ color: CATEGORY_COLORS[cat], backgroundColor: `${CATEGORY_COLORS[cat]}15` }}>{cat}</span>
                        </td>
                        <td className="px-6 py-3 text-right font-mono text-primary font-bold text-sm">
                          {r.importance.toFixed(6)}
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}