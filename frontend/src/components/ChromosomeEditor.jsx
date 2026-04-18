/**
 * ChromosomeEditor.jsx
 *
 * Interactive chromosome editor: toggle individual features on/off and
 * instantly re-run prediction to see how risk scores change.
 */

import { useState, useEffect, useCallback } from 'react'
import Plot from 'react-plotly.js'
import { api } from '../utils/api.js'
import { PLOT_LAYOUT, GRID, COLORS, PLOTLY_CONFIG } from '../utils/plotTheme.js'

const RISK_COLORS = { High: COLORS.red, Medium: COLORS.yellow, Low: COLORS.teal }
const RISK_ORDER  = { High: 2, Medium: 1, Low: 0 }

const FEATURE_CATEGORY = {
  avg_cyclomatic_complexity: 'A', halstead_volume: 'A', halstead_effort: 'A',
  depth_of_inheritance_tree: 'A', number_of_methods_per_class: 'A',
  weighted_methods_per_class: 'A', nesting_depth: 'A', class_coupling: 'A',
  comment_density: 'B', whitespace_ratio: 'B', docstring_presence: 'B',
  avg_identifier_length: 'B', code_duplication_pct: 'B',
  commit_frequency: 'C', author_count: 'C', bug_fix_ratio: 'C',
  code_age_days: 'C', code_churn: 'C', added_deleted_ratio: 'C',
}
const CAT_COLORS = { A: COLORS.accent, B: COLORS.blue, C: COLORS.teal }
const CAT_LABELS = { A: 'Structural', B: 'Textual', C: 'Evolutionary' }

function riskLevel(score) {
  if (score >= 2.0) return 'High'
  if (score >= 0.5) return 'Medium'
  return 'Low'
}

function BitToggle({ index, value, name, changed, disabled, onToggle }) {
  const cat   = FEATURE_CATEGORY[name] || 'X'
  const color = value ? (CAT_COLORS[cat] || COLORS.muted) : 'transparent'
  const ring  = changed ? `2px solid ${COLORS.yellow}` : '2px solid transparent'

  return (
    <button
      onClick={() => !disabled && onToggle(index)}
      disabled={disabled}
      title={`${name} (Category ${cat}: ${CAT_LABELS[cat] || 'Unknown'})\nClick to toggle`}
      className={`flex flex-col items-center gap-1 p-1.5 rounded transition-all disabled:opacity-40 hover:opacity-80 active:scale-95 ${changed ? 'bg-surface-container shadow-[0_0_10px_rgba(255,224,130,0.2)]' : 'bg-surface-container-lowest border border-outline-variant'}`}
      style={{ outline: ring, outlineOffset: '1px' }}
    >
      <span
        className="w-6 h-6 rounded flex items-center justify-center text-[10px] font-bold border transition-colors shadow-sm"
        style={{ 
            background: color, 
            color: value ? 'white' : COLORS.muted,
            borderColor: value ? 'transparent' : 'rgba(81,67,65,0.4)'
        }}
      >
        {value ? cat : '·'}
      </span>
      <span className="text-on-surface-variant font-mono" style={{ fontSize: 9, writingMode: 'vertical-rl', transform: 'rotate(180deg)', lineHeight: 1, letterSpacing: '1px' }}>
        {name.replace(/_/g, ' ').toUpperCase().slice(0, 14)}
      </span>
    </button>
  )
}

export default function ChromosomeEditor({ cfg }) {
  const [ga,           setGa]          = useState(null)
  const [allFeatures,  setAllFeatures]  = useState([])   
  const [chromosome,   setChromosome]   = useState([])   
  const [gaChromosome, setGaChromosome] = useState([])   
  const [presets,      setPresets]      = useState([])   
  const [presetName,   setPresetName]   = useState('')

  const [baseResult,   setBaseResult]   = useState(null)  
  const [editResult,   setEditResult]   = useState(null)  
  const [loading,      setLoading]      = useState(false)
  const [error,        setError]        = useState(null)

  useEffect(() => {
    Promise.all([
      api.results('ga_results'),
      api.featureNames(cfg.processed_file),
    ]).then(([gaRes, featRes]) => {
      const data = gaRes.data
      const features = featRes.data.features
      setGa(data)
      setAllFeatures(features)
      // Truncate chromosome to match actual feature count — the GA may have
      // been run on a different version of the dataset with more columns.
      const truncated = data.chromosome.slice(0, features.length)
      // Pad with zeros if chromosome is shorter than feature count
      while (truncated.length < features.length) truncated.push(0)
      setGaChromosome([...truncated])
      setChromosome([...truncated])
    }).catch(() => {})
  }, [cfg.processed_file])

  const n_active   = chromosome.filter(Boolean).length
  const n_total    = chromosome.length
  const changed    = chromosome.map((v, i) => v !== gaChromosome[i])
  const n_changed  = changed.filter(Boolean).length

  const predict = useCallback(async (chrom, setter) => {
    setLoading(true); setError(null)
    try {
      const res = await api.predict({ processed_file: cfg.processed_file, chromosome: chrom })
      setter(res.data)
    } catch (e) {
      setError(e.response?.data?.detail || e.message)
    } finally {
      setLoading(false)
    }
  }, [cfg.processed_file])

  const runComparison = async () => {
    setBaseResult(null); setEditResult(null)
    await predict(gaChromosome, setBaseResult)
    await predict(chromosome,   setEditResult)
  }

  const toggleBit = (i) => {
    const next = [...chromosome]
    if (next[i] === 1 && next.filter(Boolean).length <= 1) return
    next[i] = next[i] ? 0 : 1
    setChromosome(next)
    setEditResult(null)  
  }

  const resetToGA = () => { setChromosome([...gaChromosome]); setEditResult(null) }
  const selectAll  = (cat) => setChromosome(chromosome.map((v, i) =>
    FEATURE_CATEGORY[allFeatures[i]] === cat ? 1 : v))
  const clearAll   = (cat) => {
    const next = chromosome.map((v, i) =>
      FEATURE_CATEGORY[allFeatures[i]] === cat
        ? (chromosome.filter((b, j) => j !== i && FEATURE_CATEGORY[allFeatures[j]] === cat || FEATURE_CATEGORY[allFeatures[j]] !== cat).some(Boolean) ? 0 : v)
        : v)
    if (next.some(Boolean)) setChromosome(next)
  }

  const savePreset = () => {
    if (!presetName.trim()) return
    setPresets(p => [...p, { name: presetName.trim(), chromosome: [...chromosome] }])
    setPresetName('')
  }
  const loadPreset = (chrom) => { setChromosome([...chrom]); setEditResult(null) }

  const diffRows = (() => {
    if (!baseResult || !editResult) return []
    const baseMap = Object.fromEntries(baseResult.files.map(f => [f.file, f]))
    return editResult.files.map(f => {
      const base = baseMap[f.file]
      if (!base) return null
      const oldRisk  = base.risk_level
      const newRisk  = f.risk_level
      const scoreDiff = f.predicted_score - base.predicted_score
      return { file: f.file, oldRisk, newRisk, oldScore: base.predicted_score,
               newScore: f.predicted_score, scoreDiff, changed: oldRisk !== newRisk }
    }).filter(Boolean).sort((a, b) => Math.abs(b.scoreDiff) - Math.abs(a.scoreDiff))
  })()

  const levelChanges = diffRows.filter(r => r.changed)

  if (!ga) return (
     <div className="flex flex-col items-center justify-center mt-20 text-on-surface-variant h-[60vh] glass-panel rounded-xl border border-outline-variant border-dashed">
      <span className="material-symbols-outlined text-4xl mb-3 text-outline-variant">family_history</span>
      <p className="text-sm font-medium tracking-wide">NO VECTOR SEQUENCE AVAILABLE</p>
      <p className="text-xs mt-1 font-mono">Run the GA optimizer first to load a chromosome.</p>
    </div>
  )

  return (
    <div className="space-y-6 pb-20">
      <div>
        <h2 className="text-3xl font-bold text-on-surface mb-2">Chromosome Editor & Simulator</h2>
        <p className="text-sm text-on-surface-variant">Manually ablate feature dimensions and instantly simulate deviance fluctuations.</p>
      </div>

      {/* Chromosome strip */}
      <div className="glass-panel rounded-xl p-6 relative">
        <h3 className="text-xs font-bold text-primary tracking-widest uppercase mb-4 flex items-center justify-between">
           <div className="flex items-center gap-2">
             <span className="material-symbols-outlined text-[16px]">dns</span>
             Active Vector Sequencer
           </div>
           <div className="flex bg-surface-container-high rounded border border-outline-variant overflow-hidden">
                {['A', 'B', 'C'].map(cat => (
                  <div key={cat} className="flex border-r border-outline-variant last:border-0">
                    <button onClick={() => selectAll(cat)} className="px-2 py-1 text-[10px] font-bold hover:bg-surface-container-highest transition-colors" style={{ color: CAT_COLORS[cat] }}>+{cat}</button>
                    <button onClick={() => clearAll(cat)} className="px-2 py-1 text-[10px] font-bold text-on-surface-variant hover:bg-surface-container-highest transition-colors">−{cat}</button>
                  </div>
                ))}
           </div>
        </h3>

        <div className="flex items-center justify-between mb-4 bg-surface-container-lowest border border-outline-variant rounded-lg p-3">
          <div className="flex items-center gap-6">
            <div className="flex flex-col">
                <span className="text-[10px] font-bold text-on-surface-variant tracking-wider uppercase mb-1">Dimensions</span>
                <span className="text-lg font-mono text-primary font-bold">{n_active}<span className="text-sm text-on-surface-variant">/{n_total}</span></span>
            </div>
            {n_changed > 0 && (
              <div className="flex flex-col border-l border-outline-variant pl-6">
                <span className="text-[10px] font-bold text-yellow-500/70 tracking-wider uppercase mb-1">Delta</span>
                <span className="text-sm text-yellow-400 font-mono">±{n_changed} DIMENSIONS</span>
              </div>
            )}
          </div>
          <button onClick={resetToGA} className="px-4 py-2 rounded-lg text-xs font-bold text-on-surface hover:bg-surface-container-high border border-outline-variant transition-colors flex items-center gap-2">
            <span className="material-symbols-outlined text-[16px] text-primary">restart_alt</span> REVERT TO ORIGIN
          </button>
        </div>

        <div className="bg-surface-container-lowest border border-outline-variant p-4 rounded-lg flex flex-wrap gap-2 overflow-x-auto min-h-[120px] custom-scrollbar">
          {chromosome.map((bit, i) => (
            <BitToggle key={i} index={i} value={bit}
                       name={allFeatures[i] || `f${i}`}
                       changed={changed[i]}
                       disabled={loading}
                       onToggle={toggleBit} />
          ))}
        </div>

        {/* Category legend */}
        <div className="flex gap-8 mt-4">
          {Object.entries(CAT_LABELS).map(([cat, label]) => {
            const count = chromosome.filter((v, i) => v && FEATURE_CATEGORY[allFeatures[i]] === cat).length
            return (
              <div key={cat} className="flex items-center gap-2 text-xs">
                <span className="w-5 h-5 rounded flex items-center justify-center text-white font-bold" style={{ background: CAT_COLORS[cat], fontSize: 10 }}>{cat}</span>
                <span className="text-on-surface-variant font-medium uppercase tracking-wider">{label}</span>
                <span className="font-mono font-bold" style={{ color: CAT_COLORS[cat] }}>{count}</span>
              </div>
            )
          })}
        </div>
      </div>

      {/* Presets and Actions */}
      <div className="glass-panel rounded-xl p-4 flex flex-col md:flex-row gap-4 items-center justify-between">
         <div className="flex items-center gap-3">
             <button onClick={runComparison} disabled={loading}
                     className="px-6 py-3 rounded-lg bg-[rgba(186,18,36,0.1)] hover:bg-[rgba(186,18,36,0.2)] text-primary font-bold text-sm border border-primary transition-all flex items-center gap-2 crimson-glow disabled:opacity-40 disabled:cursor-not-allowed">
               <span className="material-symbols-outlined text-[20px]">{loading ? 'sync' : 'compare_arrows'}</span>
               {loading ? 'SIMULATING DEVIANCE...' : 'COMPARE FLUCTUATIONS'}
             </button>
             <button onClick={() => predict(chromosome, setEditResult)} disabled={loading}
                     className="px-4 py-3 rounded-lg bg-surface-container-high hover:bg-surface-variant text-on-surface text-sm border border-outline-variant transition-colors font-medium">
               Quick Sim
             </button>
             {error && <span className="text-primary text-xs font-mono bg-primary/10 px-3 py-1.5 rounded">{error}</span>}
         </div>

         <div className="flex items-center gap-2">
            {presets.length > 0 && (
                <div className="flex items-center gap-2 mr-4 text-xs font-mono">
                  <span className="text-on-surface-variant mr-1">Presets:</span>
                  {presets.map((p, i) => (
                    <button key={i} onClick={() => loadPreset(p.chromosome)}
                            className="px-3 py-1 rounded bg-[rgba(186,18,36,0.1)] text-primary hover:bg-[rgba(186,18,36,0.2)] border border-[rgba(186,18,36,0.3)] transition-colors">
                      {p.name}
                    </button>
                  ))}
                </div>
            )}
            <div className="flex bg-surface-container-high rounded-lg border border-outline-variant overflow-hidden">
                <input value={presetName} onChange={e => setPresetName(e.target.value)}
                     placeholder="ID Sequence..."
                     className="bg-transparent px-3 py-2 text-xs text-on-surface font-mono outline-none w-32 focus:w-40 transition-all border-r border-outline-variant" />
                <button onClick={savePreset} disabled={!presetName.trim()}
                        className="px-4 py-2 text-xs font-bold text-on-surface bg-surface-container hover:bg-surface-container-lowest disabled:opacity-40 transition-colors">
                  SAVE
                </button>
            </div>
         </div>
      </div>

      {/* Results */}
      {(baseResult || editResult) && (
        <div className="space-y-6">
          {/* Side-by-side scatter */}
          {baseResult && editResult && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {[
                { label: 'Origin (GA Vector)', result: baseResult, color: COLORS.muted },
                { label: 'Edited Vector',      result: editResult, color: COLORS.accent },
              ].map(({ label, result, color }) => {
                const maxV = Math.max(...result.files.map(f => Math.max(f.true_bugs, f.predicted_score))) + 1
                return (
                  <div key={label} className="glass-panel rounded-xl p-6">
                    <div className="flex justify-between items-start mb-4">
                        <div>
                            <h3 className="text-[10px] font-bold text-primary tracking-widest uppercase mb-1">{label}</h3>
                            <div className="text-2xl font-mono" style={{ color: color }}>{result.mse?.toFixed(4)} <span className="text-[10px] text-on-surface-variant tracking-widest uppercase ml-1 block mt-1">Deviance (MSE)</span></div>
                        </div>
                    </div>
                    
                    <div className="bg-surface-container-lowest border border-outline-variant rounded-lg p-2">
                        <Plot
                          data={['High', 'Medium', 'Low'].map(level => ({
                            x:    result.files.filter(f => f.risk_level === level).map(f => f.true_bugs),
                            y:    result.files.filter(f => f.risk_level === level).map(f => f.predicted_score),
                            text: result.files.filter(f => f.risk_level === level).map(f => f.file.split('/').pop()),
                            name: level, type: 'scatter', mode: 'markers',
                            marker: { color: RISK_COLORS[level], size: 6, opacity: 0.8 },
                          }))}
                          layout={{
                            ...PLOT_LAYOUT, height: 260,
                            xaxis: { title: 'Ground Truth', ...GRID },
                            yaxis: { title: 'Predicted Vol', ...GRID },
                            showlegend: false,
                            margin: { l: 40, r: 10, t: 10, b: 40 },
                          }}
                          config={PLOTLY_CONFIG} style={{ width: '100%' }}
                        />
                    </div>
                  </div>
                )
              })}
            </div>
          )}

          {/* Diff table */}
          {baseResult && editResult && (
            <div className="glass-panel rounded-xl flex flex-col mt-6">
              <div className="p-4 border-b border-outline-variant flex items-center justify-between bg-surface-container-lowest">
                <h3 className="text-xs font-bold text-primary tracking-widest uppercase flex items-center gap-2">
                    <span className="material-symbols-outlined text-[16px]">difference</span> 
                    Vector Delta Log
                </h3>
                {levelChanges.length > 0 ? (
                  <span className="text-[10px] bg-yellow-500/20 text-yellow-400 border border-yellow-500/30 px-3 py-1 rounded-full font-bold uppercase tracking-widest">
                    {levelChanges.length} Node{levelChanges.length > 1 ? 's' : ''} Shifted
                  </span>
                ) : (
                  <span className="text-[10px] bg-green-500/20 text-green-400 border border-green-500/30 px-3 py-1 rounded-full font-bold uppercase tracking-widest">System Stable</span>
                )}
              </div>

              {levelChanges.length > 0 && (
                <div className="p-4 bg-surface-container-highest border-b border-outline-variant flex gap-2 flex-wrap text-[10px] uppercase font-bold tracking-widest">
                  {levelChanges.slice(0, 8).map(r => (
                    <div key={r.file} className="bg-surface-container-lowest border border-outline-variant rounded px-3 py-1.5 flex items-center gap-2">
                      <span className="font-mono text-on-surface max-w-[100px] truncate">{r.file.split('/').pop()}</span>
                      <span style={{ color: RISK_COLORS[r.oldRisk] }}>{r.oldRisk[0]}</span>
                      <span className="material-symbols-outlined text-[12px] text-on-surface-variant">arrow_forward</span>
                      <span style={{ color: RISK_COLORS[r.newRisk] }}>{r.newRisk[0]}</span>
                    </div>
                  ))}
                  {levelChanges.length > 8 && (
                    <span className="text-on-surface-variant self-center px-2 py-1">+{levelChanges.length - 8} MORE</span>
                  )}
                </div>
              )}

              <div className="flex-1 overflow-x-auto max-h-96 custom-scrollbar bg-surface-container-lowest rounded-b-xl">
                <table className="text-xs w-full text-left border-collapse">
                  <thead className="sticky top-0 bg-surface-container-high z-10 border-b border-outline-variant shadow-sm">
                    <tr>
                      <th className="px-4 py-3 font-semibold text-on-surface-variant">NODE IDENTIFIER</th>
                      <th className="px-4 py-3 font-semibold text-on-surface-variant text-right">ORIGIN VOL</th>
                      <th className="px-4 py-3 font-semibold text-on-surface-variant text-right">EDITED VOL</th>
                      <th className="px-4 py-3 font-semibold text-on-surface-variant text-right">DELTA (Δ)</th>
                      <th className="px-4 py-3 font-semibold text-on-surface-variant text-right">ORIGIN TIER</th>
                      <th className="px-4 py-3 font-semibold text-on-surface-variant text-right">EDITED TIER</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-outline-variant/30">
                    {diffRows.map((r, i) => (
                      <tr key={r.file} className={`hover:bg-surface-container transition-colors ${r.changed ? 'bg-[rgba(255,224,130,0.05)]' : ''}`}>
                        <td className="px-4 py-2 font-mono text-on-surface max-w-[200px] truncate" title={r.file}>
                          {r.file.split('/').pop()}
                        </td>
                        <td className="px-4 py-2 text-right font-mono text-on-surface-variant">{r.oldScore.toFixed(3)}</td>
                        <td className="px-4 py-2 text-right font-mono text-primary font-bold">{r.newScore.toFixed(3)}</td>
                        <td className={`px-4 py-2 text-right font-mono font-bold ${
                          r.scoreDiff > 0 ? 'text-primary' : r.scoreDiff < 0 ? 'text-green-400' : 'text-on-surface-variant'
                        }`}>
                          {r.scoreDiff >= 0 ? '+' : ''}{r.scoreDiff.toFixed(3)}
                        </td>
                        <td className="px-4 py-2 text-right font-bold uppercase tracking-wider" style={{ color: RISK_COLORS[r.oldRisk] }}>{r.oldRisk}</td>
                        <td className="px-4 py-2 text-right font-bold uppercase tracking-wider" style={{ color: RISK_COLORS[r.newRisk] }}>{r.newRisk}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}