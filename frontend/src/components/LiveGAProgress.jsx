/**
 * LiveGAProgress.jsx
 *
 * Subscribes to /ws/ga-progress and renders a convergence chart that updates
 * generation-by-generation in real time — without parsing raw log text.
 */

import { useState, useEffect, useRef } from 'react'
import Plot from 'react-plotly.js'
import { PLOT_LAYOUT, GRID, COLORS, PLOTLY_CONFIG } from '../utils/plotTheme.js'

export default function LiveGAProgress({ active }) {
  const [history,  setHistory]  = useState([])
  const [status,   setStatus]   = useState('idle')   // idle | connecting | live | complete | error
  const [complete, setComplete] = useState(null)
  const wsRef      = useRef(null)
  const retryRef   = useRef(0)       
  const retryTimer = useRef(null)    
  const activeRef  = useRef(active)  
  const MAX_RETRIES = 5
  const BASE_DELAY  = 1000  

  useEffect(() => { activeRef.current = active }, [active])

  useEffect(() => {
    if (!active) {
      clearTimeout(retryTimer.current)
      wsRef.current?.close()
      wsRef.current = null
      retryRef.current = 0
      return
    }

    retryRef.current = 0
    setHistory([])
    setComplete(null)

    function connect() {
      setStatus('connecting')
      const protocol = location.protocol === 'https:' ? 'wss' : 'ws'
      const ws = new WebSocket(`${protocol}://${location.host}/ws/ga-progress`)

      ws.onopen = () => {
        retryRef.current = 0   
        setStatus('live')
      }

      ws.onerror = () => {
        setStatus('error')
      }

      ws.onmessage = (evt) => {
        const msg = JSON.parse(evt.data)

        if (msg.type === 'snapshot') {
          setHistory(msg.data.history || [])
        }
        if (msg.type === 'generation') {
          setHistory(h => {
            const exists = h.some(x => x.generation === msg.data.generation)
            return exists ? h : [...h, msg.data]
          })
        }
        if (msg.type === 'complete') {
          setComplete(msg.data)
          setStatus('complete')
          ws.close()
          retryRef.current = MAX_RETRIES  
        }
      }

      ws.onclose = () => {
        wsRef.current = null
        if (!activeRef.current || retryRef.current >= MAX_RETRIES) {
          if (retryRef.current >= MAX_RETRIES) setStatus('error')
          return
        }
        const attempt = retryRef.current
        retryRef.current += 1
        const delay = BASE_DELAY * Math.pow(2, attempt)
        setStatus('connecting')
        retryTimer.current = setTimeout(connect, delay)
      }

      wsRef.current = ws
    }

    connect()

    return () => {
      clearTimeout(retryTimer.current)
      wsRef.current?.close()
      wsRef.current = null
    }
  }, [active]) 

  if (!active && history.length === 0) return null

  const gens      = history.map(h => h.generation)
  const globalMse = history.map(h => h.global_best_mse ?? h.best_mse)
  const genMse    = history.map(h => h.best_mse)
  const nFeats    = history.map(h => h.n_features)
  const mutRate   = history.map(h => h.mutation_rate)

  const statusColors = {
    idle:       'text-on-surface-variant',
    connecting: 'text-yellow-400 animate-pulse',
    live:       'text-green-400 animate-pulse',
    complete:   'text-primary',
    error:      'text-primary font-bold',
  }
  const statusLabels = {
    idle:       'IDLE',
    connecting: retryRef.current > 0
                  ? `RECONNECTING... (${retryRef.current}/${MAX_RETRIES})`
                  : 'ESTABLISHING UPLINK...',
    live:       '● LIVE TELEMETRY',
    complete:   '✓ SEQUENCE COMPLETE',
    error:      retryRef.current >= MAX_RETRIES
                  ? `LINK FAILED (${MAX_RETRIES} RETRIES)`
                  : 'LINK ERROR',
  }

  return (
    <div className="mt-8 glass-panel rounded-xl p-6">
      <div className="flex items-center justify-between mb-6 bg-surface-container-highest border border-outline-variant p-3 rounded-lg">
        <h3 className="text-[10px] font-bold text-primary tracking-widest uppercase flex items-center gap-2">
            <span className="material-symbols-outlined text-[16px]">show_chart</span>
            Live Evolutionary Matrix
        </h3>
        <div className={`text-[10px] font-bold tracking-widest uppercase flex items-center gap-2 ${statusColors[status]} bg-surface-container-lowest px-3 py-1.5 rounded-full border border-outline-variant`}>
          {statusLabels[status]}
          {history.length > 0 && <span className="font-mono text-on-surface ml-1 px-1.5 bg-black/40 rounded">GEN {history[history.length - 1].generation}</span>}
        </div>
      </div>

      {history.length === 0 ? (
        <div className="h-[300px] flex flex-col items-center justify-center border border-dashed border-outline-variant rounded-lg">
          <span className="material-symbols-outlined text-4xl text-outline-variant mb-2 animate-pulse">satellite_alt</span>
          <p className="text-on-surface-variant text-[10px] uppercase tracking-widest font-bold">
            {status === 'connecting' ? 'Awaiting first telemetry packet...' : 'Awaiting data link...'}
          </p>
        </div>
      ) : (
        <div className="space-y-4">
          {/* Live metrics strip */}
          {history.length > 0 && (() => {
            const last = history[history.length - 1]
            return (
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {[
                  ['Global Minimum', globalMse[globalMse.length-1]?.toFixed(4)],
                  ['Current Matrix', genMse[genMse.length-1]?.toFixed(4)],
                  ['Active Features', nFeats[nFeats.length-1]],
                  ['Mutation Index', last.mutation_rate?.toFixed(3)],
                ].map(([label, val]) => (
                  <div key={label} className="bg-surface-container-lowest border border-outline-variant rounded-lg p-3">
                    <div className="text-[10px] text-on-surface-variant font-bold uppercase tracking-widest mb-1">{label}</div>
                    <div className="text-xl text-primary font-bold font-mono">{val ?? '—'}</div>
                  </div>
                ))}
              </div>
            )
          })()}

          <div className="bg-surface-container-lowest border border-outline-variant rounded-lg p-2">
              <Plot
                data={[
                  {
                    x: gens, y: globalMse, name: 'Global Min Deviance',
                    type: 'scatter', mode: 'lines+markers',
                    line: { color: COLORS.accent, width: 2.5, shape: 'spline' },
                    marker: { size: 6, line: { color: '#000', width: 1 } },
                  },
                  {
                    x: gens, y: genMse, name: 'Gen Min Deviance',
                    type: 'scatter', mode: 'lines',
                    line: { color: COLORS.light, width: 1.5, dash: 'dot', shape: 'spline' },
                  },
                  {
                    x: gens, y: mutRate, name: 'Mutation Index', yaxis: 'y2',
                    type: 'scatter', mode: 'lines',
                    line: { color: COLORS.yellow, width: 1.5, dash: 'dash' },
                  },
                ]}
                layout={{
                  ...PLOT_LAYOUT,
                  height: 260,
                  yaxis:  { title: 'MSE (Deviance)', ...GRID },
                  yaxis2: { title: 'Mutation Index', overlaying: 'y', side: 'right' },
                  xaxis:  { title: 'Generation Phase', ...GRID },
                  legend: { orientation: 'h', y: -0.25 },
                  margin: { l: 40, r: 40, t: 20, b: 60 },
                }}
                config={PLOTLY_CONFIG}
                style={{ width: '100%' }}
              />
          </div>

          <div className="bg-surface-container-lowest border border-outline-variant rounded-lg p-2">
              <Plot
                data={[{
                  x: gens, y: nFeats, type: 'bar',
                  marker: { color: COLORS.teal, opacity: 0.8 },
                  name: 'Feature Allocation',
                }]}
                layout={{
                  ...PLOT_LAYOUT,
                  height: 120,
                  yaxis:  { title: 'Features', ...GRID },
                  xaxis:  { ...GRID },
                  margin: { l: 40, r: 40, t: 10, b: 30 },
                  showlegend: false,
                }}
                config={PLOTLY_CONFIG}
                style={{ width: '100%' }}
              />
          </div>

          {complete && (
            <div className="bg-primary/10 border border-primary/30 rounded-lg p-4 flex flex-col md:flex-row gap-4 justify-between items-start md:items-center">
               <div className="flex items-center gap-3">
                   <div className="bg-primary/20 p-2 rounded-full flex items-center justify-center">
                       <span className="material-symbols-outlined text-primary text-2xl">verified</span>
                   </div>
                   <div>
                       <div className="text-primary font-bold text-sm tracking-widest uppercase mb-1">Matrix Resolution Complete</div>
                       <div className="text-[10px] font-mono text-on-surface-variant flex gap-3">
                          <span>DEVIANCE: <strong className="text-on-surface">{complete.best_mse?.toFixed(4)}</strong></span>
                          <span>DIMENSIONS: <strong className="text-on-surface">{complete.n_selected}/{complete.n_total}</strong></span>
                       </div>
                   </div>
               </div>
               
               <div className="bg-black/40 rounded p-2 max-w-sm md:max-w-md w-full overflow-x-auto custom-scrollbar flex items-center gap-2 text-[10px] font-mono text-on-surface-variant">
                  <span className="text-primary font-bold whitespace-nowrap">EXTRACTED VECTOR:</span>
                  <span className="whitespace-nowrap">{(complete.feature_names || []).join(' → ')}</span>
               </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}