import { lazy, Suspense, useState, useEffect, useRef, useCallback } from 'react'
import { api, connectLogStream } from '../utils/api.js'

const LiveGAProgress = lazy(() => import('./LiveGAProgress.jsx'))

const STAGE_KEYS = [
  ['raw_data',    'Mine repository'],
  ['clean_data',  'Clean & preprocess'],
  ['hyperparams', 'Hyperparameter tuning'],
  ['ga_results',  'GA optimisation'],
  ['baselines',   'Baseline comparison'],
  ['ablation',    'Ablation study'],
  ['stats',       'Significance tests'],
  ['multi_repo',  'Multi-repo comparison'],
  ['sensitivity', 'Sensitivity sweep'],
  ['report',      'HTML report'],
]

function StageList({ fileStatus }) {
  return (
    <div className="space-y-4">
      {STAGE_KEYS.map(([key, label], idx) => {
        const done = fileStatus[key]
        return (
          <div key={key} className="flex items-center gap-4">
            <div className={`w-8 h-8 rounded-full flex items-center justify-center border text-sm font-medium ${done ? 'border-primary bg-[rgba(186,18,36,0.1)] text-primary crimson-glow' : 'border-outline-variant bg-surface-container text-on-surface-variant'}`}>
              {done ? <span className="material-symbols-outlined text-[18px]">check</span> : idx + 1}
            </div>
            <span className={done ? 'text-on-surface font-medium' : 'text-on-surface-variant'}>{label}</span>
          </div>
        )
      })}
    </div>
  )
}

function LogConsole({ lines, status }) {
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [lines.length])

  const statusColor = { done: 'text-green-400', failed: 'text-primary', running: 'text-secondary', queued: 'text-on-surface-variant' }

  return (
    <div className="noir-glass rounded-xl overflow-hidden flex flex-col" style={{ height: 420 }}>
      {/* Console Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-outline-variant bg-surface-container-lowest">
        <div className="flex items-center gap-2">
          <span className="material-symbols-outlined text-on-surface-variant text-[18px]">terminal</span>
          <span className="text-xs font-mono text-on-surface-variant uppercase tracking-widest">Execution Shell</span>
        </div>
        {status && (
          <div className="flex items-center">
             <span className={`inline-block w-2 h-2 rounded-full mr-2 bg-current ${statusColor[status] || 'text-on-surface-variant'} ${status === 'running' ? 'pulse-glow' : ''}`}></span>
             <span className={`text-[10px] font-bold uppercase tracking-widest ${statusColor[status] || 'text-on-surface-variant'}`}>
               {status}
             </span>
          </div>
        )}
      </div>
      {/* Console Body */}
      <div className="flex-1 overflow-y-auto p-4 log-console text-xs space-y-1 bg-surface-container-lowest font-mono">
        {lines.length === 0 ? (
          <p className="text-on-surface-variant italic">System ready. Awaiting command execution...</p>
        ) : (
          lines.map((line, i) => {
            const isErr  = line.includes('[ERROR]') || line.includes('Error') || line.includes('Traceback')
            const isGood = line.includes('✓') || line.includes('Done') || line.includes('Saved')
            const isGen  = line.match(/Gen\s+\d+/)
            return (
              <div key={i} className={
                isErr  ? 'text-primary' :
                isGood ? 'text-green-400' :
                isGen  ? 'text-secondary' :
                'text-on-surface'
              }>
                {line || '\u00a0'}
              </div>
            )
          })
        )}
        <div ref={bottomRef} />
      </div>
    </div>
  )
}

export default function PipelineTab({ cfg, fileStatus, onRefresh, onJobStart, onJobDone }) {
  const [jobId,    setJobId]    = useState(null)
  const [jobStatus,setJobStatus]= useState(null)
  const [logLines, setLogLines] = useState([])
  const [summary,  setSummary]  = useState(null)
  const wsRef = useRef(null)

  // Load dataset summary when clean file exists
  useEffect(() => {
    if (fileStatus.clean_data) {
      api.dataSummary(cfg.processed_file)
        .then(r => setSummary(r.data))
        .catch(() => {})
    }
  }, [fileStatus.clean_data, cfg.processed_file])

  const startJob = useCallback(async (overrides = {}) => {
    // Close any existing WS
    wsRef.current?.close()
    setLogLines([])
    setJobStatus('queued')

    try {
      const payload = { ...cfg, ...overrides }
      const { data } = await api.runPipeline(payload)
      setJobId(data.job_id)
      onJobStart?.()

      // Connect WebSocket for live logs
      const ws = connectLogStream(data.job_id, {
        onLog:     (line)  => setLogLines(l => [...l, line]),
        onHistory: (lines) => setLogLines(lines),
        onStatus:  (s)     => { setJobStatus(s); onRefresh(); if (s !== 'running') onJobDone?.() },
        onError:   (msg)   => setLogLines(l => [...l, `[WS ERROR] ${msg}`]),
      })
      wsRef.current = ws
    } catch (err) {
      setLogLines([`Failed to start pipeline: ${err.message}`])
      setJobStatus('failed')
    }
  }, [cfg, onRefresh])

  const cancelJob = useCallback(async () => {
    if (jobId) {
      await api.cancelJob(jobId).catch(() => {})
      wsRef.current?.close()
      setJobStatus('cancelled')
      onJobDone?.()
    }
  }, [jobId, onJobDone])

  const isRunning = jobStatus === 'running' || jobStatus === 'queued'

  return (
    <div className="space-y-6 pb-20">
      {/* Header Section */}
      <div>
        <h1 className="text-3xl font-bold text-on-surface mb-2">Execution Pipeline</h1>
        <p className="text-sm text-on-surface-variant">
          Initiate deep optimization processes. The terminal provides raw, real-time diagnostic output.
        </p>
      </div>

      <div className="grid grid-cols-12 gap-6">
        {/* Stages View */}
        <div className="col-span-4 glass-panel rounded-xl flex flex-col h-[420px]">
          <div className="p-4 border-b border-outline-variant flex items-center justify-between">
            <h3 className="text-xs font-bold text-primary tracking-widest uppercase">Pipeline Matrix</h3>
          </div>
          <div className="p-6 flex-1 overflow-y-auto">
            <StageList fileStatus={fileStatus} />
          </div>
        </div>

        {/* Console View */}
        <div className="col-span-8">
            <LogConsole lines={logLines} status={jobStatus} />
        </div>
      </div>

      {/* Action Bar */}
      <div className="glass-panel p-4 rounded-xl flex items-center gap-4">
          {!isRunning ? (
            <>
              <button 
                onClick={() => startJob({ run_all: true })}
                className="flex-1 py-3 px-6 rounded-lg bg-[rgba(186,18,36,0.1)] hover:bg-[rgba(186,18,36,0.2)] text-primary font-bold text-sm border border-[rgba(186,18,36,0.3)] transition-all flex justify-center items-center gap-2 crimson-glow"
              >
                <span className="material-symbols-outlined text-[20px]">play_arrow</span>
                START FULL SEQUENCE
              </button>
              
              <button 
                onClick={() => startJob({})}
                className="py-3 px-6 rounded-lg bg-surface-container-high hover:bg-surface-variant text-on-surface font-medium text-sm border border-outline-variant transition-all flex items-center gap-2"
              >
                <span className="material-symbols-outlined text-[18px]">quick_reference_all</span>
                GA Core Only
              </button>

              <button 
                onClick={() => startJob({ run_baselines: cfg.run_baselines, run_ablation: cfg.run_ablation, run_stats: cfg.run_stats, run_multi: cfg.run_multi, run_sensitivity: cfg.run_sensitivity, run_report: cfg.run_report })}
                className="py-3 px-6 rounded-lg bg-surface-container-high hover:bg-surface-variant text-on-surface font-medium text-sm border border-outline-variant transition-all flex items-center gap-2"
              >
                <span className="material-symbols-outlined text-[18px]">science</span>
                Research Modules
              </button>
            </>
          ) : (
            <button 
                onClick={cancelJob}
                className="flex-1 py-3 px-6 rounded-lg bg-surface-container hover:bg-surface-container-highest text-primary font-bold text-sm border border-primary transition-all flex justify-center items-center gap-2"
              >
                <span className="material-symbols-outlined text-[20px]">stop</span>
                ABORT SEQUENCE
            </button>
          )}

          {fileStatus.report && (
            <button
              onClick={() => api.downloadReport()}
              className="py-3 px-6 rounded-lg bg-surface-container-high hover:bg-surface-variant text-secondary font-medium text-sm border border-outline-variant transition-all flex items-center gap-2"
            >
              <span className="material-symbols-outlined text-[18px]">download</span>
              Export Analytics
            </button>
          )}
      </div>

      {/* Dataset Summary (If available) */}
      <div className="grid grid-cols-12 gap-6">
        {summary && (
          <div className="col-span-4 glass-panel rounded-xl p-6">
             <h3 className="text-xs font-bold text-primary tracking-widest uppercase mb-4">Dataset Telemetry</h3>
             <div className="space-y-4">
               {[
                 ['Total Records Tested', summary.n_files, 'dataset'],
                 ['Features Analyzed', summary.n_features, 'scatter_plot'],
                 ['Bug-prone Elements', `${summary.bug_prone} (${summary.bug_prone_pct}%)`, 'bug_report'],
               ].map(([label, val, icon]) => (
                  <div key={label} className="flex justify-between items-center p-3 rounded-lg bg-surface-container-lowest border border-outline-variant">
                     <div className="flex items-center gap-3">
                       <span className="material-symbols-outlined text-on-surface-variant text-[18px]">{icon}</span>
                       <span className="text-xs text-on-surface-variant">{label}</span>
                     </div>
                     <span className="text-sm font-mono text-on-surface font-bold">{val}</span>
                  </div>
               ))}
             </div>
          </div>
        )}

        {/* Dataset preview table */}
        {summary?.preview?.length > 0 && (
          <div className={`glass-panel rounded-xl p-6 flex flex-col ${summary ? 'col-span-8' : 'col-span-12'}`}>
            <h3 className="text-xs font-bold text-primary tracking-widest uppercase mb-4">Data Matrix Preview</h3>
            <div className="overflow-x-auto rounded-lg border border-outline-variant bg-surface-container-lowest custom-scrollbar">
              <table className="text-xs w-full text-left border-collapse">
                <thead className="bg-surface-container-high border-b border-outline-variant">
                  <tr>
                    {Object.keys(summary.preview[0]).slice(0, 8).map(col => (
                      <th key={col} className="px-4 py-3 text-on-surface-variant font-semibold whitespace-nowrap">
                        {col}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-outline-variant/30">
                  {summary.preview.map((row, i) => (
                    <tr key={i} className="hover:bg-surface-container-high transition-colors">
                      {Object.values(row).slice(0, 8).map((val, j) => (
                        <td key={j} className="px-4 py-2 text-on-surface whitespace-nowrap font-mono text-xs">
                          {val === null ? '—' : typeof val === 'number' ? val.toFixed(3) : String(val)}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>

      {/* Live GA convergence chart (subscribes to /ws/ga-progress) */}
      {isRunning && (
        <Suspense
          fallback={
            <div className="glass-panel rounded-xl min-h-[240px] flex items-center justify-center border border-outline-variant">
              <p className="text-sm tracking-widest text-on-surface-variant uppercase">
                Loading live telemetry...
              </p>
            </div>
          }
        >
          <LiveGAProgress active={isRunning} />
        </Suspense>
      )}

    </div>
  )
}
