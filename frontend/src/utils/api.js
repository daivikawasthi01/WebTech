import axios from 'axios'

const API_ORIGIN = import.meta.env.VITE_API_ORIGIN || 'http://127.0.0.1:8000'
const BASE = `${API_ORIGIN}/api`

export const api = {
  status:       ()          => axios.get(`${BASE}/status`),
  results:      (module)    => axios.get(`${BASE}/results/${module}`),
  dataSummary:  (file)      => axios.get(`${BASE}/data/summary`, { params: { processed_file: file } }),
  runPipeline:  (cfg)       => axios.post(`${BASE}/pipeline/run`, cfg),
  listJobs:     ()          => axios.get(`${BASE}/jobs`),
  getJob:       (id)        => axios.get(`${BASE}/jobs/${id}`),
  cancelJob:    (id)        => axios.delete(`${BASE}/jobs/${id}`),
  predict:      (req)       => axios.post(`${BASE}/predict`, req),
  downloadReport: ()        => window.open(`${BASE}/report`, '_blank'),
  featureNames: (file)      => axios.get(`${BASE}/feature-names`, { params: { processed_file: file } }),
}

/**
 * Connect to the WebSocket log stream for a job.
 * @param {string} jobId
 * @param {{ onLog, onHistory, onStatus, onError }} callbacks
 * @returns WebSocket instance (call .close() to disconnect)
 */
export function connectLogStream(jobId, { onLog, onHistory, onStatus, onError }) {
  const wsOrigin = API_ORIGIN.replace(/^http/, 'ws')
  const ws = new WebSocket(`${wsOrigin}/ws/logs/${jobId}`)

  ws.onmessage = (evt) => {
    const msg = JSON.parse(evt.data)
    if (msg.type === 'log')     onLog?.(msg.line)
    if (msg.type === 'history') onHistory?.(msg.lines)
    if (msg.type === 'status')  onStatus?.(msg.status)
    if (msg.type === 'error')   onError?.(msg.message)
  }

  ws.onerror = () => onError?.('WebSocket connection error')

  return ws
}
