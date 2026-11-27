import { useEffect, useMemo, useRef, useState } from 'react'
import {
  Activity,
  AlertTriangle,
  BarChart3,
  Brain,
  CheckCircle2,
  Circle,
  Database,
  Gauge,
  HeartPulse,
  Info,
  Layers3,
  Menu,
  Play,
  RefreshCw,
  ShieldCheck,
  Timer,
  X,
} from 'lucide-react'

const API = {
  health: '/api/health',
  pods: '/api/kubernetes/pods',
  metrics: '/api/kubernetes/metrics',
  aiStatus: '/api/ai/status',
  offlineEvaluation: '/api/ai/offline-evaluation',
  remediationDryRun: '/api/remediation/dry-run',
  telemetryStatus: '/api/telemetry/status',
  preprocessing: '/api/telemetry/preprocessing',
  start: '/api/demo/start',
  status: '/api/demo/status',
}

const navigation = [
  ['overview', 'Overview', Activity],
  ['telemetry', 'Live Telemetry', Gauge],
  ['detection', 'Detection', AlertTriangle],
  ['model', 'Offline LSTM', Brain],
  ['preprocessing', 'Preprocessing', Database],
  ['diagnosis', 'Diagnosis', HeartPulse],
  ['healing', 'Self-Healing', ShieldCheck],
  ['demo', 'Recovery Demo', Play],
  ['evaluation', 'Fault Evaluation', BarChart3],
]

// Sends a request to the backend. Gives up after timeoutMs so a slow
// request can never hang forever.
async function request(url, options = {}, timeoutMs = 20000) {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), timeoutMs)
  try {
    const response = await fetch(url, { ...options, signal: controller.signal })
    const data = await response.json().catch(() => ({}))
    if (!response.ok) {
      throw new Error(data.detail || 'Request failed')
    }
    return data
  } catch (err) {
    if (err.name === 'AbortError') throw new Error('Request timed out')
    throw err
  } finally {
    clearTimeout(timer)
  }
}

const endpointLabels = {
  pods: 'Kubernetes pod information',
  metrics: 'Kubernetes metrics',
  demo: 'recovery demonstration status',
  ai: 'AI status',
  offlineEvaluation: 'offline LSTM evaluation',
  dryRun: 'remediation dry-run',
  telemetry: 'telemetry status',
  preprocessing: 'telemetry preprocessing',
}

function getPodName(pod) {
  return String(pod.name || pod.pod_name || '')
}

function isApplicationPod(pod) {
  return !getPodName(pod).startsWith('cloudheal-')
}

function isPodHealthy(pod) {
  return String(pod.status || '').toLowerCase() === 'running' &&
    Number(pod.total_containers || 0) > 0 &&
    Number(pod.ready || 0) === Number(pod.total_containers || 0)
}

function formatTime(value) {
  if (!value) return 'Not available'
  const date = new Date(value)
  return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleTimeString()
}

function sumCounts(counts = {}) {
  return Object.values(counts).reduce((total, count) => total + Number(count || 0), 0)
}

function formatFieldCounts(counts = {}) {
  const nonZero = Object.entries(counts).filter(([, count]) => Number(count) > 0)
  return nonZero.length
    ? nonZero.map(([field, count]) => `${field}: ${count}`).join(', ')
    : 'None reported'
}

function formatPodChanges(changes = {}) {
  const nonZero = Object.entries(changes).filter(([, count]) => Number(count) > 0)
  return nonZero.length
    ? nonZero.map(([service, count]) => `${service}: ${count}`).join(', ')
    : 'None reported'
}

function formatTimestampRange(range) {
  if (!range?.start || !range?.end) return 'Not available'
  return `${formatTime(range.start)} - ${formatTime(range.end)}`
}

function StatusPill({ status, children }) {
  const tone = status === 'healthy' || status === 'active' || status === 'verified'
    ? 'border-emerald-400/30 bg-emerald-400/10 text-emerald-300'
    : status === 'warning' || status === 'partial'
      ? 'border-amber-400/30 bg-amber-400/10 text-amber-300'
      : status === 'error'
        ? 'border-rose-400/30 bg-rose-400/10 text-rose-300'
        : 'border-slate-600 bg-slate-800 text-slate-300'

  return (
    <span className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs font-medium ${tone}`}>
      <span className="h-1.5 w-1.5 rounded-full bg-current" />
      {children}
    </span>
  )
}

function Panel({ id, title, description, icon: Icon, action, guide, children }) {
  return (
    <section id={id} className="scroll-mt-24 rounded-2xl border border-slate-800 bg-slate-900/80 p-5 shadow-xl shadow-black/10">
      <div className="mb-5 flex flex-wrap items-start justify-between gap-3">
        <div className="flex items-start gap-3">
          <div className="rounded-lg bg-cyan-400/10 p-2 text-cyan-300">
            <Icon size={19} />
          </div>
          <div>
            <h2 className="font-semibold text-slate-100">{title}</h2>
            <p className="mt-1 text-sm text-slate-400">{description}</p>
          </div>
        </div>
        {action}
      </div>
      {guide && <SectionGuide {...guide} />}
      {children}
    </section>
  )
}

function SectionGuide({ what, result }) {
  return (
    <div className="mb-5 rounded-xl border border-cyan-400/15 bg-cyan-400/5 p-4">
      <div className="flex flex-col gap-2 text-sm sm:flex-row sm:items-start sm:justify-between sm:gap-6">
        <div>
          <p className="font-semibold text-cyan-200">What this shows</p>
          <p className="mt-1 max-w-3xl text-slate-300">{what}</p>
        </div>
        <div className="shrink-0 sm:max-w-xs sm:text-right">
          <p className="font-semibold text-cyan-200">Right now</p>
          <p className="mt-1 text-emerald-200">{result}</p>
        </div>
      </div>
    </div>
  )
}

function Metric({ label, value, hint }) {
  return (
    <div className="rounded-xl border border-slate-800 bg-slate-950/60 p-4">
      <p className="text-xs uppercase tracking-wide text-slate-500">{label}</p>
      <p className="mt-2 text-2xl font-semibold text-slate-100">{value}</p>
      {hint && <p className="mt-1 text-xs text-slate-500">{hint}</p>}
    </div>
  )
}

function NotImplemented({ children }) {
  return (
    <div className="flex items-start gap-2 rounded-lg border border-amber-400/20 bg-amber-400/5 p-3 text-sm text-amber-200">
      <Info size={16} className="mt-0.5 shrink-0" />
      <span>{children}</span>
    </div>
  )
}

function App() {
  const [anomalyStatus, setAnomalyStatus] = useState(null)
  const [health, setHealth] = useState(null)
  const [pods, setPods] = useState([])
  const [metrics, setMetrics] = useState([])
  const [demo, setDemo] = useState(null)
  const [aiStatus, setAiStatus] = useState(null)
  const [diagnosisError, setDiagnosisError] = useState('')
  const [offlineEvaluation, setOfflineEvaluation] = useState(null)
  const [dryRun, setDryRun] = useState(null)
  const [telemetryStatus, setTelemetryStatus] = useState(null)
  const [preprocessing, setPreprocessing] = useState(null)
  const [preprocessingLoading, setPreprocessingLoading] = useState(true)
  const [preprocessingError, setPreprocessingError] = useState('')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const [mobileNav, setMobileNav] = useState(false)

  // Prevents one dashboard refresh from starting before the previous one ends.
  const loadingRef = useRef(false)

  async function loadDashboard() {
    if (loadingRef.current) return
    loadingRef.current = true
    try {
      await loadDashboardOnce()
    } finally {
      loadingRef.current = false
    }
  }

  async function loadDashboardOnce() {
    const results = await Promise.allSettled([
      request(API.health),
      request(API.pods),
      request(API.metrics),
      request(API.status),
      request(API.aiStatus),
      request(API.offlineEvaluation),
      request(API.remediationDryRun),
      request(API.telemetryStatus),
      request(API.preprocessing),
      request('/api/anomalies/status'),
    ])

    const [healthResult, podsResult, metricsResult, demoResult, aiResult, offlineResult, dryRunResult, telemetryResult, preprocessingResult, anomalyResult] = results

    if (healthResult.status === 'fulfilled') {
      setHealth(healthResult.value)
    }

    if (podsResult.status === 'fulfilled') {
      const podData = podsResult.value
      setPods((Array.isArray(podData) ? podData : podData.pods || []).filter(isApplicationPod))
    }

    if (metricsResult.status === 'fulfilled') {
      setMetrics((metricsResult.value.metrics || []).filter(isApplicationPod))
    }

    if (demoResult.status === 'fulfilled') setDemo(demoResult.value.demo || null)

    if (aiResult.status === 'fulfilled') {
      setAiStatus(aiResult.value)
      setDiagnosisError('')
    } else {
      setAiStatus(null)
      setDiagnosisError(aiResult.reason.message || 'Diagnosis could not be loaded.')
    }

    if (offlineResult.status === 'fulfilled') setOfflineEvaluation(offlineResult.value)
    if (dryRunResult.status === 'fulfilled') setDryRun(dryRunResult.value)
    if (telemetryResult.status === 'fulfilled') setTelemetryStatus(telemetryResult.value)

    if (preprocessingResult.status === 'fulfilled') {
      setPreprocessing(preprocessingResult.value)
      setPreprocessingError('')
    } else {
      setPreprocessingError(preprocessingResult.reason.message || 'Preprocessing summary could not be loaded.')
    }

    if (anomalyResult.status === 'fulfilled') {
      setAnomalyStatus(anomalyResult.value)
    } else {
      setAnomalyStatus(null)
    }

    setPreprocessingLoading(false)

    const failedRequests = [
      ['pods', podsResult],
      ['metrics', metricsResult],
      ['demo', demoResult],
      ['ai', aiResult],
      ['offlineEvaluation', offlineResult],
      ['dryRun', dryRunResult],
      ['telemetry', telemetryResult],
      ['preprocessing', preprocessingResult],
    ].filter(([, result]) => result.status === 'rejected')

    if (healthResult.status === 'rejected' && healthResult.reason?.message !== 'Request timed out') {
      setError(`Backend unavailable: ${healthResult.reason.message || 'health check failed'}`)
    } else if (healthResult.status === 'rejected') {
      // Slow response, not a real outage: keep the last known state and try again next refresh.
    } else if (failedRequests.length) {
      const failedLabels = failedRequests.map(([name]) => endpointLabels[name]).join(', ')
      setError(`Backend is connected, but ${failedLabels} could not be loaded.`)
    } else {
      setError('')
    }
  }

  useEffect(() => {
    loadDashboard()
    const interval = setInterval(loadDashboard, 3000)
    return () => clearInterval(interval)
  }, [])

  const totals = useMemo(() => ({
    healthy: pods.filter(isPodHealthy).length,
    pending: pods.filter((pod) => String(pod.status).toLowerCase() === 'pending').length,
    failed: pods.filter((pod) => ['failed', 'unknown'].includes(String(pod.status).toLowerCase())).length,
    cpu: metrics.reduce((sum, pod) => sum + Number(pod.cpu_millicores || 0), 0),
    memory: metrics.reduce((sum, pod) => sum + Number(pod.memory_mib || 0), 0),
  }), [pods, metrics])
  const metricNames = useMemo(() => new Set(metrics.map(getPodName)), [metrics])
  const podsWithoutMetrics = pods.filter((pod) => !metricNames.has(getPodName(pod)))
  const telemetryResult = telemetryStatus
    ? telemetryStatus.running
      ? `Running; ${telemetryStatus.records_collected ?? 0} records collected.`
      : telemetryStatus.last_error || 'Not running.'
    : 'Waiting for the telemetry status response.'

  const demoRunning = demo?.running === true && !demo?.recovery_verified && demo?.stage !== 'Failed'
  const demoComplete = demo?.recovery_verified === true

  // While a demonstration runs, check only its status once per second.
  useEffect(() => {
    if (!demoRunning) return undefined

    let cancelled = false
    const timer = setInterval(async () => {
      try {
        const status = await request(API.status)
        if (!cancelled) setDemo(status.demo || null)
      } catch {
        // One failed check is fine; the next one tries again.
      }
    }, 1000)

    return () => {
      cancelled = true
      clearInterval(timer)
    }
  }, [demoRunning])

  async function startDemo() {
    if (busy || demoRunning) return

    setBusy(true)
    setError('')

    try {
      const started = await request(API.start, { method: 'POST' })
      if (started.demo) setDemo(started.demo)
    } catch (startError) {
      setError(`Unable to start demonstration: ${startError.message}`)
    } finally {
      setBusy(false)
    }
  }

  const demoSteps = [
    ['Application healthy', 'Baseline application state before the controlled test.', true],
    ['Controlled pod failure initiated', 'CloudHeal requests deletion of one ready checkoutservice pod.', demo?.progress >= 25],
    ['Kubernetes detects missing replica', 'The original pod is confirmed unavailable.', demo?.progress >= 45],
    ['Replacement pod created', 'The Deployment controller creates a replacement pod.', demo?.progress >= 75],
    ['Replacement pod becomes ready', 'The replacement passes the readiness checks.', demo?.progress >= 90],
    ['Recovery verification completed', 'CloudHeal confirms Deployment availability and readiness.', demoComplete],
  ]

  return (
    <div className="min-h-screen bg-[#070d18] text-slate-200">
      <aside className="fixed inset-y-0 left-0 z-30 hidden w-64 border-r border-slate-800 bg-[#0a1220] p-5 lg:block">
        <Brand />
        <div className="mt-9">
          <p className="mb-3 text-xs font-semibold uppercase tracking-widest text-slate-600">CloudHeal workflow</p>
          <nav className="space-y-1">
            {navigation.map(([id, label, Icon]) => (
              <a key={id} href={`#${id}`} className="flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm text-slate-400 transition hover:bg-slate-800 hover:text-cyan-300">
                <Icon size={17} />
                {label}
              </a>
            ))}
          </nav>
        </div>
        <div className="absolute bottom-5 left-5 right-5 rounded-xl border border-slate-800 bg-slate-950/60 p-3 text-xs text-slate-500">
          Live values from the CloudHeal API
        </div>
      </aside>

      <div className="lg:pl-64">
        <header className="sticky top-0 z-20 border-b border-slate-800 bg-[#070d18]/95 px-5 py-4 backdrop-blur md:px-8">
          <div className="flex items-center justify-between gap-4">
            <div className="flex items-center gap-3">
              <button onClick={() => setMobileNav(!mobileNav)} className="rounded-lg border border-slate-700 p-2 lg:hidden" aria-label="Toggle navigation">
                <Menu size={18} />
              </button>
              <div>
                <p className="text-xs uppercase tracking-widest text-cyan-300">Cloud-native self-healing platform</p>
                <h1 className="text-xl font-bold text-white">CloudHeal Operations Console</h1>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <StatusPill status={health?.status === 'healthy' ? 'healthy' : 'error'}>
                {health?.status === 'healthy' ? 'Backend connected' : 'Backend unavailable'}
              </StatusPill>
              <button onClick={loadDashboard} className="rounded-lg border border-slate-700 p-2 text-slate-300 hover:border-cyan-400 hover:text-cyan-300" title="Refresh data">
                <RefreshCw size={17} />
              </button>
            </div>
          </div>
          {mobileNav && (
            <nav className="mt-4 grid grid-cols-2 gap-2 lg:hidden">
              {navigation.map(([id, label]) => <a key={id} onClick={() => setMobileNav(false)} href={`#${id}`} className="rounded-lg bg-slate-800 px-3 py-2 text-sm text-slate-300">{label}</a>)}
            </nav>
          )}
        </header>

        <main className="mx-auto max-w-7xl space-y-6 p-5 md:p-8">
          {error && (
            <div className="flex items-center justify-between rounded-xl border border-rose-400/30 bg-rose-400/10 p-3 text-sm text-rose-200">
              <span>{error}</span>
              <button onClick={() => setError('')} aria-label="Dismiss error"><X size={16} /></button>
            </div>
          )}

          <section id="overview" className="scroll-mt-24">
            <div className="mb-5">
              <p className="text-sm text-cyan-300">System overview</p>
              <h2 className="mt-1 text-3xl font-bold text-white">Observe. Explain. Recover.</h2>
              <p className="mt-2 max-w-3xl text-sm text-slate-400">
                CloudHeal watches a small online shopping application, explains unusual behaviour, and demonstrates safe recovery using Kubernetes.
              </p>
            </div>
            <SectionGuide
              what="CloudHeal watches the small services that make up the shopping application."
              result={health?.status === 'healthy' ? `Backend connected; ${totals.healthy} of ${pods.length} application pods are healthy.` : 'Waiting for the backend health response.'}
            />
            <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
              <Metric label="Healthy pods" value={`${totals.healthy}/${pods.length}`} hint={`${totals.healthy} of ${pods.length} application pods are running and ready.`} />
              <Metric label="CPU usage" value={`${totals.cpu.toFixed(1)}m`} hint="Total current CPU, measured in millicores." />
              <Metric label="Memory usage" value={`${totals.memory.toFixed(1)} MiB`} hint="Total current memory measured from Kubernetes." />
              <Metric label="Collector" value={telemetryStatus?.running ? 'Running' : 'Not running'} hint={telemetryResult} />
            </div>
          </section>

          <Panel id="telemetry" title="Live Telemetry Monitoring" description="CloudHeal continuously collects CPU and memory usage from the running application containers." icon={Activity}
            guide={{
              what: 'Live CPU and memory readings from the application.',
              result: telemetryResult,
            }}
          >
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
              <Metric label="Collection interval" value="10 seconds" hint="A new sample is collected every 10 seconds." />
              <Metric label="Records this session" value={telemetryStatus?.records_collected ?? 'Unavailable'} hint="Each record is one collected pod measurement." />
              <Metric label="Last collection" value={formatTime(telemetryStatus?.last_successful_collection)} hint="Successful collection only" />
              <Metric label="Saved data" value={telemetryStatus?.telemetry_file_available ? 'Available' : 'Unavailable'} hint="Raw records in data/live/telemetry.csv" />
            </div>
            <div className="mt-4 overflow-x-auto rounded-xl border border-slate-800">
              <table className="w-full min-w-[560px] text-left text-sm">
                <thead className="bg-slate-950/80 text-xs uppercase tracking-wide text-slate-500">
                  <tr><th className="px-4 py-3">Service pod</th><th className="px-4 py-3">Status</th><th className="px-4 py-3">CPU</th><th className="px-4 py-3">Memory</th></tr>
                </thead>
                <tbody className="divide-y divide-slate-800">
                  {metrics.map((pod) => (
                    <tr key={pod.name} className="text-slate-300">
                      <td className="px-4 py-3">{pod.name}</td><td className="px-4 py-3"><StatusPill status="healthy">Measured</StatusPill></td>
                      <td className="px-4 py-3">{Number(pod.cpu_millicores || 0).toFixed(2)}m</td><td className="px-4 py-3">{Number(pod.memory_mib || 0).toFixed(2)} MiB</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {!metrics.length && <p className="p-5 text-sm text-slate-500">No live metric rows are currently available.</p>}
            </div>
            {podsWithoutMetrics.length > 0 && <NotImplemented>{podsWithoutMetrics.length} application pod(s) returned by Kubernetes do not currently have a metrics row. They are not counted as measured in this table.</NotImplemented>}
            <NotImplemented>Only the latest real measurements are shown; historical charts are not available yet.</NotImplemented>
          </Panel>

          <Panel
            id="anomaly-detection"
            title="Live Anomaly Detection"
            description="Deep learning project monitoring using live CPU and memory telemetry."
            icon={Activity}
          >
            <div className="grid gap-3 sm:grid-cols-3">
              <Metric
                label="Controller"
                value={anomalyStatus?.running ? 'Running' : 'Stopped'}
                hint="Background anomaly monitoring"
              />
              <Metric
                label="Detection mode"
                value={anomalyStatus?.mode || 'Waiting'}
                hint="Current controller mode"
              />
              <Metric
                label="Automatic healing"
                value={anomalyStatus?.remediation_enabled ? 'Enabled' : 'Disabled'}
                hint="Actual Kubernetes restarts are currently disabled"
              />
            </div>

            <div className="mt-4 overflow-x-auto rounded-xl border border-slate-800">
              <table className="w-full min-w-[500px] text-left text-sm">
                <thead className="bg-slate-950/80 text-xs uppercase tracking-wide text-slate-500">
                  <tr>
                    <th className="px-4 py-3">Service</th>
                    <th className="px-4 py-3">Detection status</th>
                    <th className="px-4 py-3">CPU</th>
                    <th className="px-4 py-3">Memory</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800">
                  {Object.entries(anomalyStatus?.services || {}).map(([service, data]) => (
                    <tr key={service} className="text-slate-300">
                      <td className="px-4 py-3">{service}</td>
                      <td className="px-4 py-3">
                        <StatusPill status={data.status === 'NORMAL' ? 'healthy' : 'warning'}>
                          {data.status}
                        </StatusPill>
                      </td>
                      <td className="px-4 py-3">
                        {data.cpu_millicores == null ? '—' : `${Number(data.cpu_millicores).toFixed(2)}m`}
                      </td>
                      <td className="px-4 py-3">
                        {data.memory_mib == null ? '—' : `${Number(data.memory_mib).toFixed(2)} MiB`}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>

              {!Object.keys(anomalyStatus?.services || {}).length && (
                <p className="p-5 text-sm text-slate-500">
                  Waiting for live telemetry and baseline collection.
                </p>
              )}
            </div>

            {anomalyStatus?.last_error && (
              <p className="mt-3 text-sm text-red-400">
                Controller error: {anomalyStatus.last_error}
              </p>
            )}
          </Panel>

          <Panel id="model" title="Deep Learning Model — Offline Test" description="Shows results from a model tested with saved data." icon={Brain}
            guide={{
              what: 'A saved deep-learning test that learned patterns from older data.',
              result: offlineEvaluation ? `Offline saved results: F1 ${(offlineEvaluation.metrics.f1_score * 100).toFixed(3)}%, precision ${(offlineEvaluation.metrics.precision * 100).toFixed(3)}%, recall ${(offlineEvaluation.metrics.recall * 100).toFixed(3)}%.` : 'Saved offline results are loading.',
            }}
          >
            <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
              <StatusPill status="partial">OFFLINE EVALUATION ONLY</StatusPill>
              <span className="text-xs text-slate-500">Artifact: {offlineEvaluation?.artifact || 'model/lstm/lstm_autoencoder.pth'}</span>
            </div>
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
              <Metric label="Input features" value={offlineEvaluation?.features ?? 'Unavailable'} hint="Saved RCAEval configuration" />
              <Metric label="Sequence length" value={offlineEvaluation?.sequence_length ?? 'Unavailable'} hint="Rows per sequence" />
              <Metric label="Architecture" value={offlineEvaluation ? `${offlineEvaluation.hidden_size} → ${offlineEvaluation.latent_size}` : 'Unavailable'} hint="Hidden size → latent size" />
              <Metric label="Threshold" value={offlineEvaluation ? offlineEvaluation.threshold.toFixed(3) : 'Unavailable'} hint="Saved reconstruction threshold" />
            </div>
            <div className="mt-4 grid gap-3 sm:grid-cols-3">
              <Metric label="Test F1" value={offlineEvaluation ? `${(offlineEvaluation.metrics.f1_score * 100).toFixed(3)}%` : 'Unavailable'} hint="Window-level saved result" />
              <Metric label="Precision" value={offlineEvaluation ? `${(offlineEvaluation.metrics.precision * 100).toFixed(3)}%` : 'Unavailable'} hint="Window-level saved result" />
              <Metric label="Recall" value={offlineEvaluation ? `${(offlineEvaluation.metrics.recall * 100).toFixed(3)}%` : 'Unavailable'} hint="Window-level saved result" />
            </div>
            <div className="mt-4 rounded-xl border border-violet-400/20 bg-violet-400/5 p-4 text-sm text-violet-200">
              Offline model only. Live telemetry does not yet provide the 47 inputs this model needs.
            </div>
          </Panel>

          <Panel id="demo" title="Recovery Demo" description="Test recovery safely and watch what happens." icon={Play} action={(
            <button disabled={busy || demoRunning} onClick={startDemo} className="inline-flex items-center gap-2 rounded-lg bg-cyan-400 px-4 py-2 text-sm font-semibold text-slate-950 disabled:cursor-not-allowed disabled:opacity-40">
              <Play size={15} /> {demoRunning ? 'Demonstration running' : 'Start demonstration'}
            </button>
          )}
            guide={{
              what: 'A step-by-step check that recovery really worked.',
              result: demoComplete ? 'Recovery verified by the backend.' : demo?.error ? `Recovery failed: ${demo.error}` : 'Not run yet.',
            }}
          >
            <div className="mb-5 flex flex-wrap gap-3 text-sm">
              <StatusPill status={demoComplete ? 'verified' : demo?.stage === 'Failed' ? 'error' : demoRunning ? 'active' : 'partial'}>
                {demoComplete ? 'Recovery verified' : demo?.stage || 'Ready to start'}
              </StatusPill>
              <span className="text-slate-500">Affected service: <span className="text-slate-300">{demo?.affected_service || 'checkoutservice'}</span></span>
            </div>
            <div className="space-y-3">
              {demoSteps.map(([title, description, complete], index) => (
                <div key={title} className="flex gap-3">
                  <div className={`mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-full border ${complete ? 'border-emerald-400/40 bg-emerald-400/10 text-emerald-300' : 'border-slate-700 text-slate-600'}`}>
                    {complete ? <CheckCircle2 size={16} /> : <Circle size={16} />}
                  </div>
                  <div className="min-w-0 flex-1 border-b border-slate-800 pb-3">
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <p className={`text-sm font-medium ${complete ? 'text-slate-200' : 'text-slate-500'}`}>{index + 1}. {title}</p>
                      {complete && <span className="text-xs text-emerald-300">Observed/verified</span>}
                    </div>
                    <p className="mt-1 text-xs text-slate-500">{description}</p>
                  </div>
                </div>
              ))}
            </div>
            <div className="mt-5 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
              <Metric label="Original pod" value={demo?.old_pod || 'Not available'} />
              <Metric label="Replacement pod" value={demo?.replacement_pod || 'Not available'} />
              <Metric label="Backend stage" value={demo?.stage || 'Idle'} />
              <Metric label="Verification" value={demoComplete ? 'Passed' : demo?.error ? 'Failed' : 'Pending'} />
            </div>
            {demo?.message && <p className="mt-4 rounded-lg bg-slate-950/60 p-3 text-sm text-slate-400">{demo.message}</p>}
          </Panel>

          <Panel id="preprocessing" title="Data Preparation" description="Shows what happens to collected readings before analysis." icon={Database}
            guide={{
              what: 'The collected readings before they are prepared for analysis.',
              result: preprocessingLoading ? 'Loading the latest quality summary.' : preprocessingError ? 'Preprocessing summary unavailable.' : `${preprocessing?.valid_records ?? 0} valid records reported.`,
            }}
          >
            {preprocessingLoading && (
              <div className="rounded-lg border border-slate-700 bg-slate-950/60 p-4 text-sm text-slate-400">
                Loading preprocessing quality summary...
              </div>
            )}
            {!preprocessingLoading && preprocessingError && (
              <div className="rounded-lg border border-rose-400/30 bg-rose-400/10 p-4 text-sm text-rose-200">
                Preprocessing summary unavailable: {preprocessingError}
              </div>
            )}
            {!preprocessingLoading && !preprocessingError && preprocessing && (
              <>
                <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
                  <Metric label="Total raw records" value={preprocessing.total_rows} hint="Rows read from telemetry.csv" />
                  <Metric label="Valid records" value={preprocessing.valid_records} hint="Rows accepted for reporting" />
                  <Metric label="Rejected records" value={preprocessing.rejected_records} hint="Rows rejected by validation" />
                  <Metric label="Duplicate records" value={preprocessing.exact_duplicate_records} hint="Exact duplicate rows" />
                </div>
                <div className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
                  <Metric label="Missing values" value={sumCounts(preprocessing.missing_field_counts)} hint={formatFieldCounts(preprocessing.missing_field_counts)} />
                  <Metric label="Invalid values" value={sumCounts(preprocessing.invalid_field_counts)} hint={formatFieldCounts(preprocessing.invalid_field_counts)} />
                  <Metric label="Service/pod groups" value={preprocessing.group_count} hint={`${preprocessing.service_count} services`} />
                  <Metric label="Pod identity changes" value={sumCounts(preprocessing.pod_changes_by_service)} hint={formatPodChanges(preprocessing.pod_changes_by_service)} />
                </div>
                <div className="mt-4 grid gap-3 md:grid-cols-2">
                  <div className="rounded-xl border border-slate-800 bg-slate-950/60 p-4 text-sm">
                    <p className="font-medium text-slate-200">Timestamp range</p>
                    <p className="mt-2 text-slate-400">{formatTimestampRange(preprocessing.timestamp_range)}</p>
                  </div>
                  <div className="rounded-xl border border-slate-800 bg-slate-950/60 p-4 text-sm">
                    <p className="font-medium text-slate-200">Last preprocessing update</p>
                    <p className="mt-2 text-slate-400">{formatTime(preprocessing.last_preprocessing_update)}</p>
                  </div>
                </div>
                <NotImplemented>Validation is connected to the dashboard. CPU and memory remain in their original millicore and MiB units; live 47-feature preparation and LSTM inference are not enabled.</NotImplemented>
              </>
            )}
          </Panel>

          <Panel id="evaluation" title="Testing and Results" description="Shows the tests and results available in this project." icon={BarChart3}
            guide={{
              what: 'A safe test of detection and recovery using a controlled failure.',
              result: demoComplete ? 'The controlled recovery demonstration is verified.' : 'The controlled recovery demonstration has not been completed in this session.',
            }}
          >
            <div className="grid gap-3 md:grid-cols-2">
              <div className="rounded-xl border border-slate-800 bg-slate-950/60 p-4">
                <p className="font-medium text-slate-200">Available experiment</p>
                <p className="mt-2 text-sm text-slate-400">Controlled checkoutservice pod deletion and Kubernetes replacement verification.</p>
                <p className="mt-3 text-xs text-slate-500">Status: {demo?.stage || 'Not started'}</p>
              </div>
              <div className="rounded-xl border border-slate-800 bg-slate-950/60 p-4">
                <p className="font-medium text-slate-200">Offline model evaluation</p>
                <p className="mt-2 text-sm text-slate-400">Saved RCAEval test metrics are displayed in the Offline LSTM section.</p>
                <p className="mt-3 text-xs text-slate-500">No live detection performance is claimed.</p>
              </div>
            </div>
          </Panel>

          <footer className="flex flex-wrap items-center justify-between gap-3 border-t border-slate-800 pt-5 text-xs text-slate-500">
            <span>CloudHeal · K3s · Online Boutique · FastAPI · React/Vite</span>
            <span className="flex items-center gap-2"><Timer size={14} /> Data refresh: 3 seconds</span>
          </footer>
        </main>
      </div>
    </div>
  )
}

function Brand() {
  return (
    <div className="flex items-center gap-3">
      <div className="rounded-xl bg-cyan-400 p-2 text-slate-950"><Layers3 size={22} /></div>
      <div><p className="font-bold text-white">CloudHeal</p><p className="text-xs text-slate-500">Operations Console</p></div>
    </div>
  )
}

export default App