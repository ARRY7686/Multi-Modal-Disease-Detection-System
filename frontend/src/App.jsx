import { useCallback, useEffect, useRef, useState } from 'react'
import { checkHealth, errorMessage, predict } from './api'
import About from './components/About'
import HistoryPanel from './components/HistoryPanel'
import { Alert, Github, Layers, Pulse, Spark } from './components/Icons'
import ImageDropzone from './components/ImageDropzone'
import ModalityPicker from './components/ModalityPicker'
import ModelCard from './components/ModelCard'
import PatientDetails from './components/PatientDetails'
import ResultsPanel from './components/ResultsPanel'
import useHistory, { thumbnail } from './hooks/useHistory'
import { buildDetailsText } from './lib/domain'

const TABS = [
  ['analyze', 'Analyze'],
  ['model', 'Model card'],
  ['about', 'About'],
]
const tabFromHash = () => {
  const h = window.location.hash.slice(1)
  return TABS.some(([k]) => k === h) ? h : 'analyze'
}
const EMPTY_DETAILS = { kind: 'structured', age: '', sex: '', view: '', site: '', text: '' }

function readDataUrl(file) {
  return new Promise((resolve) => {
    const reader = new FileReader()
    reader.onloadend = () => resolve(reader.result)
    reader.readAsDataURL(file)
  })
}

function StatusPill({ health }) {
  const state = health === null ? 'checking' : health === false ? 'offline' : health.status
  const style = {
    checking: ['bg-ink-muted animate-pulse', 'Connecting'],
    offline: ['bg-status-critical', 'API offline'],
    degraded: ['bg-status-warning', 'No trained weights'],
    ok: ['bg-status-good', 'Models loaded'],
  }[state]
  return (
    <span className="chip">
      <span className={`h-2 w-2 rounded-full ${style[0]}`} />
      {style[1]}
    </span>
  )
}

export default function App() {
  const [tab, setTabState] = useState(tabFromHash)
  const [health, setHealth] = useState(null)
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [modality, setModality] = useState('auto')
  const [details, setDetails] = useState(EMPTY_DETAILS)
  const [compare, setCompare] = useState(true)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [active, setActive] = useState(null)
  const previews = useRef({}) // entry id -> full-size image, this session only
  const resultsRef = useRef(null)
  const history = useHistory()

  // Tabs live in the URL hash, so #model and #about are linkable
  const setTab = (k) => {
    window.history.replaceState(null, '', k === 'analyze' ? window.location.pathname : `#${k}`)
    setTabState(k)
  }
  useEffect(() => {
    const onHash = () => setTabState(tabFromHash())
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [])

  useEffect(() => {
    let alive = true
    const poll = () =>
      checkHealth()
        .then((h) => alive && setHealth(h))
        .catch(() => alive && setHealth(false))
    poll()
    const id = setInterval(poll, 30000)
    return () => {
      alive = false
      clearInterval(id)
    }
  }, [])

  const onImage = async (f) => {
    setFile(f)
    setPreview(f ? await readDataUrl(f) : null)
    setError(null)
  }

  const text = details.kind === 'structured' ? buildDetailsText({ modality, ...details }) : details.text.trim()
  const canRun = (file || text) && !loading && health !== false
  const willCompare = compare && file && text

  const run = useCallback(async () => {
    if (!file && !text) return setError('Add an image, patient details, or both.')
    setLoading(true)
    setError(null)
    try {
      const first = await predict({ image: file, symptoms: text, modality })
      const results = { [first.mode]: first }
      if (willCompare) {
        // Score the same case each way, with the modality fixed to the one the
        // multimodal call used so all three rank the same classes.
        const [img, txt] = await Promise.allSettled([
          predict({ image: file, modality: first.modality }),
          predict({ symptoms: text, modality: first.modality }),
        ])
        if (img.status === 'fulfilled') results.image_only = img.value
        if (txt.status === 'fulfilled') results.text_only = txt.value
      }
      const entry = {
        id: `${Date.now()}`,
        ts: Date.now(),
        primary: first.mode,
        results,
        input: { modality, text, fileName: file?.name || null },
        thumb: await thumbnail(preview),
      }
      previews.current[entry.id] = preview
      history.add(entry)
      setActive(entry)
      // Single-column layout: results render below the inputs
      if (window.innerWidth < 1024) {
        requestAnimationFrame(() => resultsRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }))
      }
    } catch (e) {
      setError(errorMessage(e))
    } finally {
      setLoading(false)
    }
  }, [file, text, modality, willCompare, preview, history])

  // Ctrl/Cmd + Enter runs the analysis
  useEffect(() => {
    const onKey = (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'Enter' && tab === 'analyze' && canRun) run()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [run, canRun, tab])

  const reset = () => {
    onImage(null)
    setDetails(EMPTY_DETAILS)
    setModality('auto')
    setActive(null)
    setError(null)
  }

  return (
    <div className="min-h-screen">
      <header className="sticky top-0 z-30 border-b border-line bg-page/85 backdrop-blur no-print">
        <div className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-3 px-4 py-3 sm:px-6">
          <div className="flex items-center gap-3">
            <span className="rounded-xl bg-accent-soft p-2 text-accent">
              <Pulse className="h-5 w-5" />
            </span>
            <div>
              <div className="text-sm font-semibold text-ink">Multi-Modal Disease Detection</div>
              <div className="text-xs text-ink-muted">Image + patient details · 25 findings · Grad-CAM</div>
            </div>
          </div>
          <nav className="seg order-3 w-full sm:order-none sm:w-auto" aria-label="Sections">
            {TABS.map(([k, label]) => (
              <button
                key={k}
                type="button"
                onClick={() => setTab(k)}
                className={`seg-btn px-4 ${tab === k ? 'seg-btn-on' : ''}`}
              >
                {label}
              </button>
            ))}
          </nav>
          <div className="flex items-center gap-2">
            <StatusPill health={health} />
            <a
              href="https://github.com/ARRY7686/Multi-Modal-Disease-Detection-System"
              target="_blank"
              rel="noreferrer"
              className="rounded-lg p-2 text-ink-2 hover:bg-surface-2 hover:text-ink"
              aria-label="GitHub repository"
            >
              <Github />
            </a>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-4 py-6 sm:px-6 sm:py-8">
        {tab === 'analyze' && (
          <div className="grid gap-6 lg:grid-cols-[minmax(0,5fr)_minmax(0,7fr)]">
            <div className="space-y-6 no-print">
              <ImageDropzone preview={preview} fileName={file?.name} onChange={onImage} />
              <ModalityPicker value={modality} onChange={setModality} />
              <PatientDetails modality={modality} details={details} onChange={setDetails} />

              <section className="card card-pad space-y-4">
                <label className="flex cursor-pointer items-start gap-3">
                  <input
                    type="checkbox"
                    checked={compare}
                    onChange={(e) => setCompare(e.target.checked)}
                    className="mt-0.5 h-4 w-4 accent-[#3987e5]"
                  />
                  <span>
                    <span className="flex items-center gap-1.5 text-sm text-ink">
                      <Layers className="h-4 w-4 text-ink-2" /> Compare all three modes
                    </span>
                    <span className="block text-xs text-ink-muted">
                      With an image and details, also score image only and details only.
                    </span>
                  </span>
                </label>
                {error && (
                  <div className="flex gap-2 rounded-xl border border-status-critical/40 bg-status-critical/10 p-3 text-sm text-ink-2">
                    <Alert className="mt-0.5 h-4 w-4 shrink-0 text-status-critical" />
                    {error}
                  </div>
                )}
                <div className="flex gap-3">
                  <button type="button" onClick={run} disabled={!canRun} className="btn-primary flex-1">
                    {loading ? (
                      <>
                        <span className="h-4 w-4 animate-spin rounded-full border-2 border-white/30 border-t-white" />
                        Analyzing
                      </>
                    ) : (
                      <>
                        <Spark /> Analyze
                      </>
                    )}
                  </button>
                  <button type="button" onClick={reset} className="btn-ghost">
                    Reset
                  </button>
                </div>
                <p className="text-center text-[11px] text-ink-muted">
                  {file && text ? 'Multimodal' : file ? 'Image only' : text ? 'Details only' : 'No input yet'}
                  {' · '}
                  <kbd className="rounded border border-line-strong px-1">Ctrl</kbd> +{' '}
                  <kbd className="rounded border border-line-strong px-1">Enter</kbd>
                </p>
              </section>

              <HistoryPanel
                entries={history.entries}
                activeId={active?.id}
                onSelect={setActive}
                onClear={history.clear}
              />
            </div>

            <div ref={resultsRef} className="scroll-mt-24">
              {active ? (
                <ResultsPanel entry={active} original={previews.current[active.id] || active.thumb} />
              ) : (
                <div className="card flex min-h-[28rem] flex-col items-center justify-center p-10 text-center">
                  <span className="mb-4 rounded-2xl bg-surface-3 p-4 text-ink-2">
                    <Pulse className="h-7 w-7" />
                  </span>
                  <p className="text-base font-medium text-ink">{loading ? 'Running the models' : 'Results appear here'}</p>
                  <p className="mt-1 max-w-sm text-sm text-ink-muted">
                    {loading
                      ? 'The first request after a cold start can take a little longer on CPU.'
                      : 'Add a scan, the patient details, or both, then press Analyze.'}
                  </p>
                </div>
              )}
            </div>
          </div>
        )}
        {tab === 'model' && <ModelCard />}
        {tab === 'about' && <About />}
      </main>

      <footer className="border-t border-line py-6 text-center text-xs text-ink-muted no-print">
        Research prototype. Not for clinical use.
      </footer>
    </div>
  )
}
