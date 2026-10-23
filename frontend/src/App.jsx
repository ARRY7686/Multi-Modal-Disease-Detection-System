import { useCallback, useEffect, useRef, useState } from 'react'
import { checkHealth, errorMessage, predict } from './api'
import About from './components/About'
import Header from './components/Header'
import HistoryPanel from './components/HistoryPanel'
import { Alert } from './components/Icons'
import ModalityPicker from './components/ModalityPicker'
import ModelCard from './components/ModelCard'
import PatientDetails from './components/PatientDetails'
import ResultsPanel from './components/ResultsPanel'
import Section from './components/Section'
import Stage from './components/Stage'
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

const FEATURES = [
  ['#a99ff0', 'Fusion', 'EfficientNet-B4 × BioBERT'],
  ['#d946ef', 'Explainable', 'Grad-CAM + attention'],
  ['#199e70', 'Leak-free', 'patient-disjoint test'],
]

function readDataUrl(file) {
  return new Promise((resolve) => {
    const reader = new FileReader()
    reader.onloadend = () => resolve(reader.result)
    reader.readAsDataURL(file)
  })
}

function Squiggle() {
  return (
    <svg viewBox="0 0 64 16" className="ml-3 inline-block h-5 w-16 align-middle" aria-hidden="true">
      <path
        d="M1 8c4-7 8-7 12 0s8 7 12 0 8-7 12 0 8 7 12 0 8-7 12 0"
        fill="none"
        stroke="url(#sq)"
        strokeWidth="1.6"
        strokeLinecap="round"
      />
      <defs>
        <linearGradient id="sq" x1="0" x2="1">
          <stop offset="0" stopColor="#a99ff0" />
          <stop offset="1" stopColor="#d946ef" />
        </linearGradient>
      </defs>
    </svg>
  )
}

function Steps({ steps }) {
  return (
    <nav className="no-print fixed left-6 top-1/2 z-30 hidden -translate-y-1/2 flex-col items-center gap-5 2xl:flex" aria-label="Progress">
      {steps.map(([id, label, state]) => (
        <a
          key={id}
          href={`#${id}`}
          title={label}
          onClick={(e) => {
            e.preventDefault()
            document.getElementById(id)?.scrollIntoView({ behavior: 'smooth', block: 'start' })
          }}
        >
          <span
            className={`block rounded-full transition-all ${
              state === 'current'
                ? 'h-3 w-3 border border-violet ring-4 ring-violet/15'
                : state === 'done'
                  ? 'h-2.5 w-2.5 bg-ink'
                  : 'h-1.5 w-1.5 bg-ink-muted'
            }`}
          />
        </a>
      ))}
    </nav>
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
  const history = useHistory()

  // Tabs live in the URL hash, so #model and #about are linkable
  const setTab = (k) => {
    window.history.replaceState(null, '', k === 'analyze' ? window.location.pathname : `#${k}`)
    setTabState(k)
    window.scrollTo({ top: 0 })
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

  const onImage = useCallback(async (f) => {
    setFile(f)
    setPreview(f ? await readDataUrl(f) : null)
    setActive(null)
    setError(null)
  }, [])

  const text = details.kind === 'structured' ? buildDetailsText({ modality, ...details }) : details.text.trim()
  const canRun = (file || text) && !loading && health !== false
  const willCompare = compare && file && text

  const run = useCallback(async () => {
    if (!file && !text) return setError('Add a scan, patient details, or both.')
    setLoading(true)
    setError(null)
    try {
      const first = await predict({ image: file, symptoms: text, modality })
      const results = { [first.mode]: first }
      if (willCompare) {
        // Same case each way, with the modality fixed to the multimodal call's,
        // so all three rank the same classes.
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
    } catch (e) {
      setError(errorMessage(e))
    } finally {
      setLoading(false)
    }
  }, [file, text, modality, willCompare, preview, history])

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
    setError(null)
  }

  const selectHistory = (e) => {
    setActive(e)
    setPreview(previews.current[e.id] || e.thumb)
    setFile(null)
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }

  const stageResult = active ? active.results[active.primary] : null
  const steps = [
    ['scan', 'Scan', preview ? 'done' : 'current'],
    ['inputs', 'Details', text ? 'done' : preview ? 'current' : 'todo'],
    ['findings', 'Findings', active ? 'current' : 'todo'],
  ]

  return (
    <div className="min-h-screen overflow-x-clip">
      <Header tabs={TABS} tab={tab} onTab={setTab} health={health} />

      <main className="mx-auto max-w-[1400px] px-5 pb-16 sm:px-8">
        {tab === 'analyze' && (
          <>
            <Steps steps={steps} />
            <div className="grid gap-x-12 gap-y-10 pt-10 lg:grid-cols-[minmax(0,5fr)_minmax(0,7fr)] lg:pt-16">
              {/* Headline */}
              <div className="no-print">
                <div className="eyebrow mb-4">Image + patient details · 25 findings</div>
                <h1 className="display text-[40px] leading-[1.08] sm:text-[52px]">
                  Reading the scan
                  <br />
                  and the patient, together
                  <Squiggle />
                </h1>
                <ul className="mt-8 grid gap-5 sm:grid-cols-3 lg:grid-cols-1 xl:grid-cols-3">
                  {FEATURES.map(([color, title, sub]) => (
                    <li key={title} className="flex gap-3">
                      <span className="tri mt-1.5" style={{ borderBottomColor: color }} />
                      <span>
                        <span className="block text-[15px] text-ink">{title}</span>
                        <span className="text-xs text-ink-muted">{sub}</span>
                      </span>
                    </li>
                  ))}
                </ul>
              </div>

              {/* Stage */}
              <div
                id="scan"
                className="scroll-mt-28 lg:sticky lg:top-24 lg:col-start-2 lg:row-span-2 lg:row-start-1 lg:self-start"
              >
                <Stage preview={preview} fileName={file?.name} onImage={onImage} result={stageResult} loading={loading} />
              </div>

              {/* Inputs */}
              <div id="inputs" className="glass no-print scroll-mt-28 space-y-7 p-6 sm:p-7">
                <ModalityPicker value={modality} onChange={setModality} />
                <PatientDetails modality={modality} details={details} onChange={setDetails} />
                <label className="flex cursor-pointer items-start gap-3">
                  <input
                    type="checkbox"
                    checked={compare}
                    onChange={(e) => setCompare(e.target.checked)}
                    className="mt-0.5 h-4 w-4 accent-[#9085e9]"
                  />
                  <span>
                    <span className="block text-sm text-ink">Compare all three modes</span>
                    <span className="block text-xs text-ink-muted">With a scan and details, also score each on its own.</span>
                  </span>
                </label>
                {error && (
                  <div className="flex gap-2 rounded-2xl border border-status-critical/40 bg-status-critical/10 p-3 text-sm text-ink-2">
                    <Alert className="mt-0.5 h-4 w-4 shrink-0 text-status-critical" />
                    {error}
                  </div>
                )}
                <div className="flex flex-wrap items-center gap-3">
                  <button type="button" onClick={run} disabled={!canRun} className="pill-solid px-7 py-3">
                    {loading ? (
                      <>
                        <span className="h-3.5 w-3.5 animate-spin rounded-full border-2 border-page/30 border-t-page" />
                        Analyzing
                      </>
                    ) : (
                      'Analyze'
                    )}
                  </button>
                  <button type="button" onClick={reset} className="pill px-6 py-3">
                    Reset
                  </button>
                  <span className="text-[11px] text-ink-muted sm:ml-auto">
                    {file && text ? 'Multimodal' : file ? 'Image only' : text ? 'Details only' : 'No input yet'} · Ctrl + Enter
                  </span>
                </div>
              </div>
            </div>

            <div className="mt-16 space-y-6">
              {active ? (
                <ResultsPanel entry={active} />
              ) : (
                <Section
                  id="findings"
                  n="01"
                  highlight="Findings"
                  title="appear here"
                  aside={
                    <button type="button" onClick={() => setTab('model')} className="pill pill-sm">
                      Read the model card
                    </button>
                  }
                >
                  <p className="max-w-3xl text-sm leading-relaxed text-ink-2">
                    Scores for each finding with its decision threshold, the three modes side by side, and which words of
                    the patient details the text encoder attended to. On held-out data the image does almost all of the
                    work; the Model card shows the numbers.
                  </p>
                </Section>
              )}
              {history.entries.length > 0 && (
                <Section
                  id="history"
                  n="··"
                  highlight="Recent"
                  title="analyses"
                  className="no-print"
                  aside={
                    <button type="button" onClick={history.clear} className="pill pill-sm">
                      Clear
                    </button>
                  }
                >
                  <HistoryPanel entries={history.entries} activeId={active?.id} onSelect={selectHistory} />
                </Section>
              )}
            </div>
          </>
        )}
        {tab === 'model' && <ModelCard />}
        {tab === 'about' && <About />}
      </main>

      <footer className="no-print border-t border-line py-8 text-center text-[11px] uppercase tracking-[0.16em] text-ink-muted">
        Research prototype · not for clinical use
      </footer>
    </div>
  )
}
