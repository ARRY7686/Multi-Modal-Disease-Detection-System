import { STATIC } from '../lib/gallery'
import { Github } from './Icons'

const REPO = 'https://github.com/ARRY7686/Multi-Modal-Disease-Detection-System'

function Logo() {
  return (
    <svg viewBox="0 0 32 32" className="h-8 w-8" aria-hidden="true">
      <circle cx="16" cy="16" r="13" fill="none" stroke="#f3f1f8" strokeWidth="1.6" />
      <path d="M5 17h6l2.5-6 4 11 2.5-5H27" fill="none" stroke="#a99ff0" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  )
}

function Status({ health }) {
  const state = STATIC ? 'static' : health === null ? 'checking' : health === false ? 'offline' : health.status
  const [dot, label] = {
    static: ['bg-violet', 'Static demo'],
    checking: ['bg-ink-muted animate-pulse', 'Connecting'],
    offline: ['bg-status-critical', 'API offline'],
    degraded: ['bg-status-warning', 'No trained weights'],
    ok: ['bg-status-good', 'Models loaded'],
  }[state]
  return (
    <span className="pill pill-sm cursor-default hover:border-line-strong">
      <span className={`h-1.5 w-1.5 rounded-full ${dot}`} />
      {label}
    </span>
  )
}

// Hairline-topped tabs; the active one gets a bright line and a ring marker.
export default function Header({ tabs, tab, onTab, health }) {
  return (
    <header className="no-print sticky top-0 z-40 border-b border-line bg-page/70 backdrop-blur-xl">
      <div className="mx-auto flex max-w-[1400px] flex-wrap items-center gap-x-10 gap-y-3 px-5 py-4 sm:px-8">
        <a href="#" onClick={(e) => { e.preventDefault(); onTab('analyze') }} className="flex items-center gap-3">
          <Logo />
          <span className="text-[15px] tracking-wide text-ink">
            MULTIMODAL <span className="text-violet">DX</span>
          </span>
        </a>
        <nav className="order-3 flex w-full gap-4 md:order-none md:w-auto md:flex-1 lg:gap-6" aria-label="Sections">
          {tabs.map(([k, label]) => {
            const on = tab === k
            return (
              <button
                key={k}
                type="button"
                onClick={() => onTab(k)}
                aria-current={on ? 'page' : undefined}
                className={`flex flex-1 items-center justify-between border-t pt-2 text-left text-[11px] uppercase tracking-[0.14em] transition-colors md:max-w-[190px] ${
                  on ? 'border-ink text-ink' : 'border-line-strong text-ink-muted hover:text-ink-2'
                }`}
              >
                {label}
                {on && <span className="h-2.5 w-2.5 rounded-full border border-ink" />}
              </button>
            )
          })}
        </nav>
        <div className="ml-auto flex items-center gap-3">
          <Status health={health} />
          <a href={REPO} target="_blank" rel="noreferrer" className="pill pill-sm" aria-label="GitHub repository">
            <Github className="h-3.5 w-3.5" /> Source
          </a>
        </div>
      </div>
    </header>
  )
}
