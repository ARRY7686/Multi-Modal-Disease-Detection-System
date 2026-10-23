import { LESION_SITES, buildDetailsText } from '../lib/domain'

const FREE_TEXT_EXAMPLES = [
  '62-year-old male. PA chest radiograph.',
  '45-year-old female. Lesion on the back.',
  'Elderly patient with a changing mole on the face.',
]

function Field({ label, children }) {
  return (
    <label className="block">
      <span className="eyebrow mb-1 block">{label}</span>
      {children}
    </label>
  )
}

export default function PatientDetails({ modality, details, onChange }) {
  const set = (patch) => onChange({ ...details, ...patch })
  const structured = details.kind === 'structured'
  const generated = buildDetailsText({ modality, ...details })

  return (
    <section className="card card-pad">
      <div className="mb-3 flex items-center justify-between gap-3">
        <h2 className="card-title">Patient details</h2>
        <div className="seg w-44 sm:w-52" role="tablist" aria-label="Input style">
          {[
            ['structured', 'Structured'],
            ['free', 'Free text'],
          ].map(([k, label]) => (
            <button
              key={k}
              type="button"
              role="tab"
              aria-selected={details.kind === k}
              onClick={() => set({ kind: k })}
              className={`seg-btn ${details.kind === k ? 'seg-btn-on' : ''}`}
            >
              {label}
            </button>
          ))}
        </div>
      </div>

      {structured ? (
        <>
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
            <Field label="Age">
              <input
                type="number"
                min="1"
                max="100"
                inputMode="numeric"
                placeholder="e.g. 58"
                value={details.age}
                onChange={(e) => set({ age: e.target.value })}
                className="field num"
              />
            </Field>
            <Field label="Sex">
              <select value={details.sex} onChange={(e) => set({ sex: e.target.value })} className="field">
                <option value="">Not given</option>
                <option value="male">Male</option>
                <option value="female">Female</option>
              </select>
            </Field>
            {modality === 'chest' && (
              <Field label="X-ray view">
                <select value={details.view} onChange={(e) => set({ view: e.target.value })} className="field">
                  <option value="">Not given</option>
                  <option value="PA">PA</option>
                  <option value="AP">AP</option>
                </select>
              </Field>
            )}
            {modality === 'skin' && (
              <Field label="Lesion site">
                <select value={details.site} onChange={(e) => set({ site: e.target.value })} className="field">
                  <option value="">Not given</option>
                  {LESION_SITES.map((s) => (
                    <option key={s} value={s}>
                      {s}
                    </option>
                  ))}
                </select>
              </Field>
            )}
          </div>
          <div className="mt-4 rounded-xl border border-line bg-surface-2 px-3 py-2.5">
            <span className="eyebrow">Text sent to the model</span>
            <p className="mt-1 font-mono text-xs text-ink-2">
              {generated || <span className="text-ink-muted">none (image only)</span>}
            </p>
          </div>
          <p className="mt-2 text-xs leading-relaxed text-ink-muted">
            {modality === 'auto' && 'Pick a modality to add the X-ray view or lesion site. '}
            {modality === 'brain'
              ? 'Brain MRIs came with no clinical details, so the model only ever saw this placeholder sentence.'
              : 'Written in the exact sentence shape of the training metadata (age, sex, view or site).'}
          </p>
        </>
      ) : (
        <>
          <textarea
            rows={4}
            value={details.text}
            onChange={(e) => set({ text: e.target.value })}
            placeholder="Clinical notes"
            className="field resize-none"
          />
          <div className="mt-2 flex flex-wrap gap-2">
            {FREE_TEXT_EXAMPLES.map((ex) => (
              <button
                key={ex}
                type="button"
                onClick={() => set({ text: ex })}
                className="chip hover:border-ink-muted hover:text-ink"
              >
                {ex}
              </button>
            ))}
          </div>
          <p className="mt-2 text-xs leading-relaxed text-status-warning">
            The text model was trained only on age, sex and view or site sentences. Free-form symptoms are
            outside what it learned, so expect weak text-only scores.
          </p>
        </>
      )}
    </section>
  )
}
