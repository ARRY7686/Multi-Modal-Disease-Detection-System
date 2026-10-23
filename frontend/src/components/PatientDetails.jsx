import { LESION_SITES, buildDetailsText } from '../lib/domain'

const FREE_TEXT_EXAMPLES = [
  '62-year-old male. PA chest radiograph.',
  '45-year-old female. Lesion on the back.',
  'Elderly patient with a changing mole on the face.',
]

function Field({ label, children }) {
  return (
    <label className="block">
      <span className="mb-1.5 block text-[11px] uppercase tracking-[0.12em] text-ink-muted">{label}</span>
      {children}
    </label>
  )
}

export default function PatientDetails({ modality, details, onChange }) {
  const set = (patch) => onChange({ ...details, ...patch })
  const structured = details.kind === 'structured'
  const generated = buildDetailsText({ modality, ...details })

  return (
    <div>
      <div className="mb-3 flex items-center justify-between gap-3">
        <span className="eyebrow">Patient details</span>
        <div className="toggle-group" role="tablist" aria-label="Input style">
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
              className={`toggle ${details.kind === k ? 'toggle-on' : ''}`}
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
          <div className="mt-4 border-l border-violet/60 pl-3">
            <span className="text-[11px] uppercase tracking-[0.12em] text-ink-muted">Sent to the model</span>
            <p className="mt-1 font-mono text-xs text-ink-2">
              {generated || <span className="text-ink-muted">nothing: image only</span>}
            </p>
          </div>
          <p className="mt-3 text-xs leading-relaxed text-ink-muted">
            {modality === 'auto' && 'Pick a modality to add the X-ray view or lesion site. '}
            {modality === 'brain'
              ? 'Brain MRIs came with no clinical details, so the model only ever saw this placeholder sentence.'
              : 'Same sentence shape as the training metadata.'}
          </p>
        </>
      ) : (
        <>
          <textarea
            rows={3}
            value={details.text}
            onChange={(e) => set({ text: e.target.value })}
            placeholder="Clinical notes"
            className="field resize-none"
          />
          <div className="mt-2 flex flex-wrap gap-2">
            {FREE_TEXT_EXAMPLES.map((ex) => (
              <button key={ex} type="button" onClick={() => set({ text: ex })} className="chip hover:border-line-strong hover:text-ink">
                {ex}
              </button>
            ))}
          </div>
          <p className="mt-3 text-xs leading-relaxed text-status-warning">
            The text model only saw age, sex and view or site sentences in training. Free-form symptoms are outside
            that, so expect weak details-only scores.
          </p>
        </>
      )}
    </div>
  )
}
