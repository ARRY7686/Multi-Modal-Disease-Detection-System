import { useEffect, useRef, useState } from 'react'
import { Close, Upload } from './Icons'

const MAX_MB = 10

export default function ImageDropzone({ preview, fileName, onChange }) {
  const inputRef = useRef(null)
  const [dragging, setDragging] = useState(false)
  const [error, setError] = useState(null)

  const accept = (file) => {
    if (!file) return
    if (!file.type.startsWith('image/')) return setError('That file is not an image.')
    if (file.size > MAX_MB * 1024 * 1024) return setError(`Images up to ${MAX_MB} MB.`)
    setError(null)
    onChange(file)
  }

  // Paste an image from the clipboard anywhere on the page
  useEffect(() => {
    const onPaste = (e) => {
      const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith('image/'))
      if (item) accept(item.getAsFile())
    }
    window.addEventListener('paste', onPaste)
    return () => window.removeEventListener('paste', onPaste)
  })

  const clear = () => {
    onChange(null)
    if (inputRef.current) inputRef.current.value = ''
  }

  return (
    <section className="card card-pad">
      <div className="mb-3 flex items-baseline justify-between">
        <h2 className="card-title">Medical image</h2>
        <span className="text-xs text-ink-muted">optional</span>
      </div>

      {preview ? (
        <div className="relative overflow-hidden rounded-xl border border-line bg-black">
          <img src={preview} alt="Uploaded scan" className="h-64 w-full object-contain" />
          <div className="absolute inset-x-0 bottom-0 flex items-center justify-between bg-gradient-to-t from-black/80 to-transparent px-3 pb-2 pt-8">
            <span className="truncate text-xs text-ink-2">{fileName}</span>
            <button type="button" onClick={() => inputRef.current?.click()} className="text-xs text-ink-2 hover:text-ink">
              Replace
            </button>
          </div>
          <button
            type="button"
            onClick={clear}
            className="absolute right-2 top-2 rounded-full bg-black/70 p-1.5 text-ink-2 hover:text-ink"
            aria-label="Remove image"
          >
            <Close />
          </button>
        </div>
      ) : (
        <button
          type="button"
          onClick={() => inputRef.current?.click()}
          onDragOver={(e) => {
            e.preventDefault()
            setDragging(true)
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault()
            setDragging(false)
            accept(e.dataTransfer.files?.[0])
          }}
          className={`flex h-56 w-full flex-col items-center justify-center rounded-xl border border-dashed text-center transition-colors ${
            dragging ? 'border-accent bg-accent-soft' : 'border-line-strong hover:border-ink-muted hover:bg-surface-2'
          }`}
        >
          <span className="mb-3 rounded-full bg-surface-3 p-3 text-ink-2">
            <Upload className="h-5 w-5" />
          </span>
          <span className="text-sm font-medium text-ink">Drop an image, click to browse, or paste</span>
          <span className="mt-1 text-xs text-ink-muted">Chest X-ray, dermoscopy photo or brain MRI slice (JPG, PNG)</span>
        </button>
      )}
      {error && <p className="mt-2 text-xs text-status-serious">{error}</p>}
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={(e) => accept(e.target.files?.[0])}
      />
    </section>
  )
}
