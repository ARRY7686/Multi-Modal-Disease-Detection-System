import { useCallback, useEffect, useState } from 'react'

const KEY = 'mmdd.history.v1'
const LIMIT = 12

function load() {
  try {
    const raw = window.localStorage.getItem(KEY)
    return raw ? JSON.parse(raw) : []
  } catch {
    return []
  }
}

function save(entries) {
  try {
    window.localStorage.setItem(KEY, JSON.stringify(entries))
  } catch {
    // Storage full or blocked (private window): history stays in memory only.
  }
}

// Heatmaps are large base64 PNGs, so only the in-memory copy keeps them.
const lighten = (entry) => ({
  ...entry,
  results: Object.fromEntries(
    Object.entries(entry.results).map(([mode, r]) => [
      mode,
      r && { ...r, explanation: r.explanation && { ...r.explanation, heatmap_base64: null } },
    ]),
  ),
})

export default function useHistory() {
  const [entries, setEntries] = useState(load)

  useEffect(() => save(entries.map(lighten)), [entries])

  const add = useCallback((entry) => {
    setEntries((prev) => [entry, ...prev.filter((e) => e.id !== entry.id)].slice(0, LIMIT))
  }, [])
  const clear = useCallback(() => setEntries([]), [])

  return { entries, add, clear }
}

/** Small JPEG data URL of an image file, for history thumbnails. */
export function thumbnail(dataUrl, size = 112) {
  return new Promise((resolve) => {
    if (!dataUrl) return resolve(null)
    const img = new Image()
    img.onload = () => {
      const scale = size / Math.max(img.width, img.height)
      const canvas = document.createElement('canvas')
      canvas.width = Math.max(1, Math.round(img.width * scale))
      canvas.height = Math.max(1, Math.round(img.height * scale))
      canvas.getContext('2d').drawImage(img, 0, 0, canvas.width, canvas.height)
      resolve(canvas.toDataURL('image/jpeg', 0.7))
    }
    img.onerror = () => resolve(null)
    img.src = dataUrl
  })
}
