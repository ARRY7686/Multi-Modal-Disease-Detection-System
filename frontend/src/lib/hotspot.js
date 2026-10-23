// Where a Grad-CAM overlay is hottest. The API blends a JET colormap onto
// the scan, so "hot" is red well above blue. Returns {x, y} in 0..1 image
// coordinates: the centroid of the hottest 3% of pixels.
export function findHotspot(dataUrl, size = 96) {
  return new Promise((resolve) => {
    const img = new Image()
    img.onload = () => {
      const scale = size / Math.max(img.width, img.height)
      const w = Math.max(1, Math.round(img.width * scale))
      const h = Math.max(1, Math.round(img.height * scale))
      const canvas = document.createElement('canvas')
      canvas.width = w
      canvas.height = h
      const ctx = canvas.getContext('2d')
      ctx.drawImage(img, 0, 0, w, h)
      const { data } = ctx.getImageData(0, 0, w, h)
      const heat = new Float32Array(w * h)
      for (let i = 0; i < w * h; i++) heat[i] = data[i * 4] - data[i * 4 + 2]
      const cutoff = [...heat].sort((a, b) => b - a)[Math.max(0, Math.floor(w * h * 0.03))]
      let sx = 0
      let sy = 0
      let n = 0
      for (let i = 0; i < w * h; i++) {
        if (heat[i] >= cutoff) {
          sx += i % w
          sy += Math.floor(i / w)
          n++
        }
      }
      resolve(n ? { x: (sx / n + 0.5) / w, y: (sy / n + 0.5) / h } : null)
    }
    img.onerror = () => resolve(null)
    img.src = dataUrl
  })
}
