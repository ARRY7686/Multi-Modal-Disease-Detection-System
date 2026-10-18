import { useState } from 'react'

export default function HeatmapViewer({ heatmapBase64, heatmapClass, originalImage }) {
  const [showHeatmap, setShowHeatmap] = useState(true)

  if (!heatmapBase64) return null

  const heatmapSrc = `data:image/png;base64,${heatmapBase64}`

  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-lg font-semibold text-gray-800">Grad-CAM Heatmap</h2>
        <button
          onClick={() => setShowHeatmap((v) => !v)}
          className="text-xs bg-gray-100 hover:bg-gray-200 px-3 py-1.5 rounded-full
                     transition-colors text-gray-600"
        >
          {showHeatmap ? 'Show Original' : 'Show Heatmap'}
        </button>
      </div>
      <div className="relative rounded-lg overflow-hidden bg-gray-100">
        <img
          src={showHeatmap ? heatmapSrc : originalImage || heatmapSrc}
          alt={showHeatmap ? 'Grad-CAM heatmap overlay' : 'Original image'}
          className="w-full h-64 object-contain"
        />
        <div className="absolute bottom-2 left-2 text-xs bg-black/50 text-white px-2 py-1 rounded">
          {showHeatmap ? 'Heatmap Overlay' : 'Original'}
        </div>
      </div>
      <p className="text-xs text-gray-400 mt-2">
        Red regions pushed the model towards
        {heatmapClass ? <strong> {heatmapClass.replace(/_/g, ' ')}</strong> : ' its top prediction'}.
      </p>
    </div>
  )
}
