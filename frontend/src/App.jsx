import { useState, useEffect } from 'react'
import ImageUpload from './components/ImageUpload'
import SymptomInput from './components/SymptomInput'
import PredictionResults from './components/PredictionResults'
import HeatmapViewer from './components/HeatmapViewer'
import ExplanationPanel from './components/ExplanationPanel'
import { predict, predictMultimodal, checkHealth } from './api'

export default function App() {
  const [image, setImage] = useState(null)
  const [imagePreview, setImagePreview] = useState(null)
  const [symptoms, setSymptoms] = useState('')
  const [modality, setModality] = useState('auto')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [apiStatus, setApiStatus] = useState('checking')

  useEffect(() => {
    checkHealth()
      .then(() => setApiStatus('online'))
      .catch(() => setApiStatus('offline'))
  }, [])

  const handleImageChange = (file) => {
    setImage(file)
    if (file) {
      const reader = new FileReader()
      reader.onloadend = () => setImagePreview(reader.result)
      reader.readAsDataURL(file)
    } else {
      setImagePreview(null)
    }
    setResult(null)
    setError(null)
  }

  const handleSubmit = async () => {
    if (!image && !symptoms.trim()) {
      setError('Please provide an image and/or symptoms.')
      return
    }
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      let data
      if (image && symptoms.trim()) {
        data = await predictMultimodal({ image, symptoms: symptoms.trim(), modality })
      } else {
        data = await predict({ image, symptoms: symptoms.trim() || undefined, modality })
      }
      setResult(data)
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Prediction failed.')
    } finally {
      setLoading(false)
    }
  }

  const handleClear = () => {
    setImage(null)
    setImagePreview(null)
    setSymptoms('')
    setModality('auto')
    setResult(null)
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-gray-900">
              Multi-Modal Disease Detection
            </h1>
            <p className="text-sm text-gray-500 mt-1">
              AI-powered analysis of medical images &amp; patient symptoms
            </p>
          </div>
          <div className="flex items-center gap-2">
            <span
              className={`inline-block w-2.5 h-2.5 rounded-full ${
                apiStatus === 'online'
                  ? 'bg-green-500'
                  : apiStatus === 'offline'
                  ? 'bg-red-500'
                  : 'bg-yellow-400 animate-pulse'
              }`}
            />
            <span className="text-xs text-gray-500 uppercase tracking-wide">
              API {apiStatus}
            </span>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* ── Left Column: Inputs ──────────────────────────── */}
          <div className="space-y-6">
            <ImageUpload
              onImageChange={handleImageChange}
              preview={imagePreview}
            />
            <SymptomInput value={symptoms} onChange={setSymptoms} />

            {/* Modality Selector */}
            <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-4">
              <label className="block text-sm font-semibold text-gray-700 mb-2">
                Image Modality
              </label>
              <div className="flex gap-2 flex-wrap">
                {['auto', 'chest', 'skin', 'brain'].map((m) => (
                  <button
                    key={m}
                    onClick={() => setModality(m)}
                    className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                      modality === m
                        ? 'bg-blue-600 text-white'
                        : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                    }`}
                  >
                    {m === 'auto' ? 'Auto-detect' : m.charAt(0).toUpperCase() + m.slice(1)}
                  </button>
                ))}
              </div>
            </div>

            {error && (
              <div className="bg-red-50 border border-red-200 rounded-lg p-4 text-red-700 text-sm">
                {error}
              </div>
            )}

            <div className="flex gap-3">
              <button
                onClick={handleSubmit}
                disabled={loading || apiStatus !== 'online'}
                className="flex-1 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400
                           text-white font-medium py-3 px-6 rounded-lg transition-colors
                           flex items-center justify-center gap-2"
              >
                {loading ? (
                  <>
                    <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                      <circle
                        className="opacity-25"
                        cx="12" cy="12" r="10"
                        stroke="currentColor" strokeWidth="4" fill="none"
                      />
                      <path
                        className="opacity-75"
                        fill="currentColor"
                        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
                      />
                    </svg>
                    Analyzing...
                  </>
                ) : (
                  'Analyze'
                )}
              </button>
              <button
                onClick={handleClear}
                className="px-6 py-3 border border-gray-300 text-gray-700 rounded-lg
                           hover:bg-gray-100 transition-colors font-medium"
              >
                Clear
              </button>
            </div>

            {/* Mode indicator */}
            <div className="text-xs text-gray-400 text-center">
              {image && symptoms.trim()
                ? 'Mode: Multimodal (image + text)'
                : image
                ? 'Mode: Image only'
                : symptoms.trim()
                ? 'Mode: Text only'
                : 'Upload an image and/or enter symptoms'}
            </div>
          </div>

          {/* ── Right Column: Results ────────────────────────── */}
          <div className="space-y-6">
            {result && (
              <>
                <PredictionResults result={result} />
                {result.explanation?.heatmap_base64 && (
                  <HeatmapViewer
                    heatmapBase64={result.explanation.heatmap_base64}
                    originalImage={imagePreview}
                  />
                )}
                {result.explanation && (
                  <ExplanationPanel explanation={result.explanation} />
                )}
              </>
            )}
            {!result && !loading && (
              <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-12 text-center text-gray-400">
                <svg
                  className="mx-auto h-16 w-16 mb-4 text-gray-300"
                  fill="none" viewBox="0 0 24 24" stroke="currentColor"
                >
                  <path
                    strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
                    d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 
                       17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 
                       00-2 2v10a2 2 0 002 2z"
                  />
                </svg>
                <p className="font-medium">Results will appear here</p>
                <p className="text-sm mt-1">
                  Submit an image and/or symptoms to get predictions
                </p>
              </div>
            )}
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="mt-12 py-6 text-center text-xs text-gray-400 border-t border-gray-200">
        Multi-Modal Disease Detection System &mdash; For Research Purposes Only
      </footer>
    </div>
  )
}
