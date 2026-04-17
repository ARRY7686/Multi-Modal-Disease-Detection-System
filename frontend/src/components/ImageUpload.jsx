import { useRef } from 'react'

export default function ImageUpload({ onImageChange, preview }) {
  const inputRef = useRef(null)

  const handleDrop = (e) => {
    e.preventDefault()
    const file = e.dataTransfer.files?.[0]
    if (file && file.type.startsWith('image/')) {
      onImageChange(file)
    }
  }

  const handleDragOver = (e) => e.preventDefault()

  const handleFileSelect = (e) => {
    const file = e.target.files?.[0]
    if (file) onImageChange(file)
  }

  const handleRemove = () => {
    onImageChange(null)
    if (inputRef.current) inputRef.current.value = ''
  }

  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
      <h2 className="text-lg font-semibold mb-3 text-gray-800">
        Medical Image
      </h2>

      {preview ? (
        <div className="relative">
          <img
            src={preview}
            alt="Upload preview"
            className="w-full h-64 object-contain rounded-lg bg-gray-100"
          />
          <button
            onClick={handleRemove}
            className="absolute top-2 right-2 bg-red-500 hover:bg-red-600 text-white
                       rounded-full w-8 h-8 flex items-center justify-center text-sm
                       shadow-md transition-colors"
            title="Remove image"
          >
            ✕
          </button>
        </div>
      ) : (
        <div
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onClick={() => inputRef.current?.click()}
          className="border-2 border-dashed border-gray-300 rounded-lg p-8
                     text-center cursor-pointer hover:border-blue-400
                     hover:bg-blue-50/30 transition-colors"
        >
          <svg
            className="mx-auto h-12 w-12 text-gray-400 mb-3"
            fill="none" viewBox="0 0 24 24" stroke="currentColor"
          >
            <path
              strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
              d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 
                 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 
                 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
            />
          </svg>
          <p className="text-sm text-gray-600 font-medium">
            Drop image here or click to browse
          </p>
          <p className="text-xs text-gray-400 mt-1">
            Chest X-ray, Dermoscopy, or Brain MRI (JPG, PNG)
          </p>
        </div>
      )}

      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        onChange={handleFileSelect}
        className="hidden"
      />
    </div>
  )
}
