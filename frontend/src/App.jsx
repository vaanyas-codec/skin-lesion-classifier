import { useState, useCallback } from 'react'
import { Upload, AlertCircle, Loader2, Activity } from 'lucide-react'
import './App.css'

const API_URL = 'https://skin-lesion-classifier-production.up.railway.app/api/predict'

// Map class codes to a rough "risk tier" for visual emphasis only.
// This is NOT a medical judgment - just a UI cue based on common dermatology categorization.
const HIGH_CONCERN_CLASSES = new Set(['mel', 'bcc', 'akiec'])

function App() {
  const [selectedFile, setSelectedFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [isDragging, setIsDragging] = useState(false)

  const resetState = () => {
    setResult(null)
    setError(null)
  }

  const handleFileSelect = (file) => {
    if (!file) return
    if (!['image/jpeg', 'image/png', 'image/jpg'].includes(file.type)) {
      setError('Please upload a JPEG or PNG image.')
      return
    }
    resetState()
    setSelectedFile(file)
    setPreviewUrl(URL.createObjectURL(file))
  }

  const handleDrop = useCallback((e) => {
    e.preventDefault()
    setIsDragging(false)
    const file = e.dataTransfer.files[0]
    handleFileSelect(file)
  }, [])

  const handleDragOver = (e) => {
    e.preventDefault()
    setIsDragging(true)
  }

  const handleDragLeave = () => setIsDragging(false)

  const handleSubmit = async () => {
    if (!selectedFile) return
    setLoading(true)
    setError(null)
    setResult(null)

    const formData = new FormData()
    formData.append('file', selectedFile)

    try {
      const response = await fetch(API_URL, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        const errBody = await response.json().catch(() => null)
        throw new Error(errBody?.detail || `Request failed with status ${response.status}`)
      }

      const data = await response.json()
      setResult(data)
    } catch (err) {
      setError(err.message || 'Something went wrong while contacting the server.')
    } finally {
      setLoading(false)
    }
  }

  const handleReset = () => {
    setSelectedFile(null)
    setPreviewUrl(null)
    resetState()
  }

  const sortedProbabilities = result
    ? Object.entries(result.all_probabilities).sort((a, b) => b[1] - a[1])
    : []

  return (
    <div className="app">
      <header className="header">
        <Activity size={28} strokeWidth={2.2} />
        <div>
          <h1>Skin Lesion Classifier</h1>
          <p className="subtitle">Research demo using a fine-tuned EfficientNet-B0 — not a diagnostic tool</p>
        </div>
      </header>

      <main className="main">
        {!previewUrl && (
          <div
            className={`dropzone ${isDragging ? 'dragging' : ''}`}
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onClick={() => document.getElementById('file-input').click()}
          >
            <Upload size={36} strokeWidth={1.5} />
            <p><strong>Click to upload</strong> or drag and drop</p>
            <p className="dropzone-hint">JPEG or PNG, dermatoscope-style image preferred</p>
            <input
              id="file-input"
              type="file"
              accept="image/jpeg,image/png,image/jpg"
              onChange={(e) => handleFileSelect(e.target.files[0])}
              hidden
            />
          </div>
        )}

        {previewUrl && (
          <div className="preview-section">
            <div className="image-row">
              <div className="image-card">
                <span className="image-label">Uploaded image</span>
                <img src={previewUrl} alt="Uploaded lesion" />
              </div>


            </div>

            <div className="actions">
              {!result && (
                <button className="btn-primary" onClick={handleSubmit} disabled={loading}>
                  {loading ? (
                    <>
                      <Loader2 size={18} className="spin" /> Analyzing...
                    </>
                  ) : (
                    'Analyze image'
                  )}
                </button>
              )}
              <button className="btn-secondary" onClick={handleReset}>
                Upload a different image
              </button>
            </div>
          </div>
        )}

        {error && (
          <div className="error-banner">
            <AlertCircle size={18} />
            <span>{error}</span>
          </div>
        )}

        {result && (
          <div className="results">
            <div className={`result-headline ${HIGH_CONCERN_CLASSES.has(result.predicted_class) ? 'concern' : 'benign'}`}>
              <span className="result-label">Predicted</span>
              <h2>{result.predicted_class_name}</h2>
              <span className="confidence">{(result.confidence * 100).toFixed(1)}% confidence</span>
            </div>

            <div className="probability-list">
              <h3>Full probability breakdown</h3>
              {sortedProbabilities.map(([classCode, prob]) => (
                <div key={classCode} className="probability-row">
                  <span className="prob-label">{classCode}</span>
                  <div className="prob-bar-track">
                    <div
                      className="prob-bar-fill"
                      style={{ width: `${prob * 100}%` }}
                    />
                  </div>
                  <span className="prob-value">{(prob * 100).toFixed(1)}%</span>
                </div>
              ))}
            </div>

            <div className="disclaimer">
              <AlertCircle size={16} />
              <p>
                This is a research/portfolio demo trained on the HAM10000 dataset.
                It is not a substitute for professional medical evaluation.
                Please consult a dermatologist for any skin concern.
              </p>
            </div>
          </div>
        )}
      </main>
    </div>
  )
}

export default App