import { useState, ChangeEvent } from 'react';
import { ArrowLeft, Upload, Loader2, CheckCircle, XCircle, AlertTriangle } from 'lucide-react';

// Define the structure for a single analysis result
export interface ImageResult {
  file: File;
  prediction: string; // Can be 'REAL', 'FAKE', or 'Error'
  thumbnail: string;
}

// Define the structure of the API response
interface ApiResponse {
  filename: string;
  prediction: string;
}

// --- Component for displaying a single image result card ---
interface ImageResultCardProps {
  result: ImageResult;
}

function ImageResultCard({ result }: ImageResultCardProps) {
  const predictionUpper = result.prediction.toUpperCase();
  const isReal = predictionUpper === 'REAL';
  const hadError = predictionUpper === 'ERROR';

  return (
    <div className={`bg-gray-900/70 rounded-xl p-4 border-2 transition-all duration-300 ${
      hadError ? 'border-yellow-500/30 hover:border-yellow-500/50' :
      isReal ? 'border-green-500/30 hover:border-green-500/50' : 'border-red-500/30 hover:border-red-500/50'
    }`}>
      <div className="flex items-start space-x-4">
        <div className="flex-shrink-0">
          <img
            src={result.thumbnail}
            alt={result.file.name}
            className="w-24 h-24 object-cover rounded-lg"
          />
        </div>
        <div className="flex-1 min-w-0">
          <h4 className="text-white font-medium text-sm mb-2 truncate" title={result.file.name}>
            {result.file.name}
          </h4>
          <div className="flex items-center space-x-2">
            {hadError ? (
              <>
                <AlertTriangle size={20} className="text-yellow-500 flex-shrink-0" />
                <span className="text-xl font-bold text-yellow-500">ERROR</span>
              </>
            ) : isReal ? (
              <>
                <CheckCircle size={20} className="text-green-500 flex-shrink-0" />
                <span className="text-xl font-bold text-green-500">REAL</span>
              </>
            ) : (
              <>
                <XCircle size={20} className="text-red-500 flex-shrink-0" />
                <span className="text-xl font-bold text-red-500">FAKE</span>
              </>
            )}
          </div>
          <p className="text-gray-500 text-xs mt-2">
            {(result.file.size / 1024).toFixed(1)} KB
          </p>
        </div>
      </div>
    </div>
  );
}


// --- Main Page Component ---
interface ImageAnalysisPageProps {
  onNavigateHome: () => void;
}

export default function ImageAnalysisPage({ onNavigateHome }: ImageAnalysisPageProps) {
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [results, setResults] = useState<ImageResult[]>([]);
  const [error, setError] = useState<string | null>(null);

  const handleFileSelect = (e: ChangeEvent<HTMLInputElement>) => {
    setError(null);
    const files = e.target.files;
    if (files) {
      const imageFiles = Array.from(files).filter(file =>
        ['image/jpeg', 'image/jpg', 'image/png'].includes(file.type)
      );
      setSelectedFiles(imageFiles);
      setResults([]);
    }
  };

  const handleAnalyze = async () => {
    if (selectedFiles.length === 0) return;
    setIsAnalyzing(true);
    setResults([]);
    setError(null);

    const formData = new FormData();
    selectedFiles.forEach(file => {
      formData.append('files', file);
    });

    try {
      const response = await fetch('http://localhost:5000/predict/image', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Server error: ${response.status} ${response.statusText}`);
      }

      const apiResults: ApiResponse[] = await response.json();
      const fileMap = new Map(selectedFiles.map(file => [file.name, file]));

      const formattedResults = apiResults.map(apiResult => {
        const originalFile = fileMap.get(apiResult.filename);
        if (!originalFile) return null;
        return {
          file: originalFile,
          prediction: apiResult.prediction,
          thumbnail: URL.createObjectURL(originalFile),
        };
      }).filter((result): result is ImageResult => result !== null);

      setResults(formattedResults);
    } catch (err) {
      console.error("Error during analysis:", err);
      setError(err instanceof Error ? err.message : "An unknown error occurred.");
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleReset = () => {
    setSelectedFiles([]);
    setResults([]);
    setError(null);
  };

  return (
    <div className="min-h-screen py-12 px-4">
      <div className="max-w-5xl mx-auto">
        <button
          onClick={onNavigateHome}
          className="flex items-center space-x-2 text-gray-300 hover:text-white transition-colors mb-8 group"
        >
          <ArrowLeft size={20} className="group-hover:-translate-x-1 transition-transform" />
          <span>Back to Home</span>
        </button>

        <div className="bg-gray-800/50 backdrop-blur-sm rounded-2xl p-8 shadow-2xl border border-gray-700">
          <h2 className="text-3xl font-bold text-white mb-2">Image Forensics</h2>
          <p className="text-gray-400 mb-8">Trust me, I'll reveal if its REAL or FAKE</p>

          {results.length === 0 ? (
            <>
              <div className="border-2 border-dashed border-gray-600 rounded-xl p-12 text-center hover:border-blue-500 transition-colors cursor-pointer bg-gray-900/30">
                <input
                  type="file"
                  id="image-upload"
                  multiple
                  accept=".jpg,.jpeg,.png"
                  onChange={handleFileSelect}
                  className="hidden"
                />
                <label htmlFor="image-upload" className="cursor-pointer block">
                  <div className="flex flex-col items-center space-y-4">
                    <div className="bg-blue-600/10 p-6 rounded-full">
                      <Upload size={48} className="text-blue-400" />
                    </div>
                    <div>
                      <p className="text-lg font-semibold text-white mb-1">
                        Upload Images
                      </p>
                      <p className="text-sm text-gray-400">
                        Supports: JPG, JPEG, PNG
                      </p>
                    </div>
                  </div>
                </label>
              </div>

              {error && (
                <div className="my-4 p-4 bg-red-900/50 border border-red-500/50 text-red-300 rounded-lg text-center">
                  <p>{error}</p>
                </div>
              )}

              {selectedFiles.length > 0 && (
                <div className="mt-6">
                  <h3 className="text-lg font-semibold text-white mb-3">
                    Selected Files ({selectedFiles.length})
                  </h3>
                  <div className="bg-gray-900/50 rounded-lg p-4 max-h-48 overflow-y-auto">
                    <ul className="space-y-2">
                      {selectedFiles.map((file, index) => (
                        <li key={index} className="text-gray-300 text-sm flex items-center space-x-2">
                          <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                          <span className="truncate" title={file.name}>{file.name}</span>
                          <span className="text-gray-500 flex-shrink-0">
                            ({(file.size / 1024).toFixed(1)} KB)
                          </span>
                        </li>
                      ))}
                    </ul>
                  </div>
                  <button
                    onClick={handleAnalyze}
                    disabled={isAnalyzing}
                    className="mt-6 w-full bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-500 hover:to-blue-600 text-white font-semibold py-4 rounded-xl transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-2 shadow-lg hover:shadow-xl"
                  >
                    {isAnalyzing ? (
                      <>
                        <Loader2 size={20} className="animate-spin" />
                        <span>One sec...and its done</span>
                      </>
                    ) : (
                      <span>Start ({selectedFiles.length} {selectedFiles.length === 1 ? 'file' : 'files'})</span>
                    )}
                  </button>
                </div>
              )}
            </>
          ) : (
            <>
              <div className="mb-6">
                <h3 className="text-2xl font-semibold text-white mb-4">
                  Analysis Results
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 max-h-[600px] overflow-y-auto pr-2">
                  {results.map((result, index) => (
                    <ImageResultCard key={index} result={result} />
                  ))}
                </div>
              </div>
              <button
                onClick={handleReset}
                className="w-full bg-gradient-to-r from-gray-700 to-gray-800 hover:from-gray-600 hover:to-gray-700 text-white font-semibold py-4 rounded-xl transition-all duration-300 shadow-lg hover:shadow-xl"
              >
                Next Round ?
              </button>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

