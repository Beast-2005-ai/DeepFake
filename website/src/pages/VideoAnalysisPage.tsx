import { useState, ChangeEvent } from 'react';
import { ArrowLeft, Upload, Loader2, CheckCircle, XCircle, AlertTriangle, Video } from 'lucide-react';

// Define the structure for a single video analysis result
interface VideoResult {
  filename: string;
  prediction: string; // 'REAL', 'FAKE', or 'Error'
}

// --- Component for displaying a single video result ---
interface VideoResultCardProps {
  result: VideoResult;
}
function VideoResultCard({ result }: VideoResultCardProps) {
  const predictionUpper = result.prediction.toUpperCase();
  const isReal = predictionUpper === 'REAL';
  const hadError = predictionUpper === 'ERROR';

  return (
    <div className={`bg-gray-900/70 rounded-xl p-4 border-2 transition-all duration-300 ${
      hadError ? 'border-yellow-500/30' :
      isReal ? 'border-green-500/30' : 'border-red-500/30'
    }`}>
      <div className="flex items-center space-x-4">
        <Video size={24} className={
            hadError ? 'text-yellow-500' : isReal ? 'text-green-500' : 'text-red-500'
        } />
        <div className="flex-1 min-w-0">
          <h4 className="text-white font-medium text-sm truncate" title={result.filename}>
            {result.filename}
          </h4>
        </div>
        <div className="flex items-center space-x-2 flex-shrink-0">
          {hadError ? (
            <>
              <AlertTriangle size={20} className="text-yellow-500" />
              <span className="text-lg font-bold text-yellow-500">ERROR</span>
            </>
          ) : isReal ? (
            <>
              <CheckCircle size={20} className="text-green-500" />
              <span className="text-lg font-bold text-green-500">REAL</span>
            </>
          ) : (
            <>
              <XCircle size={20} className="text-red-500" />
              <span className="text-lg font-bold text-red-500">FAKE</span>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

// --- Main Page Component ---
interface VideoAnalysisPageProps {
  onNavigateHome: () => void;
}

export default function VideoAnalysisPage({ onNavigateHome }: VideoAnalysisPageProps) {
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [results, setResults] = useState<VideoResult[]>([]);
  const [error, setError] = useState<string | null>(null);

  const handleFileSelect = (e: ChangeEvent<HTMLInputElement>) => {
    setError(null);
    const files = e.target.files;
    if (files) {
      const videoFiles = Array.from(files).filter(file => file.type === 'video/mp4');
      setSelectedFiles(videoFiles);
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
      const response = await fetch('http://localhost:5000/predict/video', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Server error: ${response.status} ${response.statusText}`);
      }

      const apiResults: VideoResult[] = await response.json();
      setResults(apiResults);

    } catch (err) {
      console.error("Error during video analysis:", err);
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
      <div className="max-w-4xl mx-auto">
        <button
          onClick={onNavigateHome}
          className="flex items-center space-x-2 text-gray-300 hover:text-white transition-colors mb-8 group"
        >
          <ArrowLeft size={20} className="group-hover:-translate-x-1 transition-transform" />
          <span>Back to Home</span>
        </button>

        <div className="bg-gray-800/50 backdrop-blur-sm rounded-2xl p-8 shadow-2xl border border-gray-700">
          <h2 className="text-3xl font-bold text-white mb-2">Motion Media</h2>
          <p className="text-gray-400 mb-8">I'll check for their every movement</p>

          {results.length === 0 ? (
            <>
              <div className="border-2 border-dashed border-gray-600 rounded-xl p-12 text-center hover:border-purple-500 transition-colors cursor-pointer bg-gray-900/30">
                <input
                  type="file"
                  id="video-upload"
                  multiple
                  accept=".mp4"
                  onChange={handleFileSelect}
                  className="hidden"
                />
                <label htmlFor="video-upload" className="cursor-pointer block">
                  <div className="flex flex-col items-center space-y-4">
                    <div className="bg-purple-600/10 p-6 rounded-full">
                      <Upload size={48} className="text-purple-400" />
                    </div>
                    <div>
                      <p className="text-lg font-semibold text-white mb-1">
                        Upload Videos
                      </p>
                      <p className="text-sm text-gray-400">
                        Supports: MP4
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
                    Selected Videos ({selectedFiles.length})
                  </h3>
                  <div className="bg-gray-900/50 rounded-lg p-4 max-h-48 overflow-y-auto">
                    <ul className="space-y-2">
                      {selectedFiles.map((file, index) => (
                        <li key={index} className="text-gray-300 text-sm flex items-center space-x-2">
                          <div className="w-2 h-2 bg-purple-500 rounded-full"></div>
                          <span className="truncate" title={file.name}>{file.name}</span>
                          <span className="text-gray-500 flex-shrink-0">
                            ({(file.size / (1024 * 1024)).toFixed(2)} MB)
                          </span>
                        </li>
                      ))}
                    </ul>
                  </div>
                  <button
                    onClick={handleAnalyze}
                    disabled={isAnalyzing}
                    className="mt-6 w-full bg-gradient-to-r from-purple-600 to-purple-700 hover:from-purple-500 hover:to-purple-600 text-white font-semibold py-4 rounded-xl transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-2 shadow-lg hover:shadow-xl"
                  >
                    {isAnalyzing ? (
                      <>
                        <Loader2 size={20} className="animate-spin" />
                        <span>Its taking a while...lets see</span>
                      </>
                    ) : (
                      <span>Start ({selectedFiles.length} {selectedFiles.length === 1 ? 'video' : 'videos'})</span>
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
                <div className="space-y-3 max-h-[600px] overflow-y-auto pr-2">
                  {results.map((result, index) => (
                    <VideoResultCard key={index} result={result} />
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

