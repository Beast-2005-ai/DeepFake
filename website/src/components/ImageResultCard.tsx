import { CheckCircle, XCircle } from 'lucide-react';
import { ImageResult } from '../pages/ImageAnalysisPage';

interface ImageResultCardProps {
  result: ImageResult;
}

export default function ImageResultCard({ result }: ImageResultCardProps) {
  const isReal = result.prediction === 'REAL';

  return (
    <div className={`bg-gray-900/70 rounded-xl p-4 border-2 transition-all duration-300 ${
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
          <h4 className="text-white font-medium text-sm mb-2 truncate">
            {result.file.name}
          </h4>

          <div className="flex items-center space-x-2">
            {isReal ? (
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
