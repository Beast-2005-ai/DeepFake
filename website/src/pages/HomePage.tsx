import { ImageIcon, VideoIcon } from 'lucide-react';

interface HomePageProps {
  onNavigateToImage: () => void;
  onNavigateToVideo: () => void;
}

export default function HomePage({ onNavigateToImage, onNavigateToVideo }: HomePageProps) {
  return (
    <div className="min-h-screen flex items-center justify-center px-4">
      <div className="max-w-4xl w-full text-center">
        <div className="mb-12">
          <h1 className="text-6xl font-bold text-white mb-4 tracking-tight">
            Deepfake Detector
          </h1>
          <p className="text-xl text-gray-300 max-w-2xl mx-auto leading-relaxed">
            Separating fact from fiction, Harnessing advanced neural networks to distinguish between authentic and synthetically generated media
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 max-w-3xl mx-auto">
          <button
            onClick={onNavigateToImage}
            className="group relative bg-gradient-to-br from-blue-600 to-blue-700 hover:from-blue-500 hover:to-blue-600 text-white rounded-2xl p-8 transition-all duration-300 transform hover:scale-105 hover:shadow-2xl shadow-lg"
          >
            <div className="flex flex-col items-center space-y-4">
              <div className="bg-white/10 p-6 rounded-full group-hover:bg-white/20 transition-colors duration-300">
                <ImageIcon size={48} className="text-white" />
              </div>
              <span className="text-2xl font-semibold">Image Forensics</span>
              <span className="text-sm text-blue-100 opacity-90">
                Hunt for artifacts in still frames
              </span>
            </div>
          </button>

          <button
            onClick={onNavigateToVideo}
            className="group relative bg-gradient-to-br from-purple-600 to-purple-700 hover:from-purple-500 hover:to-purple-600 text-white rounded-2xl p-8 transition-all duration-300 transform hover:scale-105 hover:shadow-2xl shadow-lg"
          >
            <div className="flex flex-col items-center space-y-4">
              <div className="bg-white/10 p-6 rounded-full group-hover:bg-white/20 transition-colors duration-300">
                <VideoIcon size={48} className="text-white" />
              </div>
              <span className="text-2xl font-semibold">Motion Media</span>
              <span className="text-sm text-purple-100 opacity-90">
                Reveal truths in motion
              </span>
            </div>
          </button>
        </div>
      </div>
    </div>
  );
}
