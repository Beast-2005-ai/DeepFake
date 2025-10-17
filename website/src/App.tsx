import { useState } from 'react';
import HomePage from './pages/HomePage';
import ImageAnalysisPage from './pages/ImageAnalysisPage';
import VideoAnalysisPage from './pages/VideoAnalysisPage';

type Page = 'home' | 'image' | 'video';

function App() {
  const [currentPage, setCurrentPage] = useState<Page>('home');

  const navigateToHome = () => setCurrentPage('home');
  const navigateToImage = () => setCurrentPage('image');
  const navigateToVideo = () => setCurrentPage('video');

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900">
      {currentPage === 'home' && (
        <HomePage
          onNavigateToImage={navigateToImage}
          onNavigateToVideo={navigateToVideo}
        />
      )}
      {currentPage === 'image' && (
        <ImageAnalysisPage onNavigateHome={navigateToHome} />
      )}
      {currentPage === 'video' && (
        <VideoAnalysisPage onNavigateHome={navigateToHome} />
      )}
    </div>
  );
}

export default App;
