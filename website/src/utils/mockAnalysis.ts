import { ImageResult } from '../pages/ImageAnalysisPage';

const createThumbnail = (file: File): Promise<string> => {
  return new Promise((resolve) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      resolve(e.target?.result as string);
    };
    reader.readAsDataURL(file);
  });
};

const simulateAnalysis = (duration: number): Promise<void> => {
  return new Promise((resolve) => {
    setTimeout(resolve, duration);
  });
};

export const analyzeImages = async (files: File[]): Promise<ImageResult[]> => {
  await simulateAnalysis(1500 + files.length * 300);

  const results: ImageResult[] = [];

  for (const file of files) {
    const thumbnail = await createThumbnail(file);
    const prediction: 'REAL' | 'FAKE' = Math.random() > 0.5 ? 'REAL' : 'FAKE';

    results.push({
      file,
      prediction,
      thumbnail,
    });
  }

  return results;
};

export const analyzeVideo = async (file: File): Promise<'REAL' | 'FAKE'> => {
  await simulateAnalysis(2500);
  return Math.random() > 0.5 ? 'REAL' : 'FAKE';
};
