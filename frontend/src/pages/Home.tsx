import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useProjectStore } from '../stores/projectStore';

export function Home() {
  const [prompt, setPrompt] = useState('');
  const navigate = useNavigate();
  const { generateNewSession, setPrompt: storePrompt } = useProjectStore();

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!prompt.trim()) return;

    // Generate a new session ID for this project
    const sessionId = generateNewSession();
    storePrompt(prompt.trim());
    
    console.log('[Home] Starting new session:', sessionId, 'with prompt:', prompt);

    // Navigate to workspace - the session ID is now in the store
    navigate('/workspace');
  };

  const handleDemoClick = (demoPrompt: string) => {
    setPrompt(demoPrompt);
  };

  return (
    <div className="min-h-screen bg-gray-950 flex flex-col items-center justify-center p-8">
      <div className="max-w-2xl w-full">
        <h1 className="text-4xl font-bold text-white mb-2 text-center">
          AutoML Studio
        </h1>
        <p className="text-gray-400 text-center mb-8">
          Describe your ML task and we'll build the pipeline
        </p>

        <form onSubmit={handleSubmit} className="space-y-4">
          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="e.g., Build a classifier to predict customer churn using the telecom dataset..."
            className="w-full h-32 bg-gray-900 border border-gray-700 rounded-lg p-4 text-white placeholder-gray-500 focus:outline-none focus:border-blue-500 resize-none"
          />
          
          <button
            type="submit"
            disabled={!prompt.trim()}
            className="w-full py-3 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-700 disabled:cursor-not-allowed text-white font-medium rounded-lg transition-colors"
          >
            Start Building
          </button>
        </form>

        {/* Demo shortcuts */}
        <div className="mt-8">
          <p className="text-gray-500 text-sm mb-3">Quick demos:</p>
          <div className="flex flex-wrap gap-2">
            <DemoButton
              onClick={() => handleDemoClick('Classify images of cats and dogs from Hugging Face dataset Matias12f/cats_and_dogs')}
            >
              🐱 Cats vs Dogs
            </DemoButton>
            <DemoButton
              onClick={() => handleDemoClick('Predict insurance charges based on age, bmi, and smoking status')}
            >
              💰 Insurance
            </DemoButton>
            <DemoButton
              onClick={() => handleDemoClick('Build a churn prediction model for telecom customers')}
            >
              📉 Churn
            </DemoButton>
          </div>
        </div>
      </div>
    </div>
  );
}

function DemoButton({ children, onClick }: { children: React.ReactNode; onClick: () => void }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="px-3 py-1.5 bg-gray-800 hover:bg-gray-700 text-gray-300 text-sm rounded-md transition-colors"
    >
      {children}
    </button>
  );
}
