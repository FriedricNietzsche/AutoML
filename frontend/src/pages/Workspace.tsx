import { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useProjectStore } from '../stores/projectStore';
import { TrainingLoader } from '../components/TrainingLoader';

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000';

export function Workspace() {
  const navigate = useNavigate();
  const { projectId, prompt, taskType, datasetInfo } = useProjectStore();

  // Redirect if no prompt
  useEffect(() => {
    if (!prompt) {
      navigate('/');
    }
  }, [prompt, navigate]);

  // Kick off the pipeline when workspace loads
  useEffect(() => {
    if (!prompt || !projectId) return;

    const startPipeline = async () => {
      try {
        // Call parse endpoint to start the flow
        const res = await fetch(`${API_BASE}/api/parse`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            project_id: projectId,
            prompt: prompt,
          }),
        });

        if (!res.ok) {
          console.error('Failed to start pipeline:', await res.text());
        }
      } catch (err) {
        console.error('Error starting pipeline:', err);
      }
    };

    startPipeline();
  }, [projectId, prompt]);

  if (!prompt) {
    return null; // Will redirect
  }

  return (
    <div className="min-h-screen bg-gray-950 p-6">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="mb-6">
          <button
            onClick={() => navigate('/')}
            className="text-gray-400 hover:text-white text-sm mb-4 flex items-center gap-1"
          >
            ← Back
          </button>
          <h1 className="text-2xl font-bold text-white mb-2">ML Pipeline</h1>
          <p className="text-gray-400 text-sm">{prompt}</p>
        </div>

        {/* Main content - Training Loader */}
        <TrainingLoader />

        {/* Dataset info panel */}
        {datasetInfo && (
          <div className="mt-6 bg-gray-900 rounded-lg p-4">
            <h3 className="text-white font-medium mb-2">Dataset</h3>
            <div className="grid grid-cols-3 gap-4 text-sm">
              <div>
                <span className="text-gray-500">Name:</span>
                <span className="text-white ml-2">{datasetInfo.name}</span>
              </div>
              <div>
                <span className="text-gray-500">Rows:</span>
                <span className="text-white ml-2">{datasetInfo.rows.toLocaleString()}</span>
              </div>
              <div>
                <span className="text-gray-500">Columns:</span>
                <span className="text-white ml-2">{datasetInfo.cols}</span>
              </div>
            </div>
          </div>
        )}

        {/* Task type badge */}
        {taskType && (
          <div className="mt-4">
            <span className="inline-block px-3 py-1 bg-blue-900 text-blue-300 text-sm rounded-full">
              {taskType}
            </span>
          </div>
        )}
      </div>
    </div>
  );
}
