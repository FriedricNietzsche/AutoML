import React from 'react';

const TrainingDashboard: React.FC<{ projectId?: string }> = ({ projectId }) => {
    return (
        <div className="training-dashboard">
            <h2 className="text-xl font-bold">Training Dashboard</h2>
            <p className="text-xs text-gray-600">Placeholder (project {projectId ?? '—'}). Real charts wiring happens in Task 5.2.</p>
            <div className="training-metrics">
                {/* Placeholder for training metrics visualization */}
                <p>Training progress will be displayed here.</p>
            </div>
            <div className="training-logs">
                {/* Placeholder for training logs */}
                <p>Training logs will be displayed here.</p>
            </div>
        </div>
    );
};

export default TrainingDashboard;