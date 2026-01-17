import React from 'react';

interface TrainingDashboardProps {
    projectId: string;
}

const TrainingDashboard: React.FC<TrainingDashboardProps> = ({ projectId }) => {
    return (
        <div className="training-dashboard">
            <h2 className="text-xl font-bold">Training Dashboard</h2>
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