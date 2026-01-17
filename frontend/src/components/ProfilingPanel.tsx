import React from 'react';

const ProfilingPanel: React.FC<{ projectId?: string }> = ({ projectId }) => {
    return (
        <div className="profiling-panel">
            <h2 className="text-xl font-bold">Data Profiling</h2>
            <p className="text-gray-600">Placeholder (project {projectId ?? '—'}). Profiling events/assets wiring happens in Task 4.1.</p>
        </div>
    );
};

export default ProfilingPanel;