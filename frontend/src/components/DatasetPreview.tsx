import React from 'react';

interface DatasetPreviewProps {
    projectId?: string;
}

const DatasetPreview: React.FC<DatasetPreviewProps> = ({ projectId }) => {
    return (
        <div className="p-4 border rounded shadow">
            <h2 className="text-xl font-bold">Dataset Preview</h2>
            <p className="text-gray-600">Placeholder (project {projectId ?? '—'}). Dataset sample wiring happens in Task 3.2.</p>
        </div>
    );
};

export default DatasetPreview;