import React from 'react';

const FileExplorer: React.FC<{ projectId?: string }> = ({ projectId }) => {
    return (
        <div className="file-explorer">
            <h2 className="text-lg font-semibold">File Explorer</h2>
            <p className="text-xs text-gray-600">Placeholder (project {projectId ?? '—'}).</p>
            <ul className="file-list">
                {/* Placeholder for file items */}
                <li className="file-item">File 1</li>
                <li className="file-item">File 2</li>
                <li className="file-item">File 3</li>
            </ul>
        </div>
    );
};

export default FileExplorer;