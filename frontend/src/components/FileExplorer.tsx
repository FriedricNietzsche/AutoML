import React from 'react';

interface FileExplorerProps {
    projectId: string;
}

const FileExplorer: React.FC<FileExplorerProps> = ({ projectId }) => {
    return (
        <div className="file-explorer">
            <h2 className="text-lg font-semibold">File Explorer</h2>
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