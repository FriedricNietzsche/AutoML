import React from 'react';

interface ArtifactsPanelProps {
    projectId: string;
    artifacts?: Array<{ id: string; name: string; type: string; url: string }>;
}

const ArtifactsPanel: React.FC<ArtifactsPanelProps> = ({ projectId, artifacts = [] }) => {
    return (
        <div className="artifacts-panel">
            <h2>Artifacts</h2>
            <ul>
                {artifacts.map((artifact) => (
                    <li key={artifact.id}>
                        <a href={artifact.url} download>
                            {artifact.name} ({artifact.type})
                        </a>
                    </li>
                ))}
            </ul>
        </div>
    );
};

export default ArtifactsPanel;