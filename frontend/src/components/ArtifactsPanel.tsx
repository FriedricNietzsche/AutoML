import React from 'react';

type Artifact = { id: string; url: string; name: string; type: string };

const ArtifactsPanel: React.FC<{ artifacts?: Artifact[]; projectId?: string }> = ({ artifacts = [], projectId }) => {
    return (
        <div className="artifacts-panel">
            <h2>Artifacts</h2>
            <p className="text-xs text-gray-600">Placeholder (project {projectId ?? '—'}).</p>
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