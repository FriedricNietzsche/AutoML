import React from 'react';

const ArtifactsPanel = ({ artifacts }) => {
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