import React from 'react';

const ProfilingPanel: React.FC<{ profilingData: any }> = ({ profilingData }) => {
    return (
        <div className="profiling-panel">
            <h2 className="text-xl font-bold">Data Profiling</h2>
            {profilingData ? (
                <div>
                    <h3 className="text-lg">Summary Statistics</h3>
                    <ul>
                        {Object.entries(profilingData.summary).map(([key, value]) => (
                            <li key={key}>
                                <strong>{key}:</strong> {value}
                            </li>
                        ))}
                    </ul>
                    <h3 className="text-lg">Data Types</h3>
                    <ul>
                        {Object.entries(profilingData.dataTypes).map(([column, type]) => (
                            <li key={column}>
                                <strong>{column}:</strong> {type}
                            </li>
                        ))}
                    </ul>
                </div>
            ) : (
                <p>No profiling data available.</p>
            )}
        </div>
    );
};

export default ProfilingPanel;