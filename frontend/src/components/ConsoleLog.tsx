import React from 'react';

const ConsoleLog: React.FC<{ logs?: string[]; projectId?: string }> = ({ logs = [], projectId }) => {
    return (
        <div className="console-log">
            <h2 className="text-lg font-bold">Console Log</h2>
            <p className="text-xs text-gray-600">Placeholder (project {projectId ?? '—'}).</p>
            <div className="overflow-y-auto h-64 border border-gray-300 p-2">
                {logs.length === 0 ? (
                    <p>No log messages available.</p>
                ) : (
                    logs.map((log, index) => (
                        <div key={index} className="log-entry">
                            {log}
                        </div>
                    ))
                )}
            </div>
        </div>
    );
};

export default ConsoleLog;