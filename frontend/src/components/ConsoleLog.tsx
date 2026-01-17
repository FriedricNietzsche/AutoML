import React from 'react';

const ConsoleLog: React.FC<{ projectId: string; logs?: string[] }> = ({ projectId, logs = [] }) => {
    return (
        <div className="console-log">
            <h2 className="text-lg font-bold">Console Log</h2>
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