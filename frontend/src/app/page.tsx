import React, { useEffect, useMemo, useState } from 'react';
import ConsoleLog from '../components/ConsoleLog';
import { createWebSocketClient, isHello } from '../lib/ws';
import { EventMessage } from '../lib/types';

const Page = () => {
  const [status, setStatus] = useState<'idle' | 'connecting' | 'open' | 'closed' | 'error'>('idle');
  const [logs, setLogs] = useState<string[]>([]);
  const [lastEvent, setLastEvent] = useState<EventMessage | null>(null);

  const projectId = useMemo(() => 'demo-project', []);

  useEffect(() => {
    const client = createWebSocketClient({
      projectId,
      onStatusChange: (s) => {
        setStatus(s);
        setLogs((prev) => [`[status] ${s}`, ...prev].slice(0, 200));
      },
      onEvent: (evt) => {
        setLastEvent(evt);
        const tag = isHello(evt) ? 'HELLO' : evt.event.name;
        setLogs((prev) => [`[event:${tag}] ${JSON.stringify(evt.event.payload)}`, ...prev].slice(0, 200));
      },
      onError: (err) => {
        setLogs((prev) => [`[error] ${String(err)}`, ...prev].slice(0, 200));
      },
    });

    return () => client.close();
  }, [projectId]);

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-100 p-6 space-y-6">
      <div className="max-w-3xl w-full bg-white shadow rounded p-6 space-y-4">
        <h1 className="text-3xl font-bold">AutoML Agentic Builder</h1>
        <p className="text-gray-700">
          Live WebSocket handshake to backend for project <span className="font-mono">{projectId}</span>.
        </p>
        <div className="flex items-center space-x-3">
          <span className="text-sm font-semibold">Connection:</span>
          <span
            className={`px-3 py-1 rounded text-sm ${
              status === 'open'
                ? 'bg-green-100 text-green-800'
                : status === 'connecting'
                ? 'bg-yellow-100 text-yellow-800'
                : status === 'error'
                ? 'bg-red-100 text-red-800'
                : 'bg-gray-100 text-gray-800'
            }`}
          >
            {status}
          </span>
        </div>
        <div className="bg-gray-50 border border-gray-200 rounded p-3 text-sm w-full">
          <p className="font-semibold mb-1">Last event:</p>
          {lastEvent ? (
            <pre className="whitespace-pre-wrap break-words text-xs">
              {JSON.stringify(
                {
                  type: lastEvent.type,
                  name: lastEvent.event.name,
                  payload: lastEvent.event.payload,
                  stage: lastEvent.stage,
                  seq: lastEvent.seq,
                },
                null,
                2,
              )}
            </pre>
          ) : (
            <p className="text-gray-600">Waiting for server events...</p>
          )}
        </div>
        <ConsoleLog logs={logs} />
      </div>
    </div>
  );
};

export default Page;
