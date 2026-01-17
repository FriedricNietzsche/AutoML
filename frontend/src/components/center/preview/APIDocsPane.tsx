import { useCallback, useMemo, useState } from 'react';
import { ChevronDown, ChevronRight, Copy, ExternalLink } from 'lucide-react';
import type { FileSystemNode } from '../../../lib/types';
import clsx from 'clsx';
import { motion, AnimatePresence, useReducedMotion } from 'framer-motion';

interface APIDocsPaneProps {
  files: FileSystemNode[];
}

export default function APIDocsPane({ files }: APIDocsPaneProps) {
    const reducedMotion = useReducedMotion();

    const readVfsFile = useCallback((path: string): string | null => {
        const core = (nodes: FileSystemNode[]): string | null => {
            for (const n of nodes) {
                if (n.path === path) return n.content || '';
                if (n.children) {
                    const found = core(n.children);
                    if (found !== null) return found;
                }
            }
            return null;
        };
        return core(files);
    }, [files]);

    const session = useMemo(() => {
        const raw = readVfsFile('/sessions/current.json');
        if (!raw) return null;
        try {
            return JSON.parse(raw) as { id?: string; modelName?: string; goal?: string };
        } catch {
            return null;
        }
    }, [readVfsFile]);

    const modelMeta = useMemo(() => {
        const raw = readVfsFile('/artifacts/model.json');
        if (!raw) return { version: '0.1', model: 'AutoAI MockNet' };
        try {
            const parsed = JSON.parse(raw) as { version?: string; model?: string };
            return { version: parsed.version ?? '0.1', model: parsed.model ?? 'AutoAI MockNet' };
        } catch {
            return { version: '0.1', model: 'AutoAI MockNet' };
        }
    }, [readVfsFile]);

    const schema = useMemo(() => deriveRequestSchema(session?.goal ?? ''), [session?.goal]);

    return (
        <div className="h-full flex flex-col bg-replit-bg overflow-hidden">
            {/* Header */}
            <div className="shrink-0 border-b border-replit-border bg-replit-surface p-5">
                <div className="flex items-start justify-between gap-4">
                    <div>
                        <div className="text-xl font-semibold text-replit-text">Model API</div>
                        <div className="mt-1 text-sm text-replit-textMuted">
                            {session?.modelName ? <span className="font-medium">{session.modelName}</span> : 'Untitled model'}
                            <span className="mx-2">·</span>
                            <span className="px-2 py-0.5 rounded-full border border-replit-border bg-replit-bg text-xs">
                                v{modelMeta.version}
                            </span>
                        </div>
                        <div className="mt-3 text-xs font-mono text-replit-textMuted">
                            Base URL: https://api.autoai.local
                        </div>
                    </div>

                    <a
                        className="text-xs text-replit-textMuted hover:text-replit-text inline-flex items-center gap-1"
                        href="#"
                        onClick={(e) => e.preventDefault()}
                    >
                        openapi.json <ExternalLink className="w-3 h-3" />
                    </a>
                </div>
            </div>

            <div className="flex-1 overflow-y-auto">
                <div className="p-6 w-full">
                    <Endpoint
                        reducedMotion={!!reducedMotion}
                        method="POST"
                        path="/predict"
                        summary="Predict"
                        description="Deterministic local inference (no network)."
                        schema={schema}
                        modelVersion={modelMeta.version}
                    />

                    <div className="h-4" />

                    <Endpoint
                        reducedMotion={!!reducedMotion}
                        method="GET"
                        path="/health"
                        summary="Health"
                        description="Service status."
                        readonly
                    />
                </div>
            </div>
        </div>
    );
}

type HttpMethod = 'GET' | 'POST';

interface RequestSchema {
    textField: { key: string; label: string };
    numericFields: Array<{ key: string; label: string; defaultValue: number }>;
}

interface EndpointProps {
    reducedMotion: boolean;
    method: HttpMethod;
    path: string;
    summary: string;
    description?: string;
    readonly?: boolean;
    schema?: RequestSchema;
    modelVersion?: string;
}

type PredictResponse = {
    predicted_class: 'positive' | 'negative' | 'neutral';
    probabilities: { positive: number; negative: number; neutral: number };
    model_version: string;
    latency_ms: number;
    request_id: string;
};

function deriveRequestSchema(goal: string): RequestSchema {
    const normalized = goal.toLowerCase();
    const candidates: Array<{ key: string; label: string; defaultValue: number; match: RegExp }> = [
        { key: 'age', label: 'age', defaultValue: 34, match: /\bage\b/ },
        { key: 'tenure', label: 'tenure', defaultValue: 12, match: /\btenure\b/ },
        { key: 'income', label: 'income', defaultValue: 65000, match: /\bincome\b|\bsalary\b/ },
        { key: 'balance', label: 'balance', defaultValue: 1200, match: /\bbalance\b/ },
        { key: 'rating', label: 'rating', defaultValue: 4.2, match: /\brating\b|\bscore\b/ },
    ];

    const picked = candidates.filter((c) => c.match.test(normalized)).slice(0, 3);
    const numericFields = (picked.length > 0 ? picked : candidates.slice(0, 2)).map(({ key, label, defaultValue }) => ({
        key,
        label,
        defaultValue,
    }));

    return {
        textField: { key: 'text', label: 'text' },
        numericFields,
    };
}

function stableHash(str: string): number {
    let h = 2166136261;
    for (let i = 0; i < str.length; i += 1) {
        h ^= str.charCodeAt(i);
        h = Math.imul(h, 16777619);
    }
    return Math.abs(h);
}

function localPredict(payload: Record<string, unknown>, modelVersion: string): PredictResponse {
    const text = String(payload.text ?? '');
    const numericKeys = Object.keys(payload).filter((k) => k !== 'text').sort();
    const numericSum = numericKeys.reduce((acc, k) => {
        const v = Number(payload[k]);
        return acc + (Number.isFinite(v) ? v : 0);
    }, 0);

    const h = stableHash(`${text}::${numericSum.toFixed(3)}::${numericKeys.join(',')}`);
    const idx = h % 3;
    const classes: Array<PredictResponse['predicted_class']> = ['positive', 'negative', 'neutral'];

    const p0 = 0.1 + ((h % 31) / 310);
    const p1 = 0.1 + (((h >> 5) % 31) / 310);
    const p2 = 0.1 + (((h >> 10) % 31) / 310);
    const probs = [p0, p1, p2];
    probs[idx] += 0.55;
    const s = probs.reduce((a, b) => a + b, 0);
    const norm = probs.map((p) => p / s);

    return {
        predicted_class: classes[idx],
        probabilities: {
            positive: Number(norm[0].toFixed(4)),
            negative: Number(norm[1].toFixed(4)),
            neutral: Number(norm[2].toFixed(4)),
        },
        model_version: modelVersion,
        latency_ms: 18 + (h % 27),
        request_id: `req_${h.toString(16).slice(0, 8)}`,
    };
}

async function copyText(text: string) {
    try {
        await navigator.clipboard.writeText(text);
    } catch {
        // ignore
    }
}

function Endpoint({ reducedMotion, method, path, summary, description, readonly, schema, modelVersion }: EndpointProps) {
    const [isOpen, setIsOpen] = useState(true);
    const [tryOut, setTryOut] = useState(false);
    const [loading, setLoading] = useState(false);
    const [response, setResponse] = useState<PredictResponse | null>(null);

    const modelVer = modelVersion ?? '0.1';

    const [text, setText] = useState('This product is amazing!');
    const [numbers, setNumbers] = useState<Record<string, number>>(() => {
        const next: Record<string, number> = {};
        for (const f of schema?.numericFields ?? []) next[f.key] = f.defaultValue;
        return next;
    });

    const requestBody = useMemo(() => {
        const payload: Record<string, unknown> = { text };
        for (const f of schema?.numericFields ?? []) payload[f.key] = numbers[f.key];
        return payload;
    }, [text, numbers, schema]);

    const curlSnippet = useMemo(() => {
        const json = JSON.stringify(requestBody);
        return `curl -X ${method} "https://api.autoai.local${path}" -H "Content-Type: application/json" -d '${json}'`;
    }, [method, path, requestBody]);

    const onExecute = () => {
        setLoading(true);
        setResponse(null);

        window.setTimeout(() => {
            if (method === 'POST') {
                setResponse(localPredict(requestBody, modelVer));
            }
            setLoading(false);
        }, 550);
    };

    const methodPill = method === 'POST' ? 'bg-replit-accent/90 text-white' : 'bg-replit-surfaceHover/50 text-replit-text';
    const accentBorder = method === 'POST' ? 'border-replit-accent/40' : 'border-replit-border/60';

    return (
        <div className={clsx('rounded-xl border overflow-hidden', accentBorder)}>
            <button
                type="button"
                onClick={() => setIsOpen((v) => !v)}
                className={clsx(
                    'w-full text-left px-4 py-3 flex items-center justify-between',
                    'bg-replit-surface/35 backdrop-blur hover:bg-replit-surface/45 transition-colors'
                )}
            >
                <div className="flex items-center gap-3 min-w-0">
                    <span className={clsx('px-3 py-1 rounded-md text-xs font-semibold shrink-0', methodPill)}>{method}</span>
                    <span className="font-mono text-xs text-replit-text shrink-0">{path}</span>
                    <span className="text-xs text-replit-textMuted truncate">{summary}</span>
                </div>
                {isOpen ? <ChevronDown className="w-4 h-4 text-replit-textMuted" /> : <ChevronRight className="w-4 h-4 text-replit-textMuted" />}
            </button>

            <AnimatePresence>
                {isOpen && (
                    <motion.div
                        initial={reducedMotion ? { opacity: 1 } : { height: 0, opacity: 0 }}
                        animate={reducedMotion ? { opacity: 1 } : { height: 'auto', opacity: 1 }}
                        exit={reducedMotion ? { opacity: 1 } : { height: 0, opacity: 0 }}
                        transition={{ duration: reducedMotion ? 0 : 0.2 }}
                        className="border-t border-replit-border/60 bg-replit-bg"
                    >
                        <div className="p-4">
                            {description && <p className="text-sm text-replit-textMuted mb-4">{description}</p>}

                            {!readonly && (
                                <div className="flex items-center justify-end mb-4">
                                    {!tryOut ? (
                                        <button
                                            type="button"
                                            onClick={() => setTryOut(true)}
                                            className="px-3 py-1.5 rounded-md border border-replit-border/60 bg-replit-surface/30 hover:bg-replit-surface/45 text-sm text-replit-text"
                                        >
                                            Try it out
                                        </button>
                                    ) : (
                                        <button
                                            type="button"
                                            onClick={() => {
                                                setTryOut(false);
                                                setResponse(null);
                                            }}
                                            className="px-3 py-1.5 rounded-md border border-replit-border/60 bg-replit-surface/30 hover:bg-replit-surface/45 text-sm text-replit-text"
                                        >
                                            Cancel
                                        </button>
                                    )}
                                </div>
                            )}

                            {tryOut && !readonly && schema && (
                                <div className="rounded-xl border border-replit-border/60 bg-replit-surface/25 backdrop-blur p-4 mb-4">
                                    <div className="text-sm font-semibold text-replit-text mb-3">Request Body</div>

                                    <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                                        <div className="md:col-span-2">
                                            <label className="text-xs font-mono text-replit-textMuted">{schema.textField.label} *</label>
                                            <textarea
                                                value={text}
                                                onChange={(e) => setText(e.target.value)}
                                                className="mt-1 w-full h-24 p-2 rounded-md border border-replit-border/60 bg-replit-bg/20 text-replit-text font-mono text-xs outline-none focus:ring-2 focus:ring-replit-accent/40"
                                            />
                                        </div>

                                        {schema.numericFields.map((f) => (
                                            <div key={f.key}>
                                                <label className="text-xs font-mono text-replit-textMuted">{f.label} *</label>
                                                <input
                                                    type="number"
                                                    value={numbers[f.key] ?? f.defaultValue}
                                                    onChange={(e) =>
                                                        setNumbers((prev) => ({
                                                            ...prev,
                                                            [f.key]: Number(e.target.value),
                                                        }))
                                                    }
                                                    className="mt-1 w-full p-2 rounded-md border border-replit-border/60 bg-replit-bg/20 text-replit-text font-mono text-xs outline-none focus:ring-2 focus:ring-replit-accent/40"
                                                />
                                            </div>
                                        ))}
                                    </div>

                                    <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-3">
                                        <div className="rounded-lg border border-replit-border/60 bg-replit-bg/20 p-3">
                                            <div className="text-xs text-replit-textMuted mb-2 flex items-center justify-between">
                                                <span>JSON</span>
                                                <button
                                                    type="button"
                                                    onClick={() => copyText(JSON.stringify(requestBody, null, 2))}
                                                    className="text-replit-textMuted hover:text-replit-text"
                                                    title="Copy JSON"
                                                >
                                                    <Copy className="w-4 h-4" />
                                                </button>
                                            </div>
                                            <pre className="text-xs font-mono text-replit-text overflow-x-auto whitespace-pre-wrap break-words">
                                                {JSON.stringify(requestBody, null, 2)}
                                            </pre>
                                        </div>

                                        <div className="rounded-lg border border-replit-border/60 bg-replit-bg/20 p-3">
                                            <div className="text-xs text-replit-textMuted mb-2 flex items-center justify-between">
                                                <span>cURL</span>
                                                <button
                                                    type="button"
                                                    onClick={() => copyText(curlSnippet)}
                                                    className="text-replit-textMuted hover:text-replit-text"
                                                    title="Copy cURL"
                                                >
                                                    <Copy className="w-4 h-4" />
                                                </button>
                                            </div>
                                            <pre className="text-xs font-mono text-replit-text overflow-x-auto whitespace-pre-wrap break-words">{curlSnippet}</pre>
                                        </div>
                                    </div>

                                    <div className="mt-4">
                                        <button
                                            type="button"
                                            onClick={onExecute}
                                            disabled={loading}
                                            className={clsx(
                                                'w-full px-4 py-2 rounded-md font-semibold text-sm',
                                                'bg-replit-accent/90 hover:bg-replit-accent text-white transition-colors',
                                                loading && 'opacity-80 cursor-not-allowed'
                                            )}
                                        >
                                            {loading ? 'Executing…' : 'Execute'}
                                        </button>
                                    </div>
                                </div>
                            )}

                            {response && (
                                <div className="mt-5">
                                    <div className="flex items-center justify-between mb-2">
                                        <div className="text-sm font-semibold text-replit-text">Response</div>
                                        <div className="text-xs text-replit-textMuted">Latency: {response.latency_ms}ms</div>
                                    </div>

                                    <div className="rounded-xl border border-replit-border/60 bg-replit-surface/25 backdrop-blur p-3 relative group">
                                        <button
                                            type="button"
                                            className="absolute top-3 right-3 text-replit-textMuted hover:text-replit-text opacity-0 group-hover:opacity-100 transition-opacity"
                                            onClick={() => copyText(JSON.stringify(response, null, 2))}
                                            title="Copy response"
                                        >
                                            <Copy className="w-4 h-4" />
                                        </button>
                                        <div className="text-xs font-mono text-replit-textMuted mb-2">Code: 200</div>
                                        <pre className="text-xs font-mono text-replit-text overflow-x-auto whitespace-pre-wrap break-words">
                                            {JSON.stringify(response, null, 2)}
                                        </pre>
                                    </div>
                                </div>
                            )}

                            {readonly && method === 'GET' && (
                                <div className="rounded-xl border border-replit-border/60 bg-replit-surface/25 backdrop-blur p-4">
                                    <div className="text-sm font-semibold text-replit-text mb-2">Response</div>
                                    <pre className="text-xs font-mono text-replit-text">{JSON.stringify({ status: 'ok' }, null, 2)}</pre>
                                </div>
                            )}
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
}
