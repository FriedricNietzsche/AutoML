import { Component, type ReactNode } from 'react';

interface Props {
    children: ReactNode;
}

interface State {
    hasError: boolean;
    error?: Error;
}

export default class ErrorBoundary extends Component<Props, State> {
    constructor(props: Props) {
        super(props);
        this.state = { hasError: false };
    }

    static getDerivedStateFromError(error: Error): State {
        return { hasError: true, error };
    }

    componentDidCatch(error: Error, errorInfo: any) {
        console.error('Error boundary caught:', error, errorInfo);
    }

    render() {
        if (this.state.hasError) {
            return (
                <div className="flex items-center justify-center h-full w-full bg-replit-bg/30 backdrop-blur-xl">
                    <div className="max-w-md p-8 bg-replit-surface/60 rounded-lg border border-replit-border/70">
                        <h2 className="text-xl font-bold text-red-500 mb-4">Something went wrong</h2>
                        <p className="text-replit-text mb-4">
                            The component encountered an error. This might be due to missing data or initialization issues.
                        </p>
                        <button
                            onClick={() => window.location.reload()}
                            className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 transition-colors"
                        >
                            Reload Page
                        </button>
                        {this.state.error && (
                            <details className="mt-4 text-xs text-replit-textMuted">
                                <summary className="cursor-pointer">Error Details</summary>
                                <pre className="mt-2 p-2 bg-replit-bg rounded overflow-auto">
                                    {this.state.error.toString()}
                                </pre>
                            </details>
                        )}
                    </div>
                </div>
            );
        }

        return this.props.children;
    }
}
