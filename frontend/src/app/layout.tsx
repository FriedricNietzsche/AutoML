import type { Metadata } from 'next';
import './globals.css';
import Providers from './providers';

export const metadata: Metadata = {
    title: 'AutoML Agentic Builder',
    description: 'Agentic workspace for building tabular ML pipelines',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
    return (
        <html lang="en" suppressHydrationWarning>
            <body className="min-h-dvh bg-bg text-text antialiased transition-colors duration-300">
                <Providers>{children}</Providers>
            </body>
        </html>
    );
}