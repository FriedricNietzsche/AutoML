'use client';

import React from 'react';
import Editor from '@monaco-editor/react';

interface MonacoViewerProps {
    code: string;
    onChange: (newCode: string) => void;
}

const MonacoViewer: React.FC<MonacoViewerProps> = ({ code, onChange }) => {
    return (
        <Editor
            height="400px"
            defaultLanguage="javascript"
            theme="vs-dark"
            value={code}
            options={{ minimap: { enabled: false }, automaticLayout: true }}
            onChange={(value) => onChange(value ?? '')}
        />
    );
};

export default MonacoViewer;