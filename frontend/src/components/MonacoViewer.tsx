import React from 'react';
import { MonacoEditor } from 'react-monaco-editor';

interface MonacoViewerProps {
    code: string;
    onChange: (newCode: string) => void;
}

const MonacoViewer: React.FC<MonacoViewerProps> = ({ code, onChange }) => {
    const editorOptions = {
        selectOnLineNumbers: true,
        automaticLayout: true,
    };

    return (
        <MonacoEditor
            width="100%"
            height="400"
            language="javascript"
            theme="vs-dark"
            value={code}
            options={editorOptions}
            onChange={onChange}
        />
    );
};

export default MonacoViewer;