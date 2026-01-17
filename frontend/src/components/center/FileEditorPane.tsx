import Editor, { loader } from '@monaco-editor/react';

loader.init().then((monaco) => {
  monaco.languages.typescript.typescriptDefaults.setDiagnosticsOptions({
    noSemanticValidation: true,
    noSyntaxValidation: true,
  });
  
  monaco.languages.typescript.javascriptDefaults.setDiagnosticsOptions({
    noSemanticValidation: true,
    noSyntaxValidation: true,
  });

  monaco.languages.typescript.typescriptDefaults.setCompilerOptions({
    target: monaco.languages.typescript.ScriptTarget.ES2020,
    allowNonTsExtensions: true,
    moduleResolution: monaco.languages.typescript.ModuleResolutionKind.NodeJs,
    module: monaco.languages.typescript.ModuleKind.CommonJS,
    noEmit: true,
    lib: ["es2020"],
    typeRoots: [],
  });
});

interface FileEditorPaneProps {
  filePath: string;
  content: string;
  language?: string;
  onChange?: (value: string | undefined) => void;
}

export default function FileEditorPane({
  filePath,
  content,
  language = 'typescript',
  onChange,
}: FileEditorPaneProps) {
  
  const getLanguage = (path: string) => {
    if (path.endsWith('.json')) return 'json';
    if (path.endsWith('.ts') || path.endsWith('.tsx')) return 'typescript';
    if (path.endsWith('.js') || path.endsWith('.jsx')) return 'javascript';
    if (path.endsWith('.css')) return 'css';
    if (path.endsWith('.md')) return 'markdown';
    if (path.endsWith('.py')) return 'python';
    if (path.endsWith('.html')) return 'html';
    return 'plaintext';
  };

  const detectedLang = language === 'typescript' ? getLanguage(filePath) : language;

  return (
    <div className="h-full w-full bg-replit-bg/20 backdrop-blur-xl">
      <Editor
        height="100%"
        path={filePath}
        language={detectedLang}
        value={content}
        theme="vs-dark"
        onChange={onChange}
        options={{
          minimap: { enabled: false },
          fontSize: 14,
          wordWrap: 'on',
          padding: { top: 16, bottom: 16 },
          fontFamily: "'JetBrains Mono', 'Fira Code', Consolas, monospace",
          scrollBeyondLastLine: false,
          renderLineHighlight: 'all',
          lineNumbers: 'on',
          glyphMargin: false,
          folding: true,
          lineDecorationsWidth: 10,
          lineNumbersMinChars: 3,
          automaticLayout: true,
          fixedOverflowWidgets: true,
        }}
      />
    </div>
  );
}
