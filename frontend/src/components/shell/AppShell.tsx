import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import TopBar from './TopBar';
import ResizablePanel from './ResizablePanel';
import AIBuilderPanel from '../left/AIBuilderPanel';
import FilesPanel from '../right/FilesPanel';
import WorkspaceTabs, { type Tab } from '../center/WorkspaceTabs';
import DashboardPane from '../center/DashboardPane';
import PreviewPane from '../center/PreviewPane';
import ConsolePane from '../center/ConsolePane';
import PublishingPane from '../center/PublishingPane';
import FileEditorPane from '../center/FileEditorPane';
import QuickFileSwitcher from '../modals/QuickFileSwitcher';
import { useLocalStorageState } from '../../lib/useLocalStorageState';
import { useResizablePanels } from '../../lib/useResizablePanels';
import { useKeyboardShortcuts } from '../../lib/useKeyboardShortcuts';
import { initialFileSystem } from '../../lib/mockData';
import type { FileSystemNode } from '../../lib/types';
import { resolveHttpBase, joinUrl } from '../../lib/api';
import { usePipelineRunner } from '../../lib/usePipelineRunner';
import { useRouter } from '../../router/router';
import type { BuildSession, ChatMessage } from '../../lib/buildSession';
import { useTheme } from '../../lib/theme';
import { FolderOpen, PanelLeftOpen, PanelRightOpen } from 'lucide-react';
import { useProjectStore } from '../../store/projectStore';

const nowTs = () => Date.now();

const normalizePath = (path: string) => {
  if (!path.startsWith('/')) return `/${path}`;
  return path;
};

const makeId = (prefix: string) => `${prefix}_${Math.random().toString(16).slice(2)}_${Date.now()}`;

const upsertFileInTree = (
  tree: FileSystemNode[],
  filePathRaw: string,
  nextContent: string | ((prev: string) => string)
): FileSystemNode[] => {
  const filePath = normalizePath(filePathRaw);
  const parts = filePath.split('/').filter(Boolean);
  if (parts.length === 0) return tree;

  // Find root folder at '/'
  const rootIndex = tree.findIndex((n) => n.type === 'folder' && n.path === '/');
  const root: FileSystemNode =
    rootIndex >= 0
      ? tree[rootIndex]
      : {
          id: 'root',
          name: 'Files',
          type: 'folder',
          path: '/',
          isOpen: true,
          updatedAt: nowTs(),
          children: tree,
        };

  const upsertUnderFolder = (folder: FileSystemNode, idx: number): FileSystemNode => {
    const isLeaf = idx === parts.length - 1;
    const seg = parts[idx];
    const children = folder.children ? [...folder.children] : [];

    if (isLeaf) {
      const targetPath = `${folder.path === '/' ? '' : folder.path}/${seg}`;
      const existingIndex = children.findIndex((c) => c.type === 'file' && c.path === targetPath);
      const prevContent = existingIndex >= 0 ? children[existingIndex].content || '' : '';
      const content = typeof nextContent === 'function' ? nextContent(prevContent) : nextContent;
      const fileNode: FileSystemNode = {
        id: existingIndex >= 0 ? children[existingIndex].id : makeId('file'),
        name: seg,
        type: 'file',
        path: targetPath,
        content,
        updatedAt: nowTs(),
      };
      if (existingIndex >= 0) {
        children[existingIndex] = fileNode;
      } else {
        children.push(fileNode);
      }
      return { ...folder, children, updatedAt: nowTs(), isOpen: folder.isOpen ?? true };
    }

    // Folder
    const folderPath = `${folder.path === '/' ? '' : folder.path}/${seg}`;
    const existingFolderIndex = children.findIndex((c) => c.type === 'folder' && c.path === folderPath);
    const nextFolder: FileSystemNode =
      existingFolderIndex >= 0
        ? children[existingFolderIndex]
        : {
            id: makeId('dir'),
            name: seg,
            type: 'folder',
            path: folderPath,
            isOpen: true,
            updatedAt: nowTs(),
            children: [],
          };

    const updatedFolder = upsertUnderFolder(nextFolder, idx + 1);
    if (existingFolderIndex >= 0) {
      children[existingFolderIndex] = updatedFolder;
    } else {
      children.push(updatedFolder);
    }
    return { ...folder, children, updatedAt: nowTs(), isOpen: folder.isOpen ?? true };
  };

  const updatedRoot = upsertUnderFolder(root, 0);
  if (rootIndex >= 0) {
    const next = [...tree];
    next[rootIndex] = updatedRoot;
    return next;
  }
  // If the original tree had no explicit root, wrap it.
  return [updatedRoot];
};

export default function AppShell() {
  const { navigate } = useRouter();
  const { theme, toggleTheme } = useTheme();

  const [session, setSession] = useLocalStorageState<BuildSession | null>('autoai.buildSession.current', null);

  // Panel state
  const [leftCollapsed, setLeftCollapsed] = useLocalStorageState('leftPanelCollapsed', false);
  const [rightCollapsed, setRightCollapsed] = useLocalStorageState('rightPanelCollapsed', false);
  
  // Resizable panels
  const { sizes, handlePointerDown, containerRef } = useResizablePanels({
    left: 360,
    right: 300,
  });

  // VFS State (persisted)
  const [files, setFiles] = useLocalStorageState<FileSystemNode[]>('vfs.files', initialFileSystem, { defer: true });

  // Backend WebSocket connection (Task 1.4)
  const projectId = 'demo-project';
  const wsBase =
    (typeof import.meta !== 'undefined' && (import.meta as any).env?.VITE_WS_BASE) ||
    (typeof import.meta !== 'undefined' && (import.meta as any).env?.VITE_BACKEND_WS_BASE) ||
    undefined;
  const apiBase = useMemo(() => resolveHttpBase(wsBase), [wsBase]);

  const { connectionStatus, lastEvent, connect: connectProject, hydrate } = useProjectStore((state) => ({
    connectionStatus: state.connectionStatus,
    lastEvent: state.lastEvent,
    connect: state.connect,
    hydrate: state.hydrate,
  }));

  useEffect(() => {
    connectProject({ projectId, wsBase });
    hydrate();
  }, [connectProject, hydrate, projectId, wsBase]);

  // (VFS helpers are hoisted outside the component to keep updateFileContent stable)
  
  // Helper to update file content deep in the tree (stable reference!)
  const updateFileContent = useCallback(
    (path: string, content: string | ((prev: string) => string)) => {
      setFiles((prevFiles) => {
        const upsert =
          typeof upsertFileInTree === 'function'
            ? upsertFileInTree
            : (
                tree: FileSystemNode[],
                filePathRaw: string,
                nextContent: string | ((prev: string) => string)
              ): FileSystemNode[] => {
                const filePath = normalizePath(filePathRaw);
                const parts = filePath.split('/').filter(Boolean);
                if (parts.length === 0) return tree;

                const rootIndex = tree.findIndex((n) => n.type === 'folder' && n.path === '/');
                const root: FileSystemNode =
                  rootIndex >= 0
                    ? tree[rootIndex]
                    : {
                        id: 'root',
                        name: 'Files',
                        type: 'folder',
                        path: '/',
                        isOpen: true,
                        updatedAt: nowTs(),
                        children: tree,
                      };

                const upsertUnderFolder = (folder: FileSystemNode, idx: number): FileSystemNode => {
                  const isLeaf = idx === parts.length - 1;
                  const seg = parts[idx];
                  const children = folder.children ? [...folder.children] : [];

                  if (isLeaf) {
                    const targetPath = `${folder.path === '/' ? '' : folder.path}/${seg}`;
                    const existingIndex = children.findIndex((c) => c.type === 'file' && c.path === targetPath);
                    const prevContent = existingIndex >= 0 ? children[existingIndex].content || '' : '';
                    const resolved = typeof nextContent === 'function' ? nextContent(prevContent) : nextContent;
                    const fileNode: FileSystemNode = {
                      id: existingIndex >= 0 ? children[existingIndex].id : makeId('file'),
                      name: seg,
                      type: 'file',
                      path: targetPath,
                      content: resolved,
                      updatedAt: nowTs(),
                    };
                    if (existingIndex >= 0) children[existingIndex] = fileNode;
                    else children.push(fileNode);
                    return { ...folder, children, updatedAt: nowTs(), isOpen: folder.isOpen ?? true };
                  }

                  const folderPath = `${folder.path === '/' ? '' : folder.path}/${seg}`;
                  const existingFolderIndex = children.findIndex((c) => c.type === 'folder' && c.path === folderPath);
                  const nextFolder: FileSystemNode =
                    existingFolderIndex >= 0
                      ? children[existingFolderIndex]
                      : {
                          id: makeId('dir'),
                          name: seg,
                          type: 'folder',
                          path: folderPath,
                          isOpen: true,
                          updatedAt: nowTs(),
                          children: [],
                        };

                  const updatedFolder = upsertUnderFolder(nextFolder, idx + 1);
                  if (existingFolderIndex >= 0) children[existingFolderIndex] = updatedFolder;
                  else children.push(updatedFolder);
                  return { ...folder, children, updatedAt: nowTs(), isOpen: folder.isOpen ?? true };
                };

                const updatedRoot = upsertUnderFolder(root, 0);
                if (rootIndex >= 0) {
                  const next = [...tree];
                  next[rootIndex] = updatedRoot;
                  return next;
                }

                return [updatedRoot];
              };

        return upsert(prevFiles, path, content);
      });
    },
    [setFiles]
  );

  // Helper to get file content
  const getFileContent = (path: string): string => {
    const findNode = (nodes: FileSystemNode[]): string | null => {
      for (const node of nodes) {
        if (node.path === path) return node.content || '';
        if (node.children) {
          const found = findNode(node.children);
          if (found !== null) return found;
        }
      }
      return null;
    };
    return findNode(files) || '';
  };

  const { isRunning, runPipeline, completePipeline } = usePipelineRunner(files, updateFileContent);

  const trainingLogText = useMemo(() => {
    const findNode = (nodes: FileSystemNode[]): string | null => {
      for (const node of nodes) {
        if (node.path === '/logs/training.log') return node.content || '';
        if (node.children) {
          const found = findNode(node.children);
          if (found !== null) return found;
        }
      }
      return null;
    };
    return findNode(files) || '';
  }, [files]);

  const appendWsLog = (label: string, payload?: string) => {
    const time = new Date().toISOString().split('T')[1].slice(0, 8);
    const suffix = payload ? ` ${payload}` : '';
    updateFileContent('/logs/training.log', (prev) => (prev || '') + `[${time}] [WS] ${label}${suffix}\n`);
  };

  const [pinging, setPinging] = useState(false);
  const pingBackend = useCallback(async () => {
    const target = joinUrl(apiBase, `/api/test/emit/${projectId}`);
    appendWsLog('PING', target);
    setPinging(true);
    try {
      const res = await fetch(target, { method: 'POST' });
      const json = await res.json().catch(() => ({}));
      if (!res.ok) {
        appendWsLog('PING_FAIL', `${res.status} ${res.statusText}`);
        return;
      }
      appendWsLog('PING_OK', JSON.stringify(json));
    } catch (err) {
      appendWsLog('PING_FAIL', err instanceof Error ? err.message : String(err));
    } finally {
      setPinging(false);
    }
  }, [apiBase, appendWsLog, projectId]);

  const isBuildReady = !!session && session.status === 'ready';

  const patchSession = (patch: Partial<BuildSession>) => {
    setSession((prev) => (prev ? { ...prev, ...patch } : prev));
  };

  const aiReplyTimerRef = useRef<number | null>(null);


  const appendUserLog = (text: string) => {
    const time = new Date().toISOString().split('T')[1].slice(0, 8);
    updateFileContent('/logs/training.log', (prev) => (prev || '') + `[${time}] [USER] ${text}\n`);
  };

  const appendAiLog = (text: string) => {
    const time = new Date().toISOString().split('T')[1].slice(0, 8);
    updateFileContent('/logs/training.log', (prev) => (prev || '') + `[${time}] [AI] ${text}\n`);
  };

  // Log backend WebSocket events for visibility
  useEffect(() => {
    if (!lastEvent) return;
    const tag = lastEvent.event?.name ?? lastEvent.type ?? 'EVENT';
    const payloadPreview = lastEvent.event?.payload ? JSON.stringify(lastEvent.event.payload).slice(0, 300) : '';
    appendWsLog(tag, payloadPreview);
  }, [lastEvent]);

  const handleSendChangeRequest = (text: string) => {
    if (!text.trim()) return;
    const trimmed = text.trim();
    const now = new Date().toISOString();

    const userMessage: ChatMessage = {
      id: `msg_${Date.now()}_${Math.random().toString(16).slice(2)}`,
      role: 'user',
      text: trimmed,
      at: now,
    };

    appendUserLog(trimmed);
    patchSession({
      lastUserMessage: trimmed,
      lastUserMessageAt: now,
      aiThinking: true,
      chatHistory: [
        ...((session?.chatHistory ?? []) as ChatMessage[]),
        userMessage,
      ],
    });

    if (aiReplyTimerRef.current) window.clearTimeout(aiReplyTimerRef.current);
    aiReplyTimerRef.current = window.setTimeout(() => {
      const reply =
        "Got it. I’ll take that into account. If you want, open + to adjust constraints or add datasets.";
      const at = new Date().toISOString();
      const aiMessage: ChatMessage = {
        id: `msg_${Date.now()}_${Math.random().toString(16).slice(2)}`,
        role: 'ai',
        text: reply,
        at,
      };
      appendAiLog(reply);
      setSession((prev) => {
        if (!prev) return prev;
        const nextHistory = [
          ...((prev.chatHistory ?? []) as ChatMessage[]),
          aiMessage,
        ];
        return { ...prev, aiThinking: false, chatHistory: nextHistory };
      });
    }, 900);
  };

  useEffect(() => {
    return () => {
      if (aiReplyTimerRef.current) window.clearTimeout(aiReplyTimerRef.current);
    };
  }, []);

  // Tabs state
  const [tabs, setTabs] = useLocalStorageState<Tab[]>('workspaceTabs', [
    { id: 'preview', type: 'preview', title: 'Preview', closable: false },
    { id: 'dashboard', type: 'dashboard', title: 'Dashboard', closable: false },
    { id: 'publishing', type: 'publishing', title: 'Publishing', closable: false },
    { id: 'console', type: 'console', title: 'Console', closable: false },
  ]);
  
  const [activeTabId, setActiveTabId] = useLocalStorageState('activeTabId', 'preview');
  const [quickSwitcherOpen, setQuickSwitcherOpen] = useState(false);

  // Auto-switch to preview when running
  const handleRunPipeline = () => {
      setActiveTabId('preview');
      // If user reruns after READY, we show BUILDING state again.
      if (session && session.status === 'ready') {
        setSession({ ...session, status: 'building' });
      }
      runPipeline();
  };

  const handleGenerateData = () => {
    // Frontend-only mock: write a "generated" artifact and log.
    const payload = {
      generated_at: new Date().toISOString(),
      rows: 512,
      schema: {
        text: 'string',
        age: 'number',
        tenure: 'number',
      },
    };
    updateFileContent('/artifacts/generated_data.json', JSON.stringify(payload, null, 2));
    updateFileContent('/logs/training.log', (prev) => (prev || '') + `[${new Date().toISOString().split('T')[1].slice(0, 8)}] [INFO] Generated mock dataset artifact\n`);
  };

  const handleExportModel = () => {
    const content = getFileContent('/artifacts/model.json') || getFileContent('/config/model.json');
    const blob = new Blob([content || '{}'], { type: 'application/json' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'model.json';
    document.body.appendChild(a);
    a.click();
    a.remove();
    window.URL.revokeObjectURL(url);
  };

  // Handle file selection
  const handleFileSelect = (node: FileSystemNode) => {
    if (node.type !== 'file') return;

    const existingTab = tabs.find((tab) => tab.filePath === node.path);
    if (existingTab) {
      setActiveTabId(existingTab.id);
    } else {
      const newTab: Tab = {
        id: `file-${node.id}`,
        type: 'file',
        title: node.name,
        filePath: node.path,
        closable: true,
      };
      setTabs([...tabs, newTab]);
      setActiveTabId(newTab.id);
    }
  };

  const handleTabClose = (tabId: string) => {
    const newTabs = tabs.filter((tab) => tab.id !== tabId);
    setTabs(newTabs);
    if (activeTabId === tabId && newTabs.length > 0) {
      setActiveTabId(newTabs[0].id);
    }
  };

  // Keyboard shortcuts
  useKeyboardShortcuts({
    'ctrl+`': () => setActiveTabId('console'),
    'ctrl+p': () => setQuickSwitcherOpen(true),
    'ctrl+b': () => setLeftCollapsed(!leftCollapsed),
    'ctrl+e': () => setRightCollapsed(!rightCollapsed),
  });

  const activeTab = tabs.find((tab) => tab.id === activeTabId);

  const renderActivePane = () => {
    if (!activeTab) return null;

    switch (activeTab.type) {
      case 'preview':
        return <PreviewPane 
            files={files} 
            isRunning={isRunning} 
            onSimulationComplete={() => {
              completePipeline();
              if (session) setSession({ ...session, status: 'ready' });
            }}
            updateFileContent={updateFileContent}
            hasSession={!!session}
            sessionStatus={session?.status ?? 'building'}
        />;
      case 'dashboard':
        return <DashboardPane files={files} />;
      case 'console':
        return <ConsolePane logsText={trainingLogText} />;
      case 'publishing':
        return <PublishingPane />;
      case 'file':
        if (activeTab.filePath) {
          const content = getFileContent(activeTab.filePath);
          return (
            <FileEditorPane
              filePath={activeTab.filePath}
              content={content}
              onChange={(value: string | undefined) => activeTab.filePath && updateFileContent(activeTab.filePath, value || '')}
            />
          );
        }
        return null;
      default:
        return null;
    }
  };

  // Ensure required VFS paths exist + sync session into VFS (AppShell is source of truth)
  useEffect(() => {
    updateFileContent('/logs/training.log', (prev) => prev || '');
    updateFileContent('/artifacts/loss.json', (prev) => prev || '[]');
    updateFileContent('/artifacts/accuracy.json', (prev) => prev || '[]');
    if (session) {
      updateFileContent('/sessions/current.json', JSON.stringify(session, null, 2));
    }
  }, [session, updateFileContent]);

  // Workspace guard: if session is cleared, return to Home.
  useEffect(() => {
    if (!session) navigate('/');
  }, [session, navigate]);

  // Auto-start build on entering workspace when session is BUILDING.
  useEffect(() => {
    if (!session) return;
    if (session.status !== 'building') return;
    if (isRunning) return;
    handleRunPipeline();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [session?.id, session?.status]);

  const isDark = useMemo(() => theme === 'midnight', [theme]);

  return (
    <div className="h-screen flex flex-col bg-replit-bg text-replit-text relative overflow-hidden">
      <div className="absolute inset-0 -z-10 pointer-events-none">
        <div className="home-bg-static" />
      </div>

      <TopBar
        isBuildReady={isBuildReady}
        isPipelineRunning={isRunning}
        onRun={handleRunPipeline}
        onGenerateData={handleGenerateData}
        onExportModel={handleExportModel}
        isDark={isDark}
        onToggleTheme={toggleTheme}
        connectionStatus={connectionStatus}
        onPingBackend={pingBackend}
        isPinging={pinging}
      />

      <div ref={containerRef} className="flex-1 flex overflow-hidden">
        {/* Left Panel - AI Builder */}
        <ResizablePanel
          width={sizes.left}
          side="left"
          isCollapsed={leftCollapsed}
          onResize={handlePointerDown}
          collapsedContent={
            <>
              <button
                onClick={() => setLeftCollapsed(false)}
                className="p-2 rounded-lg hover:bg-replit-surfaceHover/40 text-replit-textMuted"
                aria-label="Expand input panel"
                title="Expand (Ctrl+B)"
              >
                <PanelLeftOpen className="w-4 h-4" />
              </button>
              <div className="h-px w-8 bg-replit-border/60" />
              <div className="text-[10px] text-replit-textMuted rotate-90 whitespace-nowrap mt-6">Input</div>
            </>
          }
        >
          {session && (
            <AIBuilderPanel
              session={session}
              onCollapse={() => setLeftCollapsed(true)}
              onEditSession={() => navigate('/')}
              onUpdateSession={patchSession}
              onSendMessage={handleSendChangeRequest}
            />
          )}
        </ResizablePanel>

        {/* Center Workspace */}
        <div className="flex-1 flex flex-col min-w-0">
          <WorkspaceTabs
            tabs={tabs}
            activeTabId={activeTabId}
            onTabClick={setActiveTabId}
            onTabClose={handleTabClose}
          />
          <div className="flex-1 overflow-hidden">
            {/* Pass files key to force re-render if needed, though props change should handle it */}
            {renderActivePane()}
          </div>
        </div>

        {/* Right Panel - Files */}
        <ResizablePanel
          width={sizes.right}
          side="right"
          isCollapsed={rightCollapsed}
          onResize={handlePointerDown}
          collapsedContent={
            <>
              <button
                onClick={() => setRightCollapsed(false)}
                className="p-2 rounded-lg hover:bg-replit-surfaceHover/40 text-replit-textMuted"
                aria-label="Expand file explorer"
                title="Expand (Ctrl+E)"
              >
                <PanelRightOpen className="w-4 h-4" />
              </button>
              <div className="h-px w-8 bg-replit-border/60" />
              <button
                onClick={() => {
                  setRightCollapsed(false);
                  setActiveTabId('dashboard');
                }}
                className="p-2 rounded-lg hover:bg-replit-surfaceHover/40 text-replit-textMuted"
                aria-label="Open Dashboard"
                title="Dashboard"
              >
                <FolderOpen className="w-4 h-4" />
              </button>
            </>
          }
        >
          {/* We need to update FilesPanel to accept the new VFS structure if it differs,
              but for now passing handleFileSelect is key. 
              Ideally we pass 'files' prop to FilesPanel so it renders our live state instead of its internal mock */}
          <FilesPanel onFileSelect={handleFileSelect} files={files} onCollapse={() => setRightCollapsed(true)} />
        </ResizablePanel>
      </div>

      <QuickFileSwitcher
        isOpen={quickSwitcherOpen}
        onClose={() => setQuickSwitcherOpen(false)}
        onFileSelect={(node) => {
          handleFileSelect(node);
          setQuickSwitcherOpen(false);
        }}
      />
    </div>
  );
}
