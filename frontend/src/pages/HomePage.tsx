import { useEffect, useMemo, useState, type KeyboardEvent } from 'react';
import { motion } from 'framer-motion';
import {
  Database,
  Image as ImageIcon,
  Brain,
  Mic,
  Moon,
  Paperclip,
  Plus,
  Send,
  Sparkles,
  Sun,
} from 'lucide-react';

import {
  createBuildSession,
  isValidHuggingFaceDatasetLink,
  setCurrentSession,
} from '../lib/buildSession';
import { useTheme } from '../lib/theme';
import { useRouter } from '../router/router';
import MatrixScreenLoader from '../components/MatrixScreenLoader';

export default function HomePage() {
  const { navigate } = useRouter();
  const { toggleTheme, theme } = useTheme();

  const isDarkMode = theme === 'midnight';

  const [message, setMessage] = useState('');
  const [datasetLink, setDatasetLink] = useState('');
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });
  const [scrollY, setScrollY] = useState(0);
  const [isStarting, setIsStarting] = useState(false);

  const datasetOk = useMemo(
    () => !datasetLink || isValidHuggingFaceDatasetLink(datasetLink),
    [datasetLink]
  );

  const canStart = message.trim().length > 0;

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      setMousePosition({ x: e.clientX, y: e.clientY });
    };
    const handleScroll = () => {
      setScrollY(window.scrollY);
    };

    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('scroll', handleScroll);
    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('scroll', handleScroll);
    };
  }, []);

  const onStart = () => {
    if (!canStart) return;

    setIsStarting(true);
    const session = createBuildSession({
      modelName: '',
      goalPrompt: message,
      kaggleLink: datasetLink,
    });
    setCurrentSession(session);
    navigate('/workspace');
  };

  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      onStart();
    }
  };

  const dropdownOptions = [
    {
      icon: <Database className="w-5 h-5" />,
      label: 'Hugging Face dataset link',
      color: 'text-orange-400',
      kind: 'dataset' as const,
    },
    {
      icon: <ImageIcon className="w-5 h-5" />,
      label: 'Upload Image',
      color: 'text-blue-400',
      kind: 'noop' as const,
    },
    {
      icon: <Paperclip className="w-5 h-5" />,
      label: 'Attach File',
      color: 'text-green-400',
      kind: 'noop' as const,
    },
    {
      icon: <Mic className="w-5 h-5" />,
      label: 'Voice Input',
      color: 'text-purple-400',
      kind: 'noop' as const,
    },
  ];

  return (
    <div
      className={`min-h-screen overflow-hidden relative flex flex-col transition-colors duration-500 ${
        isDarkMode ? 'bg-slate-950 text-white' : 'bg-gray-50 text-gray-900'
      }`}
    >
      {isStarting && <MatrixScreenLoader label="Starting build…" />}

      {/* Mouse-follow gradient */}
      <div
        className={`fixed inset-0 transition-opacity duration-500 ${
          isDarkMode ? 'opacity-30' : 'opacity-20'
        }`}
        style={{
          background: `radial-gradient(circle at ${mousePosition.x}px ${mousePosition.y}px, rgba(59, 130, 246, 0.28), transparent 50%)`,
        }}
      />

      {/* Floating orbs */}
      <div
        className={`fixed top-20 left-20 w-64 h-64 bg-blue-500 rounded-full mix-blend-multiply filter blur-3xl animate-pulse ${
          isDarkMode ? 'opacity-20' : 'opacity-10'
        }`}
      />
      <div
        className={`fixed top-40 right-20 w-96 h-96 bg-sky-500 rounded-full mix-blend-multiply filter blur-3xl animate-pulse ${
          isDarkMode ? 'opacity-20' : 'opacity-10'
        }`}
        style={{ animationDelay: '1s' }}
      />
      <div
        className={`fixed bottom-20 left-1/2 w-80 h-80 bg-indigo-500 rounded-full mix-blend-multiply filter blur-3xl animate-pulse ${
          isDarkMode ? 'opacity-20' : 'opacity-10'
        }`}
        style={{ animationDelay: '2s' }}
      />

      {/* Navigation */}
      <nav className="relative z-50 px-4 py-4">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-2 group cursor-pointer select-none">
            <div className="relative">
              <Brain className="w-8 h-8 text-sky-400 transition-transform group-hover:rotate-12" />
              <div className="absolute inset-0 bg-sky-400 blur-xl opacity-40 group-hover:opacity-65 transition-opacity" />
            </div>
            <span className="text-2xl font-bold bg-gradient-to-r from-sky-400 to-blue-500 bg-clip-text text-transparent">
              AIAI
            </span>
          </div>

          <button
            onClick={toggleTheme}
            className={`p-3 rounded-xl transition-all duration-300 transform hover:scale-110 ${
              isDarkMode
                ? 'bg-slate-800/50 hover:bg-slate-700/50'
                : 'bg-white/50 hover:bg-white/80'
            } backdrop-blur-sm border ${
              isDarkMode ? 'border-blue-500/30' : 'border-blue-300/50'
            }`}
            aria-label="Toggle theme"
          >
            {isDarkMode ? (
              <Sun className="w-6 h-6 text-yellow-400" />
            ) : (
              <Moon className="w-6 h-6 text-purple-600" />
            )}
          </button>
        </div>
      </nav>

      {/* Hero */}
      <main className="relative z-10 flex-1 flex flex-col items-center justify-center">
        <div className="mx-auto w-full max-w-7xl px-6 py-12">
          <div
            className="text-center transform transition-all duration-1000"
            style={{
              transform: `translateY(${scrollY * 0.2}px)`,
              opacity: 1 - scrollY / 500,
            }}
          >
            <div
              className={`inline-block mb-6 px-4 py-2 rounded-full text-sm font-medium backdrop-blur-sm animate-pulse ${
                isDarkMode
                  ? 'bg-blue-500/15 border border-blue-500/25 text-blue-200'
                  : 'bg-blue-500/10 border border-blue-400/30 text-blue-700'
              }`}
            >
              <Sparkles className="w-4 h-4 inline mr-2" />
              Powered by Advanced AI
            </div>

            <h1 className="text-6xl md:text-8xl font-bold mb-4 bg-gradient-to-r from-sky-400 via-blue-400 to-indigo-400 bg-clip-text text-transparent animate-gradient">
              Hey, Mohamed.
            </h1>

            <p
              className={`text-xl md:text-2xl max-w-3xl mx-auto transition-colors duration-500 ${
                isDarkMode ? 'text-gray-400' : 'text-gray-600'
              }`}
            >
              Ready to dive in?
            </p>
          </div>

          <div className="mt-10 w-full max-w-4xl mx-auto">
            <motion.div
            whileHover={{ scale: 1.01 }}
            transition={{ duration: 0.25, ease: 'easeOut' }}
            className={`relative backdrop-blur-xl rounded-3xl border shadow-2xl p-2 transition-all duration-300 ${
              isDarkMode
                ? 'bg-slate-800/50 border-blue-500/25 hover:border-blue-500/45'
                : 'bg-white/80 border-blue-300/50 hover:border-blue-400/60'
            }`}
            style={{
              boxShadow: canStart
                ? '0 0 0 1px rgba(59, 130, 246, 0.22), 0 0 70px rgba(14, 165, 233, 0.10)'
                : '0 0 0 1px rgba(59, 130, 246, 0.12), 0 0 55px rgba(14, 165, 233, 0.06)',
            }}
          >
            <div className="flex items-center gap-2">
              {/* Plus dropdown */}
              <div className="relative">
                <button
                  onClick={() => setIsDropdownOpen((v) => !v)}
                  className={`p-3 rounded-xl transition-all duration-300 transform hover:scale-110 group ${
                    isDarkMode ? 'hover:bg-slate-700/50' : 'hover:bg-gray-100'
                  }`}
                  aria-label="Open attachments"
                >
                  <Plus
                    className={`w-6 h-6 transition-all duration-300 ${
                      isDropdownOpen ? 'rotate-45' : ''
                    } ${
                      isDarkMode
                        ? 'text-gray-400 group-hover:text-blue-400'
                        : 'text-gray-600 group-hover:text-blue-600'
                    }`}
                  />
                </button>

                {isDropdownOpen && (
                  <div
                    className={`absolute bottom-full left-0 mb-2 w-80 backdrop-blur-xl border rounded-2xl shadow-2xl overflow-hidden ${
                      isDarkMode
                        ? 'bg-slate-800/95 border-blue-500/25'
                        : 'bg-white/95 border-blue-300/50'
                    }`}
                  >
                    {dropdownOptions.map((option) => (
                      <div key={option.label}>
                        <button
                          onClick={() => {
                            if (option.kind !== 'dataset') {
                              setIsDropdownOpen(false);
                            }
                          }}
                          className={`w-full flex items-center gap-3 px-4 py-3 transition-all duration-200 group ${
                            isDarkMode
                              ? 'hover:bg-slate-700/50'
                              : 'hover:bg-gray-100'
                          }`}
                        >
                          <span
                            className={`${option.color} group-hover:scale-110 transition-transform duration-200`}
                          >
                            {option.icon}
                          </span>
                          <span
                            className={`transition-colors flex-1 text-left ${
                              isDarkMode
                                ? 'text-gray-300 group-hover:text-white'
                                : 'text-gray-700 group-hover:text-gray-900'
                            }`}
                          >
                            {option.label}
                          </span>
                        </button>

                        {option.kind === 'dataset' && (
                          <div className="px-4 pb-4">
                            <input
                              value={datasetLink}
                              onChange={(e) => setDatasetLink(e.target.value)}
                              placeholder="https://huggingface.co/datasets/owner/name"
                              className={`w-full text-sm px-3 py-2 rounded-xl border outline-none bg-transparent transition ${
                                isDarkMode
                                  ? 'border-slate-600 text-white placeholder-slate-500 focus:border-blue-500'
                                  : 'border-gray-200 text-gray-900 placeholder-gray-400 focus:border-blue-500'
                              } ${
                                datasetOk
                                  ? ''
                                  : 'border-yellow-500 focus:border-yellow-500'
                              }`}
                            />
                            <div
                              className={`mt-2 text-xs ${
                                isDarkMode ? 'text-gray-500' : 'text-gray-500'
                              }`}
                            >
                              Leave blank to auto-select a dataset
                            </div>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                )}
              </div>

              {/* Text input */}
              <textarea
                value={message}
                onChange={(e) => setMessage(e.target.value)}
                onKeyDown={onKeyDown}
                placeholder="Ask anything…"
                className={`flex-1 bg-transparent outline-none resize-none py-2.5 px-2 leading-6 max-h-32 min-h-[3rem] transition-colors duration-500 ${
                  isDarkMode
                    ? 'text-white placeholder-gray-500'
                    : 'text-gray-900 placeholder-gray-400'
                }`}
                rows={1}
              />

              {/* Inline loader (doesn't block page) */}
              <div className="flex items-center justify-center h-[3rem] pr-1">
                <MatrixScreenLoader
                  variant="inline"
                  label={undefined}
                  className={isStarting ? 'opacity-90' : 'opacity-40'}
                />
              </div>

              {/* Send */}
              <button
                onClick={onStart}
                disabled={!canStart}
                className={`p-3 rounded-xl transition-all duration-300 transform ${
                  canStart
                    ? 'bg-gradient-to-r from-sky-500 to-blue-600 hover:shadow-lg hover:shadow-blue-500/50 hover:scale-110 text-white'
                    : isDarkMode
                      ? 'bg-slate-700/50 cursor-not-allowed opacity-50 text-gray-400'
                      : 'bg-gray-200/70 cursor-not-allowed opacity-70 text-gray-500'
                }`}
                aria-label="Start"
              >
                <Send className="w-6 h-6" />
              </button>
            </div>
            </motion.div>

            <p
              className={`text-center text-sm mt-4 transition-colors duration-500 ${
                isDarkMode ? 'text-gray-500' : 'text-gray-400'
              }`}
            >
              Press Enter to start, Shift + Enter for new line
            </p>
          </div>
        </div>
      </main>

      <style>{`
        @keyframes gradient {
          0%, 100% { background-position: 0% 50%; }
          50% { background-position: 100% 50%; }
        }
        .animate-gradient {
          background-size: 200% 200%;
          animation: gradient 3s ease infinite;
        }
      `}</style>
    </div>
  );
}